"""OpenMM NTL9 DDWE workflow configs and agents.

Defines all classes that Parsl workers must deserialize.
These live in a dedicated module (not ``__main__``) so that
dill can resolve them on the worker side.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from academy.exchange.cloud.client import HttpExchangeTransport
from pydantic import Field
from pydantic import field_validator
from sklearn.neighbors import LocalOutlierFactor

from deepdrivewe.ai.cvae import ConvolutionalVAE
from deepdrivewe.ai.cvae import ConvolutionalVAEConfig
from deepdrivewe.api import BaseModel
from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import TrainResult
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.parsl import WorkstationConfig
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers.lof import LOFLowResampler
from deepdrivewe.workflows.ddwe import InferenceAgent
from deepdrivewe.workflows.ddwe import TrainingAgent
from examples.openmm_ntl9_hk.workflow import RMSDBasisStateInitializer
from examples.openmm_ntl9_hk.workflow import SimulationConfig

# --- Academy Exchange Retry ------------------------------------------

_original_send = HttpExchangeTransport.send


async def _send_with_retry(
    self: HttpExchangeTransport,
    message: Any,
    _max_retries: int = 5,
    _base_delay: float = 1.0,
) -> Any:
    """Retry ``send`` with exponential backoff on 502/503/504."""
    for attempt in range(_max_retries + 1):
        try:
            return await _original_send(self, message)
        except aiohttp.ClientResponseError as e:
            if e.status in (502, 503, 504) and attempt < _max_retries:
                delay = _base_delay * (2**attempt)
                logging.getLogger('academy.exchange').warning(
                    f'Exchange returned {e.status}, '
                    f'retrying in {delay:.1f}s '
                    f'(attempt {attempt + 1}/{_max_retries})',
                )
                await asyncio.sleep(delay)
            else:
                raise


HttpExchangeTransport.send = _send_with_retry  # type: ignore[method-assign]


# --- Configuration ---------------------------------------------------


class TrainConfig(BaseModel):
    """Arguments for the training module."""

    config_path: Path | None = Field(
        default=None,
        description='The path to the model configuration file.',
    )

    checkpoint_path: Path | None = Field(
        default=None,
        description='The path to the model checkpoint file.'
        'Train from scratch by default.',
    )


class InferenceConfig(BaseModel):
    """Configuration for the resampler."""

    sims_per_bin: int = Field(
        default=72,
        description='Number of simulations per bin.',
    )
    max_allowed_weight: float = Field(
        default=1.0,
        description='Maximum allowed simulation weight.',
    )
    min_allowed_weight: float = Field(
        default=10e-40,
        description='Minimum allowed simulation weight.',
    )
    lof_n_neighbors: int = Field(
        default=20,
        description='Number of neighbors for LOF. '
        'Should not exceed sims_per_bin.',
    )
    lof_distance_metric: str = Field(
        default='cosine',
        description='Distance metric for LOF.',
    )
    consider_for_resampling: int = Field(
        default=12,
        description='Number of simulations to consider for resampling.',
    )
    max_resamples: int = Field(
        default=4,
        description='Maximum resamples per iteration.',
    )


class RMSDLOFBasisStateInitializer(RMSDBasisStateInitializer):
    """RMSD initializer with a placeholder LOF score for DDWE."""

    def __call__(self, basis_file: str) -> list[float]:
        """Compute RMSD and append an initial LOF placeholder."""
        rmsd = super().__call__(basis_file)
        return [*rmsd, -1.0]  # Second value is an initial value for LOF


class ExperimentSettings(BaseModel):
    """Full experiment configuration (YAML)."""

    output_dir: Path = Field(
        description='Directory to store results.',
    )
    num_iterations: int = Field(
        ge=1,
        description='Number of WE iterations.',
    )
    basis_states: BasisStates = Field(
        description='Basis states for the ensemble.',
    )
    basis_state_initializer: RMSDLOFBasisStateInitializer = Field(
        description='Initializer for basis state pcoords.',
    )
    target_states: list[TargetState] = Field(
        description='Target states for recycling.',
    )
    simulation_config: SimulationConfig = Field(
        description='Simulation configuration.',
    )
    train_config: TrainConfig | None = Field(
        default=None,
        description='Training configuration.',
    )
    inference_config: InferenceConfig = Field(
        description='Inference/resampling configuration.',
    )
    use_stale_model: bool = Field(
        default=False,
        description='Run inference against the previous iteration'
        "'s model so training overlaps the next round of "
        'simulations instead of blocking them.',
    )
    sim_compute_config: WorkstationConfig = Field(
        description='GPU config for simulations.',
    )
    ddwe_compute_config: WorkstationConfig = Field(
        description='GPU config for training and inference.',
    )

    @field_validator('output_dir')
    @classmethod
    def mkdir_validator(cls, value: Path) -> Path:
        """Resolve and create the output directory."""
        value = value.resolve()
        value.mkdir(parents=True, exist_ok=True)
        return value


# --- Agent Subclasses ------------------------------------------------


class CVAETrainAgent(TrainingAgent):
    """DDWE training agent using a CVAE."""

    def __init__(
        self,
        cvae_config: ConvolutionalVAEConfig | None = None,
        train_config: TrainConfig | None = None,
        output_dir: Path = Path('ddwe_output'),
    ) -> None:
        super().__init__()
        cvae_config = cvae_config or ConvolutionalVAEConfig()
        train_config = train_config or TrainConfig()
        if train_config.config_path is not None:
            cvae_config = ConvolutionalVAEConfig.from_yaml(
                train_config.config_path,
            )
        self.cvae_config = cvae_config
        self.checkpoint_path = train_config.checkpoint_path
        self.output_dir = output_dir

    def run_training(
        self,
        sim_results: list[SimResult],
    ) -> TrainResult:
        """Train the CVAE model using the simulation results."""
        # Make the output directory
        iteration = sim_results[0].metadata.iteration_id
        output_dir = self.output_dir / f'{iteration:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the last frame contact maps and rmsd from each simulation
        contact_maps = np.concatenate(
            [sim.data['contact_maps'] for sim in sim_results],
        )
        pcoords = np.concatenate([sim.data['pcoord'] for sim in sim_results])
        pcoords = pcoords.flatten()

        # Fit the model
        cvae_config = self.cvae_config.model_copy(update={'device': 'cuda:0'})
        model = ConvolutionalVAE(cvae_config, self.checkpoint_path)

        self.checkpoint_path = model.fit(
            x=contact_maps,
            model_dir=output_dir / 'model',
            scalars={'pcoords': pcoords},
        )

        # Return the train result
        result = TrainResult(
            config_path=output_dir / 'model',
            checkpoint_path=self.checkpoint_path,
        )

        return result


class CVAEInferAgent(InferenceAgent):
    """DDWE inference agent using a CVAE."""

    def __init__(
        self,
        cvae_config: ConvolutionalVAEConfig | None = None,
        inference_config: InferenceConfig | None = None,
        output_dir: Path = Path('ddwe_output'),
    ) -> None:
        super().__init__()
        self.cvae_config = cvae_config or ConvolutionalVAEConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.output_dir = output_dir

    def run_inference(
        self,
        sim_results: list[SimResult],
        train_result: TrainResult,
        basis_states: BasisStates,
        target_states: list[TargetState],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Run inference/resampling using the trained CVAE model."""
        # Make the output directory
        iteration = sim_results[0].metadata.iteration_id
        output_dir = self.output_dir / f'{iteration:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get RMSD progress coordinate
        pcoords = [r.metadata.pcoord[-1] for r in sim_results]
        self.logger.info(
            f'pcoords: best={min(pcoords)}, n_sims={len(sim_results)}',
        )

        cur_sims = [r.metadata for r in sim_results]
        config = self.inference_config

        # Build contact maps and use model for inference
        contact_maps = np.concatenate(
            [sim.data['contact_maps'] for sim in sim_results],
        )

        contact_maps = [x.astype(np.int16) for x in contact_maps]

        cvae_config = self.cvae_config.model_copy(update={'device': 'cuda:0'})
        model = ConvolutionalVAE(cvae_config, train_result.checkpoint_path)
        # Latent-space coordinates for each simulation
        z = model.predict(contact_maps)

        # Run LOF on the latent space
        clf = LocalOutlierFactor(
            n_neighbors=config.lof_n_neighbors,
            metric=config.lof_distance_metric,
        ).fit(z)

        # Get the LOF scores
        lof_scores = clf.negative_outlier_factor_

        # Add the LOF scores to the last frame of each simulation pcoord
        for sim, score in zip(
            cur_sims,
            lof_scores[-len(cur_sims) :],
            strict=False,
        ):
            sim_scores = [-1.0 for _ in range(sim.num_frames)]
            sim_scores[-1] = float(score)
            sim.append_pcoord(sim_scores)

        # Create the binner
        binner = RectilinearBinner(
            bins=[0.0, 1.0, float('inf')],
            bin_target_counts=config.sims_per_bin,
        )

        # Define the recycling policy
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=target_states[0].pcoord[0],
        )

        # Define the resampling policy
        resampler = LOFLowResampler(
            consider_for_resampling=config.consider_for_resampling,
            max_resamples=config.max_resamples,
            max_allowed_weight=config.max_allowed_weight,
            min_allowed_weight=config.min_allowed_weight,
        )

        # Assign simulations to bins and resample the weighted ensemble
        result = resampler.run(cur_sims, binner, recycler)

        return result
