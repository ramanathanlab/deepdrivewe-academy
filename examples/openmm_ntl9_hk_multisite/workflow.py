"""OpenMM NTL9 Huber-Kim workflow configs and agents.

Defines all classes that Parsl workers must deserialize.
These live in a dedicated module (not ``__main__``) so that
dill can resolve them on the worker side.
"""

from __future__ import annotations

from pathlib import Path

import MDAnalysis
from academy.handle import Handle
from MDAnalysis.analysis import rms
from pydantic import Field
from pydantic import field_validator

from deepdrivewe.api import BaseModel
from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import validate_and_resolve_file
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.parsl import ComputeConfigTypes
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import HuberKimResampler
from deepdrivewe.simulation.openmm import ContactMapRMSDReporter
from deepdrivewe.simulation.openmm import OpenMMConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation
from deepdrivewe.workflows.westpa import SimulationAgent
from deepdrivewe.workflows.westpa import WestpaAgent

# --- Configuration ---------------------------------------------------


class SimulationConfig(BaseModel):
    """Configuration for the OpenMM simulation."""

    openmm_config: OpenMMConfig = Field(
        description='The configuration for the OpenMM simulation.',
    )
    top_file: Path | None = Field(
        default=None,
        description='The topology file for the simulation.',
    )
    reference_file: Path = Field(
        description='The reference PDB file for RMSD analysis.',
    )
    cutoff_angstrom: float = Field(
        default=8.0,
        description='Cutoff distance for defining contacts.',
    )
    mda_selection: str = Field(
        default='protein and name CA',
        description='MDAnalysis selection for atoms to use.',
    )
    openmm_selection: list[str] = Field(
        default=['CA'],
        description='OpenMM atom selection strings.',
    )

    @field_validator('top_file', 'reference_file')
    @classmethod
    def resolve_file(cls, value: Path | None) -> Path | None:
        """Validate and resolve the file path."""
        return validate_and_resolve_file(value)


class InferenceConfig(BaseModel):
    """Configuration for the resampler."""

    sims_per_bin: int = Field(
        default=5,
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


class RMSDBasisStateInitializer(BaseModel):
    """Compute initial pcoords via RMSD to reference."""

    reference_file: Path = Field(
        description='Reference PDB for RMSD computation.',
    )
    mda_selection: str = Field(
        default='protein and name CA',
        description='MDAnalysis selection for atoms.',
    )

    @field_validator('reference_file')
    @classmethod
    def resolve_file(cls, value: Path | None) -> Path | None:
        """Validate and resolve the file path."""
        return validate_and_resolve_file(value)

    def __call__(self, basis_file: str) -> list[float]:
        """Compute RMSD between basis and reference."""
        basis = MDAnalysis.Universe(basis_file)
        reference = MDAnalysis.Universe(self.reference_file)
        pos = basis.select_atoms(self.mda_selection).positions
        ref_pos = reference.select_atoms(self.mda_selection).positions
        rmsd: float = rms.rmsd(pos, ref_pos, superposition=True)
        return [rmsd]


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
    basis_state_initializer: RMSDBasisStateInitializer = Field(
        description='Initializer for basis state pcoords.',
    )
    target_states: list[TargetState] = Field(
        description='Target states for recycling.',
    )
    simulation_config: SimulationConfig = Field(
        description='Simulation configuration.',
    )
    inference_config: InferenceConfig = Field(
        description='Inference/resampling configuration.',
    )
    compute_config: ComputeConfigTypes = Field(
        description='Compute configuration for running simulations.',
    )

    @field_validator('output_dir')
    @classmethod
    def mkdir_validator(cls, value: Path) -> Path:
        """Resolve and create the output directory."""
        value = value.resolve()
        value.mkdir(parents=True, exist_ok=True)
        return value


# --- Agent Subclasses ------------------------------------------------


class OpenMMSimAgent(SimulationAgent):
    """Run OpenMM MD simulations."""

    def __init__(
        self,
        westpa_handle: Handle[WestpaAgent],
        sim_config: SimulationConfig,
        output_dir: Path,
        logfile: Path | None = None,
    ) -> None:
        super().__init__(westpa_handle, logfile=logfile)
        self.sim_config = sim_config
        self.output_dir = output_dir

    def run_simulation(self, metadata: SimMetadata) -> SimResult:
        """Run an OpenMM simulation."""
        metadata.mark_simulation_start()

        sim_output_dir = self.output_dir / metadata.simulation_name
        if sim_output_dir.exists():
            for f in sim_output_dir.iterdir():
                f.unlink()
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        self.sim_config.dump_yaml(sim_output_dir / 'config.yaml')

        simulation = OpenMMSimulation(
            config=self.sim_config.openmm_config,
            top_file=self.sim_config.top_file,
            output_dir=sim_output_dir,
            checkpoint_file=metadata.parent_restart_file,
        )

        reporter = ContactMapRMSDReporter(
            report_interval=self.sim_config.openmm_config.report_steps,
            reference_file=self.sim_config.reference_file,
            cutoff_angstrom=self.sim_config.cutoff_angstrom,
            mda_selection=self.sim_config.mda_selection,
            openmm_selection=self.sim_config.openmm_selection,
        )

        simulation.run(reporters=[reporter])

        contact_maps = reporter.get_contact_maps()
        pcoord = reporter.get_rmsds()

        metadata.restart_file = simulation.restart_file
        metadata.pcoord = pcoord.tolist()
        metadata.mark_simulation_end()

        return SimResult(
            data={
                'contact_maps': contact_maps,
                'pcoord': pcoord,
            },
            metadata=metadata,
        )


class HuberKimWestpaAgent(WestpaAgent):
    """WESTPA agent with Huber-Kim resampling."""

    def __init__(
        self,
        simulation_handles: list[Handle[SimulationAgent]],
        max_iterations: int,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer | None = None,
        inference_config: InferenceConfig | None = None,
        logfile: Path | None = None,
    ) -> None:
        super().__init__(
            simulation_handles=simulation_handles,
            max_iterations=max_iterations,
            ensemble=ensemble,
            checkpointer=checkpointer,
            logfile=logfile,
        )
        self.inference_config = inference_config or InferenceConfig()

    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Apply binning, recycling, and Huber-Kim resampling."""
        pcoords = [r.metadata.pcoord[-1] for r in sim_results]
        self.logger.info(
            f'pcoords: best={min(pcoords)}, n_sims={len(sim_results)}',
        )

        cur_sims = [r.metadata for r in sim_results]
        cfg = self.inference_config

        binner = RectilinearBinner(
            bins=[0.0, 1.00]
            + [1.10 + 0.1 * i for i in range(35)]
            + [4.60 + 0.2 * i for i in range(10)]
            + [6.60 + 0.6 * i for i in range(6)]
            + [float('inf')],
            bin_target_counts=cfg.sims_per_bin,
            target_state_inds=0,
        )

        assert self.basis_states is not None
        recycler = LowRecycler(
            basis_states=self.basis_states,
            target_threshold=self.target_states[0].pcoord[0],
        )

        resampler = HuberKimResampler(
            sims_per_bin=cfg.sims_per_bin,
            max_allowed_weight=cfg.max_allowed_weight,
            min_allowed_weight=cfg.min_allowed_weight,
        )

        return resampler.run(cur_sims, binner, recycler)
