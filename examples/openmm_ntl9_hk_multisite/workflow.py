"""OpenMM NTL9 Huber-Kim workflow configs and agents.

Defines all classes that Parsl workers must deserialize.
These live in a dedicated module (not ``__main__``) so that
dill can resolve them on the worker side.
"""

from __future__ import annotations

import os
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
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import HuberKimResampler
from deepdrivewe.simulation.openmm import ContactMapRMSDReporter
from deepdrivewe.simulation.openmm import OpenMMConfig
from deepdrivewe.simulation.openmm import OpenMMSimulation
from deepdrivewe.workflows.westpa import SimulationAgent
from deepdrivewe.workflows.westpa import WestpaAgent

# --- Configuration ---------------------------------------------------


class SimulationConfig(BaseModel):
    """Configuration for the OpenMM simulation.

    Notes
    -----
    The simulation agent changes its working directory to
    ``base_dir`` on startup, so relative paths (``reference_file``,
    ``top_file``, etc.) resolve correctly without explicit
    resolution.
    """

    base_dir: Path = Field(
        description=(
            'Absolute path to the example directory on the sim host. '
            'The agent chdirs here on startup so relative paths '
            'resolve correctly.'
        ),
    )
    openmm_config: OpenMMConfig = Field(
        description='The configuration for the OpenMM simulation.',
    )
    top_file: Path | None = Field(
        default=None,
        description='Topology file path (on the simulation host).',
    )
    reference_file: Path = Field(
        description='Reference PDB path (on the simulation host).',
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


class InferenceConfig(BaseModel):
    """Configuration for the resampler.

    Notes
    -----
    ``base_dir`` serves the same purpose as
    ``SimulationConfig.base_dir`` but for the **inference** endpoint.
    When the WESTPA agent runs on a remote Globus Compute endpoint
    the worker's cwd is not the example directory, so relative paths
    (checkpointer output, logfile, etc.) would resolve incorrectly.
    ``base_dir`` is passed to the agent and used to ``os.chdir``
    before any path-dependent work begins.
    """

    base_dir: Path | None = Field(
        default=None,
        description=(
            'Absolute path to the example directory on the '
            'inference host. The WESTPA agent chdirs here on '
            'startup so relative paths resolve correctly. '
            'None keeps the worker cwd unchanged (fine for '
            'local / single-site runs).'
        ),
    )
    output_dir: Path = Field(
        default=Path('results/'),
        description=(
            'Directory for checkpoints and logs on the inference '
            'host. Resolved after chdir to base_dir.'
        ),
    )
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


class GlobusComputeConfig(BaseModel):
    """Configuration for the remote Globus Compute endpoints.

    Two endpoints are used:

    - ``simulation_endpoint_id`` runs OpenMM simulation agents and
      should be a GPU-equipped endpoint (e.g. LocalProvider with
      ``available_accelerators`` set on the engine).
    - ``inference_endpoint_id`` runs the WESTPA/Huber-Kim resampling
      agent and only needs a CPU worker. It can point at the same
      physical host as the simulation endpoint (recommended, so
      checkpoint/output paths resolve consistently) or at a different
      host that shares the ``output_dir`` filesystem.

    Both endpoints must be pre-configured and running on their hosts.
    See the README for deployment instructions.
    """

    simulation_endpoint_id: str = Field(
        description='Endpoint UUID for OpenMM simulation agents (GPU).',
    )
    inference_endpoint_id: str | None = Field(
        default=None,
        description=(
            'Endpoint UUID for the WESTPA agent (CPU). If omitted, '
            'the agent runs locally on the orchestrator via a '
            'ThreadPoolExecutor.'
        ),
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
        return value
        # return validate_and_resolve_file(value)

    def __call__(self, basis_file: str) -> list[float]:
        """Compute RMSD between basis and reference."""
        basis = MDAnalysis.Universe(basis_file)
        reference = MDAnalysis.Universe(self.reference_file)
        pos = basis.select_atoms(self.mda_selection).positions
        ref_pos = reference.select_atoms(self.mda_selection).positions
        rmsd: float = rms.rmsd(pos, ref_pos, superposition=True)
        return [rmsd]


class ExperimentSettings(BaseModel):
    """Full experiment configuration (YAML).

    Path handling
    -------------
    - ``output_dir`` is resolved locally on the **orchestrator** host
      (checkpoints, ``params.yaml``, ``runtime.log``).
    - ``sim_output_dir`` is passed through verbatim to the simulation
      agent and is resolved on the **simulation** host (HPC). For
      ``--exchange globus`` use an absolute path that exists on the
      endpoint; for ``--exchange local`` a relative path works.
    """

    output_dir: Path = Field(
        description='Directory for orchestrator outputs (local).',
    )
    sim_output_dir: Path = Field(
        description='Directory for simulation outputs (sim host).',
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
    globus_compute: GlobusComputeConfig = Field(
        description='Globus Compute endpoint + exchange settings.',
    )

    @field_validator('output_dir')
    @classmethod
    def mkdir_validator(cls, value: Path) -> Path:
        """Resolve and create the orchestrator output directory."""
        # value = value.resolve()
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

    async def agent_on_startup(self) -> None:
        """Set the working directory for path resolution on the sim host.

        When running on a remote Globus Compute endpoint the worker's cwd
        is not the example directory, so relative paths (e.g. checkpoint files,
        output_dir) would resolve incorrectly. Changing to ``base_dir`` first
        ensures they land in the right place.
        """
        await super().agent_on_startup()
        if self.sim_config.base_dir is not None:
            os.chdir(self.sim_config.base_dir)

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

    async def agent_on_startup(self) -> None:
        """Set the working directory and initialize the checkpointer.

        When running on a remote Globus Compute endpoint the
        worker's cwd is not the example directory, so relative
        paths (checkpointer output, logfile, etc.) would resolve
        incorrectly. Changing to ``base_dir`` first ensures they
        land in the right place — same rationale as
        ``SimulationConfig.base_dir`` for the sim agent.

        The checkpointer is created here (not on the orchestrator)
        because it writes checkpoints and HDF5 files to paths
        that must resolve on the inference host.
        """
        if self.inference_config.base_dir is not None:
            os.chdir(self.inference_config.base_dir)

        # Create the checkpointer on the inference host where the
        # output_dir path resolves correctly.
        output_dir = self.inference_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpointer = EnsembleCheckpointer(output_dir=output_dir)

        # Resume from the latest checkpoint if one exists on this
        # host.
        # TODO: On resume the orchestrator still dispatches fresh
        # initial_sims (from a seed ensemble) which may not match
        # the checkpoint's next_sims. Full multi-site resume
        # requires the agent to re-dispatch the correct walkers.
        checkpoint = self.checkpointer.latest_checkpoint()
        if checkpoint is not None:
            self.ensemble = self.checkpointer.load(checkpoint)

        await super().agent_on_startup()

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
