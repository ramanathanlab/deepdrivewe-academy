---
name: deepdrivewe-example
description: Use this skill when building a new example or custom workflow with DeepDriveWE (weighted ensemble MD on the Academy multi-agent framework). Covers project layout, agent subclassing, Pydantic config models, binner/recycler/resampler selection, Parsl compute configs, and verification steps.
---

# Building a New DeepDriveWE Example

This skill guides Claude through creating a new example under `examples/`
that runs a weighted ensemble workflow with DeepDriveWE. Use it when
the user asks for a new example, a new protein system, a new MD engine
wrapper, or any custom DDW workflow.

## Before You Start

Gather these requirements from the user:

1. **System / use case** -- folding protein? Ligand binding? Conformational
   sampling? Custom synthetic system?
2. **MD engine** -- OpenMM, AMBER, or a custom/synthetic engine?
3. **Progress coordinate** -- RMSD to reference? Distance? Latent-space
   embedding? Multi-dimensional?
4. **Target direction** -- low pcoord is the target (`LowRecycler`,
   typical for folding) or high pcoord is the target (`HighRecycler`,
   typical for unbinding)?
5. **Compute** -- local laptop, multi-GPU workstation, or HPC (Slurm/PBS)?
6. **Starting from** -- a fresh example or extending an existing one
   (`examples/openmm_ntl9_hk/`, `examples/minimal_westpa/`)?

If the user is vague on any of the above, ask before generating code.

## Project Layout

Every DeepDriveWE example follows this structure:

```
examples/<name>/
├── main.py              # Entry point: parse args, build ensemble, launch workflow
├── workflow.py          # Agent subclasses + Pydantic config models
├── config.yaml          # All experiment parameters (edit per-run)
├── common_files/        # Shared reference files (reference.pdb, topology, etc.)
└── inputs/
    └── bstates/         # Basis state structures in per-walker subdirectories
        ├── 00/
        ├── 01/
        └── ...
```

### Critical rule: serializable classes live in `workflow.py`

Parsl workers use `dill` to deserialize agent classes. `dill` can only
resolve classes from importable modules -- not from `__main__`. So:

- Agent subclasses (`MySimAgent`, `MyWestpaAgent`) -> **`workflow.py`**
- Pydantic config models (`SimulationConfig`, `InferenceConfig`,
  `ExperimentSettings`) -> **`workflow.py`**
- `argparse`, `Manager` setup, `ParslPoolExecutor`, signal handling ->
  **`main.py`**

The reference example `examples/openmm_ntl9_hk/` demonstrates this
split. Follow it.

## Step-by-Step

### Step 1: Define config models in `workflow.py`

Subclass `deepdrivewe.api.BaseModel` (which adds YAML load/dump) for
anything loaded from `config.yaml`:

```python
from deepdrivewe.api import BaseModel, BasisStates, TargetState
from deepdrivewe.parsl import ComputeConfigTypes
from pydantic import Field


class SimulationConfig(BaseModel):
    # Fields specific to your MD engine / reporter
    ...


class InferenceConfig(BaseModel):
    sims_per_bin: int = Field(default=4)
    # + any other resampler knobs


class ExperimentSettings(BaseModel):
    output_dir: Path
    num_iterations: int = Field(ge=1)
    basis_states: BasisStates
    target_states: list[TargetState]
    simulation_config: SimulationConfig
    inference_config: InferenceConfig
    compute_config: ComputeConfigTypes
    # + basis_state_initializer (callable that returns initial pcoord)
```

Use `pydantic.field_validator` to resolve file paths and create output
dirs at config-load time -- fail fast on bad input.

### Step 2: Subclass `SimulationAgent`

```python
from deepdrivewe.workflows.westpa import SimulationAgent

class MySimAgent(SimulationAgent):
    def __init__(self, westpa_handle, sim_config, output_dir, logfile=None):
        super().__init__(westpa_handle, logfile=logfile)
        self.sim_config = sim_config
        self.output_dir = output_dir

    def run_simulation(self, metadata: SimMetadata) -> SimResult:
        metadata.mark_simulation_start()
        # 1. Run MD starting from metadata.parent_restart_file
        # 2. Compute pcoord (RMSD / distance / latent embedding / whatever)
        # 3. Set metadata.restart_file, metadata.pcoord
        metadata.mark_simulation_end()
        return SimResult(data={'pcoord': pcoord_array}, metadata=metadata)
```

`run_simulation` is blocking; the base class offloads it to a thread
pool via `agent_run_sync`, so don't make it `async`. Heavy per-agent
state (ML models, loaded topologies) should be initialized in
`agent_on_startup` (async) or `__init__`.

### Step 3: Subclass `WestpaAgent`

```python
from deepdrivewe.workflows.westpa import WestpaAgent
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import HuberKimResampler

class MyWestpaAgent(WestpaAgent):
    def __init__(self, simulation_handles, max_iterations, ensemble,
                 checkpointer=None, inference_config=None, logfile=None):
        super().__init__(simulation_handles, max_iterations, ensemble,
                         checkpointer=checkpointer, logfile=logfile)
        self.inference_config = inference_config or InferenceConfig()

    def run_inference(self, sim_results):
        cur_sims = [r.metadata for r in sim_results]
        cfg = self.inference_config

        binner = RectilinearBinner(
            bins=[...],                              # design per use case
            bin_target_counts=cfg.sims_per_bin,
            target_state_inds=0,
        )
        recycler = LowRecycler(                      # or HighRecycler
            basis_states=self.basis_states,
            target_threshold=self.target_states[0].pcoord[0],
        )
        resampler = HuberKimResampler(
            sims_per_bin=cfg.sims_per_bin,
        )
        return resampler.run(cur_sims, binner, recycler)
```

### Step 4: Write `main.py`

The `main.py` pattern from `examples/openmm_ntl9_hk/main.py` is the
canonical template. Key structure:

1. `_export_pythonpath()` -- add the example dir to `PYTHONPATH` so
   Parsl workers can import `workflow`.
2. Parse args (`-c config.yaml`, `--exchange local|globus`).
3. Load `ExperimentSettings.from_yaml(args.config)`.
4. Dump the resolved config back to `output_dir/params.yaml` (audit).
5. Build `ParslPoolExecutor` from `cfg.compute_config.get_parsl_config(...)`.
6. Initialize or resume the `WeightedEnsemble` via `EnsembleCheckpointer`.
7. Install a SIGTERM handler that calls `gpu_executor.shutdown()` and
   `os._exit(0)` (Parsl workers survive the main process; normal
   shutdown hangs).
8. Run `run_westpa_workflow(...)` inside `async with Manager(...)`.

### Step 5: Write `config.yaml`

Mirror the field structure of `ExperimentSettings`. Keep it heavily
commented -- users will edit this per-run. Use the `openmm_ntl9_hk`
config as a template.

### Step 6: Verify

First run with `num_iterations: 2` and a tiny `simulation_length_ns`:

```bash
cd examples/<name>
python main.py -c config.yaml
```

Check:

- `results/runtime.log` shows the iteration loop firing.
- `results/checkpoints/checkpoint-000002.json` exists.
- `results/west.h5` is written.
- `results/simulation/000001/*/` contains per-walker output.

Only scale up iterations / simulation length after the small run
passes.

## Component Selection Guide

### Binners (`deepdrivewe.binners`)

| Binner | When to Use |
|---|---|
| `RectilinearBinner` | 1D progress coordinate with fixed boundaries. The default. |
| `MultiRectilinearBinner` | N-D progress coordinate (e.g., RMSD + radius of gyration). |

Bin design tip: put finer bins near the target to increase resolution
where it matters. The NTL9 example uses 0.1 A bins from 1.0-4.6 A,
0.2 A from 4.6-6.6 A, 0.6 A beyond.

### Recyclers (`deepdrivewe.recyclers`)

| Recycler | When to Use |
|---|---|
| `LowRecycler` | Target is at low pcoord (folded protein, bound ligand). |
| `HighRecycler` | Target is at high pcoord (unfolded, unbound, dissociated). |

### Resamplers (`deepdrivewe.resamplers`)

| Resampler | When to Use |
|---|---|
| `HuberKimResampler` | Standard Huber-Kim split/merge. Start here. |
| `SplitLowResampler` / `SplitHighResampler` | Split-only variants (no merging) near target. |
| `LOFLowResampler` | ML-enhanced resampling using local outlier factor in a latent space -- requires `deepdrivewe.ai` models. |

### Compute configs (`deepdrivewe.parsl`)

| Config | When to Use |
|---|---|
| `LocalConfig` | Single machine, CPU-only (testing / tiny systems). |
| `WorkstationConfig` | Multi-GPU workstation (`available_accelerators=["0","1",...]`). |
| `VistaConfig` | TACC Vista cluster. |
| Slurm / PBS variants | Other HPC schedulers -- see `ComputeConfigTypes`. |

## Common Extension Points

- **Custom progress coordinate** -- override `run_simulation`, compute
  whatever scalar/vector you want, assign to `metadata.pcoord` as
  `list[list[float]]` (shape: `(n_frames, pcoord_dim)`).
- **Custom reporter / data products** -- add keys to
  `SimResult.data` (`np.ndarray` values). These are passed to
  `run_inference` unchanged and can feed an ML-based resampler.
- **Custom resampler** -- subclass `deepdrivewe.resamplers.Resampler`
  and override `.run(cur_sims, binner, recycler)`.
- **ML-driven resampling** -- load a trained model in
  `agent_on_startup`, use it in `run_inference` to compute
  latent-space coordinates, and use `LOFLowResampler` or a custom
  resampler.

## Gotchas

- **`from __future__ import annotations`** is required in every file
  (project ruff rule).
- **Line length 79** chars. Single quotes. Numpy docstrings.
- Don't put agent classes or config models in `main.py` -- Parsl
  workers can't deserialize them from `__main__`.
- The first `SimMetadata` batch comes from
  `ensemble.next_sims` (populated by
  `ensemble.initialize_basis_states(...)`). Don't build it manually
  unless you're mocking.
- `EnsembleCheckpointer` resumes automatically if `output_dir`
  contains a checkpoint -- always construct it before the
  `WeightedEnsemble` so resumes are transparent.

## References

- Canonical production example: `examples/openmm_ntl9_hk/`
- Minimal stateless example: `examples/minimal_westpa/`
- Full DeepDriveWE (sim + train + inference): `examples/minimal_pattern/`
- API docs: see `deepdrivewe.api`, `deepdrivewe.workflows.westpa`,
  `deepdrivewe.binners`, `deepdrivewe.recyclers`,
  `deepdrivewe.resamplers`, `deepdrivewe.parsl`.
