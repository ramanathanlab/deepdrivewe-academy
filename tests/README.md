# deepdrivewe tests

## Scope

The suite covers the **math and policy core** of the weighted-ensemble
engine — the code paths where a silent bug would corrupt a production run:

| Module                  | Tests                           |
|-------------------------|---------------------------------|
| `deepdrivewe.api`       | `tests/unit/test_api.py`        |
| `deepdrivewe.binners`   | `tests/unit/test_binners.py`    |
| `deepdrivewe.resamplers`| `tests/unit/test_resamplers.py` |
| `deepdrivewe.recyclers` | `tests/unit/test_recyclers.py`  |
| `deepdrivewe.utils`     | `tests/unit/test_utils.py`      |
| `deepdrivewe.checkpoint`| `tests/integration/test_checkpoint.py` |
| `deepdrivewe.io`        | `tests/integration/test_io.py`  |

## Not covered (by design)

These modules require heavy external infrastructure and are exercised
through the `examples/` workflows rather than pytest:

- `deepdrivewe.simulation.*` — needs OpenMM, AmberTools, real MD inputs
- `deepdrivewe.ai.*` — needs torch + mdlearn training loops
- `deepdrivewe.workflows.*` — Academy orchestration glue
- `deepdrivewe.parsl` — parsl config wrapper

They are `omit`-ed from coverage in `pyproject.toml`. When CI gains those
dependencies, move them out of the omit list and add a `tests/e2e/`
suite driven by a small basis state set (e.g. `synd`).

## Running

```bash
pytest                              # all tests
pytest tests/unit                   # unit tests only
pytest -m integration               # integration only
pytest --cov --cov-report=term-missing
```

## Layout

```
tests/
├── conftest.py          # shared fixtures (sim_factory, basis_state_dir, ...)
├── test_import.py       # package import smoke test
├── unit/                # pure-python, no filesystem
└── integration/         # tmp_path-backed filesystem/HDF5
```
