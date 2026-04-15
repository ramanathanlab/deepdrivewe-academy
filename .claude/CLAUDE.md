# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepDriveWE-Academy is a Python implementation of [DeepDriveWE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136) (weighted ensemble molecular dynamics) using the [Academy](https://docs.academy-agents.org/stable/) multi-agent framework. Currently early-stage with a scaffold example agent.

## Commands

### Development Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

Full setup with MD simulation dependencies:
```bash
conda create -n deepdrivewe python=3.10 -y
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
pip install -e '.[dev,docs]'
```

### Testing & Linting
```bash
# Run all tests with coverage
tox -e py310

# Run tests directly
pytest
pytest tests/path_to_test.py::test_name    # single test

# Coverage
coverage erase && coverage run -m pytest && coverage report

# Linting + type checking (all pre-commit hooks)
pre-commit run --all-files

# Individual tools
ruff check .                    # lint
ruff format .                   # format
mypy deepdrivewe/               # type check
```

### Docs
```bash
properdocs build --strict
```

## Architecture

The project uses the **Academy** multi-agent framework (`academy-py`). Key primitives:

- **`Agent`** — Base class for all agents. Subclass and decorate async methods with `@action` to expose them as remotely invocable operations.
- **`@action`** — Marks an async method as callable by other agents or the manager.
- **`@loop` / `@event` / `@timer`** — Decorators for background control loops on agents.
- **`Manager`** — Orchestrates agent lifecycle. Created via `Manager.from_exchange_factory()`.
- **`LocalExchangeFactory`** — In-process communication (for testing/local use).
- **`Handle`** — Proxy returned by `manager.launch()` for calling actions on launched agents.
- **Lifecycle hooks** — `agent_on_startup()`, `agent_on_shutdown()` for setup/teardown.
- **`agent_run_sync()`** — Runs blocking functions in a thread pool from async context.

Key domain dependencies: `mdtraj`, `MDAnalysis` (trajectory analysis), `parsl` (parallel execution), `mdlearn` + `scikit-learn` (ML), `h5py` (data I/O).

## Code Style

- **Line length**: 79 characters
- **Quotes**: single quotes (enforced by ruff)
- **Imports**: `from __future__ import annotations` required in every file; force single-line imports
- **Docstrings**: numpy convention
- **Type annotations**: required on all function signatures (mypy strict mode, target py3.10)
- **Immutability**: prefer `@dataclass(frozen=True)` and `NamedTuple`
- **Formatting**: ruff format (single quotes, space indent)

## Git Workflow

- **Default branch**: `develop` (not `main`)
- **PRs target**: `develop`; release PRs go `develop` → `main`
- **Branch naming**: `feature/<issue>-<slug>`, `bugfix/<issue>-<slug>`, `chore/<issue>-<slug>`
- A PreToolUse hook redirects `git checkout main` to `develop`

## Releases

Releases are automated by `release-please` via the [Release workflow](../.github/workflows/release.yml). Every merge to `main` either opens or updates a "release PR". Merging that release PR creates the `vX.Y.Z` tag, a GitHub Release, a `CHANGELOG.md` entry, and bumps the version in `pyproject.toml`.

**Commit messages drive the bump** — use [Conventional Commits](https://www.conventionalcommits.org/). While the version is `< 1.0.0` (config uses `bump-minor-pre-major: true`):

| Commit type                                              | Bump                  |
|----------------------------------------------------------|-----------------------|
| `feat!:` or `BREAKING CHANGE:` in body                   | minor (capped pre-1.0) |
| `feat:`                                                  | minor                 |
| `fix:`                                                   | patch                 |
| `perf:`                                                  | patch                 |
| `chore:`, `ci:`, `docs:`, `test:`, `refactor:`, `style:` | no release            |

After `1.0.0`, standard SemVer applies and `feat!:` bumps the major. The highest-severity commit in the batch wins. To force a specific version, add `Release-As: X.Y.Z` to a commit body on `main`.

Version source of truth is `pyproject.toml`; `deepdrivewe/__init__.py` reads it via `importlib.metadata.version()`, so no other version string needs updating when cutting a release.
