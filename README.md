# deepdrivewe-academy

[![CI](https://github.com/ramanathanlab/deepdrivewe-academy/actions/workflows/ci.yml/badge.svg)](https://github.com/ramanathanlab/deepdrivewe-academy/actions/workflows/ci.yml)
[![Docs](https://github.com/ramanathanlab/deepdrivewe-academy/actions/workflows/docs.yml/badge.svg?branch=main)](https://ramanathanlab.github.io/deepdrivewe-academy)
[![Release](https://img.shields.io/github/v/release/ramanathanlab/deepdrivewe-academy?include_prereleases&sort=semver)](https://github.com/ramanathanlab/deepdrivewe-academy/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ramanathanlab/deepdrivewe-academy/main.svg)](https://results.pre-commit.ci/latest/github/ramanathanlab/deepdrivewe-academy/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Implementation of [DeepDriveWE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136) using [Academy](https://docs.academy-agents.org/stable/).

📖 **Documentation:** https://ramanathanlab.github.io/deepdrivewe-academy

## Installation

The project supports both [uv](https://docs.astral.sh/uv/) and pip.
A pinned `uv.lock` is committed for reproducible installs.

### With uv (recommended)

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
uv sync
```

`uv sync` creates a `.venv/`, installs the project in editable mode, and
resolves all dependencies from `uv.lock`. Activate the environment with
`source .venv/bin/activate`, or prefix commands with `uv run` (e.g.
`uv run python`).

### With pip

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
pip install -e .
```

### Full installation with MD dependencies

OpenMM and AmberTools are best installed via conda. After creating the
conda env, you can use either uv or pip for the Python package:

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
conda create -n deepdrivewe python=3.10 -y
conda activate deepdrivewe
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
pip install -e .   # or: uv pip install -e .
```

To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/)
for your system and drivers. To use `mdlearn`, you may need an earlier version of PyTorch:
```bash
pip install torch==1.12
```

## Contributing

For development, install the project with the `dev` and `docs` extras and
set up the pre-commit hooks.

With uv (recommended — uses `uv.lock` for reproducible installs):
```bash
uv sync --extra dev --extra docs
uv run pre-commit install
```

With pip:
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

To test the code:
```bash
# uv
uv run pre-commit run --all-files
uv run tox -e py310

# pip / activated venv
pre-commit run --all-files
tox -e py310
```

When a dependency in `pyproject.toml` changes, refresh the lock file with
`uv lock` and commit `uv.lock` alongside the change.

### Building the Documentation

Documentation is built with [ProperDocs](https://properdocs.org/) (a
continuation of MkDocs 1.x).
```bash
# uv
uv sync --extra docs
uv run properdocs serve

# pip
pip install -e '.[dev,docs]'
properdocs serve
```
Then open http://localhost:8000 in your browser. For a production build:
```bash
properdocs build --strict   # or: uv run properdocs build --strict
```

## Citation

If you use DeepDriveWE in your research, please cite:

> Leung, J. M. G.; Frazee, N. C.; Brace, A.; Bogetti, A. T.;
> Ramanathan, A.; Chong, L. T. "Unsupervised Learning of Progress
> Coordinates during Weighted Ensemble Simulations: Application to NTL9
> Protein Folding." *Journal of Chemical Theory and Computation* **2025**,
> *21* (7), 3691--3699.
> [DOI: 10.1021/acs.jctc.4c01136](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136)

BibTeX:

```bibtex
@article{leung2025unsupervised,
  title={Unsupervised Learning of Progress Coordinates during Weighted Ensemble Simulations: Application to NTL9 Protein Folding},
  author={Leung, Jeremy MG and Frazee, Nicolas C and Brace, Alexander and Bogetti, Anthony T and Ramanathan, Arvind and Chong, Lillian T},
  journal={Journal of chemical theory and computation},
  volume={21},
  number={7},
  pages={3691--3699},
  year={2025},
  publisher={ACS Publications}
}
```
