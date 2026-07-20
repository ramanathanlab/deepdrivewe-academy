# Installation

The project supports both [uv](https://docs.astral.sh/uv/) and pip.
A pinned `uv.lock` is committed to the repository, so `uv sync`
produces a reproducible install across machines.

## Basic Install

### With uv (recommended)

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
uv sync
```

`uv sync` creates a `.venv/` in the repo, installs the project in
editable mode, and resolves all dependencies from `uv.lock`. Activate
the environment with `source .venv/bin/activate`, or prefix commands
with `uv run` (for example, `uv run python -m deepdrivewe ...`).

### With pip

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
pip install -e .
```

## Full Install with MD Dependencies

For running molecular dynamics simulations with OpenMM and AmberTools,
use conda to install the simulation backends first, then install the
Python package with either uv or pip:

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy

conda create -n deepdrivewe python=3.11 -y
conda activate deepdrivewe
conda install -c conda-forge openmm=8.1
conda install omnia::ambertools -y
pip install -e .          # or: uv pip install -e .
```

## Deep Learning Models

To use the built-in AI models (convolutional VAE, adversarial
autoencoder), install the correct version of
[PyTorch](https://pytorch.org/get-started/locally/) for your system and
GPU drivers:

```bash
pip install torch
```

!!! note
    The `mdlearn` dependency may require an earlier version of PyTorch.
    If you encounter compatibility issues, try:
    ```bash
    pip install torch==1.12
    ```

## Development Setup

For contributing or running the test suite:

```bash
# uv (recommended)
uv sync --extra dev --extra docs
uv run pre-commit install

# pip
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

Verify the setup:

```bash
uv run pre-commit run --all-files   # or: pre-commit run --all-files
uv run pytest                       # or: pytest
```

## Documentation Build

To build the documentation locally:

```bash
# uv
uv sync --extra docs
uv run properdocs serve

# pip
pip install -e '.[dev,docs]'
properdocs serve
```

Then open <http://localhost:8000> in your browser. For a production
build with strict checking:

```bash
uv run properdocs build --strict   # or: properdocs build --strict
```

See the [Contributing](../contributing.md) guide for more details.
