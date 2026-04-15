# Installation

## Basic Install

Clone the repository and install in editable mode:

```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
pip install -e .
```

## Full Install with MD Dependencies

For running molecular dynamics simulations with OpenMM and AmberTools,
use conda to install the simulation backends first:

```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy

conda create -n deepdrivewe python=3.11 -y
conda activate deepdrivewe
conda install -c conda-forge openmm=8.1
conda install omnia::ambertools -y
pip install -e .
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
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

Verify the setup:

```bash
pre-commit run --all-files
pytest
```

## Documentation Build

To build the documentation locally:

```bash
pip install -e '.[dev,docs]'
properdocs serve
```

Then open <http://localhost:8000> in your browser. For a production
build with strict checking:

```bash
properdocs build --strict
```

See the [Contributing](../contributing.md) guide for more details.
