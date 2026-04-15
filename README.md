# deepdrivewe-academy
Academy implementation of DeepDriveWE.

Implementation of [DeepDriveWE](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c01136) using [Academy](https://docs.academy-agents.org/stable/).

## Installation

To install the package, run the following command:
```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
pip install -e .
```

Full installation including dependencies:
```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
conda create -n deepdrivewe python=3.10 -y
conda install omnia::ambertools -y
conda install conda-forge::openmm==7.7 -y
pip install -e .
```

To use deep learning models, install the correct version of [PyTorch](https://pytorch.org/get-started/locally/)
for your system and drivers. To use `mdlearn`, you may need an earlier version of PyTorch:
```bash
pip install torch==1.12
```

## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```

### Building the Documentation

Documentation is built with [ProperDocs](https://properdocs.org/) (a
continuation of MkDocs 1.x).
```bash
pip install -e '.[dev,docs]'
properdocs serve
```
Then open http://localhost:8000 in your browser. For a production build:
```bash
properdocs build --strict
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
