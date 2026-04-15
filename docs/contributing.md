# Contributing

Contributions are welcome! This guide covers how to set up a development
environment, run the test suite, and build the documentation locally.

## Development Setup

Create a virtual environment and install all development and
documentation dependencies:

```bash
git clone git@github.com:braceal/deepdrivewe-academy.git
cd deepdrivewe-academy
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

## Running Tests

```bash
# Run the full test suite with coverage
pytest

# Run a single test
pytest tests/path_to_test.py::test_name

# Run via tox (isolated environment)
tox -e py310
```

## Linting and Type Checking

Pre-commit hooks run ruff and mypy automatically on each commit. You can
also run them manually:

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
ruff check .         # lint
ruff format .        # format
mypy deepdrivewe/    # type check
```

## Building the Documentation

The documentation is built with
[ProperDocs](https://properdocs.org/) (a continuation of MkDocs 1.x)
and the Material theme. API reference pages are auto-generated from
docstrings.

```bash
# Install docs dependencies (included in dev setup above)
pip install -e '.[dev,docs]'

# Live preview with hot reload
properdocs serve

# Production build (strict mode)
properdocs build --strict
```

Then open <http://localhost:8000> in your browser to preview.

## Code Style

- **Line length**: 79 characters
- **Quotes**: single quotes (enforced by ruff)
- **Imports**: `from __future__ import annotations` required in every
  file; force single-line imports
- **Docstrings**: numpy convention
- **Type annotations**: required on all function signatures
- **Immutability**: prefer `@dataclass(frozen=True)` and `NamedTuple`

## Pull Requests

1. Create a branch from `develop` (not `main`).
2. Name the branch `feature/<issue>-<slug>`, `bugfix/<issue>-<slug>`,
   or `chore/<issue>-<slug>`.
3. Write tests for new functionality.
4. Make sure `pre-commit run --all-files` and `pytest` pass.
5. Open a PR targeting `develop`.
