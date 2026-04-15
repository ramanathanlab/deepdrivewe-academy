# Contributing

Contributions are welcome! This guide covers how to set up a development
environment, run the test suite, and build the documentation locally.

## Development Setup

Create a virtual environment and install all development and
documentation dependencies:

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
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

## Commit Messages & Releases

This project uses
[Conventional Commits](https://www.conventionalcommits.org/). Commit
types are not decorative — they drive automated releases. Every merge
to `main` runs
[release-please](https://github.com/googleapis/release-please), which
reads the commits since the last tag and either opens or updates a
"release PR". Merging that release PR creates the git tag (`vX.Y.Z`),
the GitHub Release, a `CHANGELOG.md` entry, and the bumped version
in `pyproject.toml`.

### Version bump rules

While the version is `< 1.0.0`, release-please runs with
`bump-minor-pre-major: true`, so breaking changes stay capped at a
minor bump:

| Commit type                              | Bump       | Example           |
|------------------------------------------|------------|-------------------|
| `feat!:` or `BREAKING CHANGE:` in body   | minor      | `0.1.0` → `0.2.0` |
| `feat:`                                  | minor      | `0.1.0` → `0.2.0` |
| `fix:`                                   | patch      | `0.1.0` → `0.1.1` |
| `perf:`                                  | patch      | `0.1.0` → `0.1.1` |
| `chore:`, `ci:`, `docs:`, `test:`, `refactor:`, `style:` | no release | —     |

Once the project cuts `1.0.0`, standard [Semantic Versioning](https://semver.org/)
rules apply and `feat!:` bumps the major (`1.2.3` → `2.0.0`).

The highest-severity commit in the batch wins — one `feat:` among
several `fix:` commits yields a minor bump, not a patch. A batch of
only `chore:` / `ci:` / `docs:` commits produces no release PR at all.

### Forcing a specific version

To override the automatic bump (for example, to graduate out of `0.x`),
include a `Release-As:` trailer in any commit body on `main`:

```
Release-As: 1.0.0
```

The next release PR will use that version instead of the one inferred
from commit types.
