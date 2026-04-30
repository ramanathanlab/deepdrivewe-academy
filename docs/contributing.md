# Contributing

Contributions are welcome! This guide covers how to set up a development
environment, run the test suite, and build the documentation locally.

## Development Setup

The project supports both [uv](https://docs.astral.sh/uv/) and pip.
A pinned `uv.lock` is committed for reproducible installs.

### With uv (recommended)

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
uv sync --extra dev --extra docs
uv run pre-commit install
```

`uv sync` creates a `.venv/` in the repo, installs the project in
editable mode, and resolves dependencies from `uv.lock`. Activate the
environment with `source .venv/bin/activate`, or prefix commands with
`uv run` (e.g. `uv run pytest`).

### With pip

```bash
git clone git@github.com:ramanathanlab/deepdrivewe-academy.git
cd deepdrivewe-academy
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

### Updating the lock file

Whenever a dependency in `pyproject.toml` changes, regenerate
`uv.lock` and commit it together with the change:

```bash
uv lock
```

Use `uv lock --upgrade` to refresh transitive pins without changing
`pyproject.toml`.

## Running Tests

```bash
# Full test suite with coverage
uv run pytest                # or: pytest

# Single test
uv run pytest tests/path_to_test.py::test_name

# Via tox (isolated environment)
uv run tox -e py310          # or: tox -e py310
```

## Linting and Type Checking

Pre-commit hooks run ruff and mypy automatically on each commit. You can
also run them manually:

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files   # or: pre-commit run --all-files

# Individual tools
uv run ruff check .         # lint
uv run ruff format .        # format
uv run mypy deepdrivewe/    # type check
```

## Building the Documentation

The documentation is built with
[ProperDocs](https://properdocs.org/) (a continuation of MkDocs 1.x)
and the Material theme. API reference pages are auto-generated from
docstrings.

```bash
# uv
uv sync --extra docs
uv run properdocs serve

# pip
pip install -e '.[dev,docs]'
properdocs serve

# Production build (strict mode)
uv run properdocs build --strict   # or: properdocs build --strict
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
