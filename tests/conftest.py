"""Shared pytest fixtures for the deepdrivewe test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from deepdrivewe.api import BasisStateInitializer
from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import TargetState
from deepdrivewe.api import WeightedEnsemble

SimFactory = Callable[..., SimMetadata]


@pytest.fixture
def sim_factory(tmp_path: Path) -> SimFactory:
    """Build `SimMetadata` instances with sensible defaults.

    Overrides any field by kwarg. A fresh restart file is written to
    ``tmp_path`` so fields that point at real files (e.g. basis state
    inputs) pass `Path.is_file()` checks.
    """
    counter = {'next_id': 0}

    def _make(**kwargs: Any) -> SimMetadata:
        sim_id = int(kwargs.pop('simulation_id', counter['next_id']))
        counter['next_id'] = max(counter['next_id'], sim_id + 1)

        restart_file = kwargs.pop(
            'restart_file',
            tmp_path / f'sim_{sim_id:06d}.ncrst',
        )
        assert isinstance(restart_file, Path)
        restart_file.touch()

        parent_restart_file = kwargs.pop(
            'parent_restart_file',
            restart_file,
        )

        defaults: dict[str, Any] = {
            'weight': 0.25,
            'simulation_id': sim_id,
            'iteration_id': 1,
            'parent_restart_file': parent_restart_file,
            'parent_pcoord': [0.0],
            'pcoord': [[0.0], [0.0]],
            'restart_file': restart_file,
            'parent_simulation_id': sim_id,
            'wtg_parent_ids': [sim_id],
        }
        defaults.update(kwargs)
        return SimMetadata(**defaults)

    return _make


@pytest.fixture
def basis_state_dir(tmp_path: Path) -> Path:
    """Create a nested basis-state directory tree.

    Layout::

        basis/
          system1/state.ncrst
          system2/state.ncrst
          system3/state.ncrst
    """
    root = tmp_path / 'basis'
    for idx in range(1, 4):
        sub = root / f'system{idx}'
        sub.mkdir(parents=True)
        (sub / 'state.ncrst').write_text(f'basis-{idx}')
    return root


@pytest.fixture
def simple_initializer() -> BasisStateInitializer:
    """Deterministic basis-state initializer returning a 1D pcoord."""

    def _init(basis_file: str) -> list[float]:
        # Use the digit in the parent folder name to make pcoord deterministic.
        parent = Path(basis_file).parent.name
        digit = ''.join(c for c in parent if c.isdigit()) or '0'
        return [float(digit)]

    return _init


@pytest.fixture
def basis_states(
    basis_state_dir: Path,
    simple_initializer: BasisStateInitializer,
) -> BasisStates:
    """Pre-loaded `BasisStates` with 4 initial ensemble members."""
    states = BasisStates(
        basis_state_dir=basis_state_dir,
        basis_state_ext='.ncrst',
        initial_ensemble_members=4,
    )
    states.load_basis_states(simple_initializer)
    return states


@pytest.fixture
def target_states() -> list[TargetState]:
    """A single target state for tests that require one."""
    return [TargetState(label='target', pcoord=[0.0])]


@pytest.fixture
def weighted_ensemble(
    basis_states: BasisStates,
    target_states: list[TargetState],
) -> WeightedEnsemble:
    """A `WeightedEnsemble` with basis states loaded and no iterations run."""
    return WeightedEnsemble(
        basis_states=basis_states,
        target_states=target_states,
        metadata=IterationMetadata(iteration_id=1),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy Generator for tests that need randomness."""
    return np.random.default_rng(seed=1234)
