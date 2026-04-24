"""Tests for `deepdrivewe.recyclers`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from deepdrivewe.api import BasisStates
from deepdrivewe.api import SimMetadata
from deepdrivewe.recyclers import HighRecycler
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.recyclers.base import Recycler

SimFactory = Callable[..., SimMetadata]


@pytest.mark.unit()
class TestLowRecycler:
    """Covers the "recycle below threshold" policy."""

    def test_recycle_selects_below_threshold(
        self,
        basis_states: BasisStates,
    ) -> None:
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=2.0,
        )
        pcoords = np.array([[0.5], [1.5], [2.5], [3.5]])
        idxs = recycler.recycle(pcoords).tolist()
        assert idxs == [0, 1]

    def test_pcoord_idx_selects_dimension(
        self,
        basis_states: BasisStates,
    ) -> None:
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=1.0,
            pcoord_idx=1,
        )
        pcoords = np.array([[10.0, 0.5], [10.0, 5.0]])
        idxs = recycler.recycle(pcoords).tolist()
        assert idxs == [0]

    def test_recycle_simulations_replaces_with_basis_state(
        self,
        sim_factory: SimFactory,
        basis_states: BasisStates,
    ) -> None:
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=2.0,
        )
        cur = [
            sim_factory(simulation_id=0, pcoord=[[0.0], [0.5]]),  # recycled
            sim_factory(simulation_id=1, pcoord=[[0.0], [3.5]]),  # kept
        ]
        nxt = [
            sim_factory(simulation_id=10),
            sim_factory(simulation_id=11),
        ]
        np.random.seed(0)
        new_cur, new_nxt = recycler.recycle_simulations(cur, nxt)

        # The recycled sim on the next iter now points at a basis-state
        # restart file and carries a negated parent id.
        recycled = new_nxt[0]
        assert recycled.parent_restart_file in {
            s.parent_restart_file for s in basis_states
        }
        # The recycler encodes the *next-iter* sim's own id (negated, -1)
        # as parent_simulation_id to flag recycling.
        assert recycled.parent_simulation_id == -(nxt[0].simulation_id + 1)

        # endpoint_type on the corresponding current sim is set to 3.
        assert new_cur[0].endpoint_type == 3
        assert new_cur[1].endpoint_type == 1

    def test_recycle_simulations_no_match_is_noop(
        self,
        sim_factory: SimFactory,
        basis_states: BasisStates,
    ) -> None:
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=-np.inf,  # never triggers
        )
        cur = [sim_factory(simulation_id=0, pcoord=[[0.0], [5.0]])]
        nxt = [sim_factory(simulation_id=10)]
        new_cur, new_nxt = recycler.recycle_simulations(cur, nxt)
        assert new_cur[0].endpoint_type == 1
        assert new_nxt[0].parent_restart_file == nxt[0].parent_restart_file


@pytest.mark.unit()
class TestHighRecycler:
    """Covers the "recycle above threshold" policy."""

    def test_recycle_selects_above_threshold(
        self,
        basis_states: BasisStates,
    ) -> None:
        recycler = HighRecycler(
            basis_states=basis_states,
            target_threshold=2.0,
        )
        pcoords = np.array([[0.5], [1.5], [2.5], [3.5]])
        idxs = recycler.recycle(pcoords).tolist()
        assert idxs == [2, 3]

    def test_pcoord_idx_selects_dimension(
        self,
        basis_states: BasisStates,
    ) -> None:
        recycler = HighRecycler(
            basis_states=basis_states,
            target_threshold=1.0,
            pcoord_idx=1,
        )
        pcoords = np.array([[0.5, 0.5], [0.5, 5.0]])
        idxs = recycler.recycle(pcoords).tolist()
        assert idxs == [1]


@pytest.mark.unit()
def test_recycler_is_abstract(basis_states: BasisStates) -> None:
    with pytest.raises(TypeError):
        Recycler(basis_states)  # type: ignore[abstract]
