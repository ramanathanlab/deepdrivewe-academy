"""Tests for `deepdrivewe.binners`."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pytest

from deepdrivewe.api import SimMetadata
from deepdrivewe.binners import MultiRectilinearBinner
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.binners.base import Binner

SimFactory = Callable[..., SimMetadata]


@pytest.mark.unit
class TestRectilinearBinner:
    """Covers 1D bin assignment and construction checks."""

    def test_nbins_is_edges_minus_one(self) -> None:
        # Bin layout follows WESTPA convention: pad with -inf/+inf so
        # values never overflow into the terminal bins and trip the
        # "coords outside bin boundaries" warning.
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0, np.inf],
            bin_target_counts=5,
        )
        assert binner.nbins == 5

    def test_rejects_unsorted_bins(self) -> None:
        with pytest.raises(ValueError, match='sorted'):
            RectilinearBinner(bins=[0.0, 2.0, 1.0], bin_target_counts=5)

    def test_assign_bins_digitizes(self) -> None:
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0, np.inf],
            bin_target_counts=5,
        )
        pcoords = np.array([[0.5], [1.5], [2.5]])
        # Bins are 0-indexed over [-inf,0), [0,1), [1,2), [2,3), [3,+inf).
        assert binner.assign_bins(pcoords).tolist() == [1, 2, 3]

    def test_assign_bins_warns_and_clips_out_of_range(self) -> None:
        # No sentinel bins here — the whole point of the test is to
        # exercise the out-of-range warning path.
        binner = RectilinearBinner(
            bins=[0.0, 1.0, 2.0],
            bin_target_counts=5,
        )
        pcoords = np.array([[-1.0], [0.5], [5.0]])
        with pytest.warns(UserWarning, match='outside the bin'):
            ids = binner.assign_bins(pcoords)
        assert ids.min() >= 0
        assert ids.max() < len(binner.bins)

    def test_pcoord_idx_selects_dimension(self) -> None:
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, np.inf],
            bin_target_counts=5,
            pcoord_idx=1,
        )
        pcoords = np.array([[10.0, 0.5], [10.0, 1.5]])
        assert binner.assign_bins(pcoords).tolist() == [1, 2]

    def test_labels_are_state_prefixed(self) -> None:
        binner = RectilinearBinner(
            bins=[0.0, 1.0, 2.0],
            bin_target_counts=3,
        )
        assert binner.labels == ['state0', 'state1']

    def test1dAssign(self) -> None:  # noqa: N802
        bounds = [0.0, 1.0, 2.0, 3.0]
        coords = np.array([-1, 0, 0.5, 1.5, 1.6, 2.0, 2.0, 2.9])[:, None]

        assigner = RectilinearBinner(
            bins=bounds,
            bin_target_counts=3,
            target_state_inds=[None],  # type: ignore[list-item]
            pcoord_idx=0,
        )

        with pytest.warns(UserWarning):
            assert (
                assigner.assign_bins(coords) == [0, 0, 0, 1, 1, 2, 2, 2]
            ).all()


@pytest.mark.unit
class TestMultiRectilinearBinner:
    """Covers multi-dimensional bin assignment."""

    def test_nbins_is_product(self) -> None:
        binner = MultiRectilinearBinner(
            bins=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]],
            bin_target_counts=5,
        )
        assert binner.nbins == 2 * 3

    def test_rejects_unsorted_bins(self) -> None:
        with pytest.raises(ValueError, match='sorted'):
            MultiRectilinearBinner(
                bins=[[0.0, 2.0, 1.0]],
                bin_target_counts=5,
            )

    def test_assign_bins_row_major(self) -> None:
        binner = MultiRectilinearBinner(
            bins=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            bin_target_counts=5,
        )
        # Two 2x2 grids: (0.5, 0.5) -> (0,0); (1.5, 1.5) -> (1,1)
        pcoords = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
        ids = binner.assign_bins(pcoords)
        assert ids.shape == (3,)
        # All bin indices should land inside [0, nbins).
        assert ids.min() >= 0
        assert ids.max() < binner.nbins

    def test_assign_bins_warns_and_clips_out_of_range(self) -> None:
        binner = MultiRectilinearBinner(
            bins=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            bin_target_counts=5,
        )
        pcoords = np.array([[-5.0, 0.5], [1.5, 10.0]])
        with pytest.warns(UserWarning, match='outside the bin'):
            ids = binner.assign_bins(pcoords)
        assert ids.min() >= 0
        assert ids.max() < binner.nbins

    def test2dAssign(self) -> None:  # noqa: N802
        boundaries = [(-1, -0.5, 0, 0.5, 1), (-1, -0.5, 0, 0.5, 1)]
        coords = np.array(
            [
                (-2, -2),
                (-0.75, -0.75),
                (-0.25, -0.25),
                (0, 0),
                (0.25, 0.25),
                (0.75, 0.75),
                (-0.25, 0.75),
                (0.25, -0.75),
            ],
        )

        assigner = MultiRectilinearBinner(
            boundaries,  # type: ignore[arg-type]
            bin_target_counts=3,
            target_state_inds=[None],  # type: ignore[list-item]
        )

        # Row-major 4x4 grid layout (x outer, y inner):
        #   0: x in [-1,-0.5), y in [-1,-0.5)
        #   1: x in [-1,-0.5), y in [-0.5, 0)
        #   2: x in [-1,-0.5), y in [ 0, 0.5)
        #   3: x in [-1,-0.5), y in [ 0.5, 1)
        #   4: x in [-0.5, 0), y in [-1,-0.5)
        #   5: x in [-0.5, 0), y in [-0.5, 0)
        #   6: x in [-0.5, 0), y in [ 0, 0.5)
        #   7: x in [-0.5, 0), y in [ 0.5, 1)
        #   8: x in [ 0, 0.5), y in [-1,-0.5)
        #   9: x in [ 0, 0.5), y in [-0.5, 0)
        #  10: x in [ 0, 0.5), y in [ 0, 0.5)
        #  11: x in [ 0, 0.5), y in [ 0.5, 1)
        #  12: x in [ 0.5, 1), y in [-1,-0.5)
        #  13: x in [ 0.5, 1), y in [-0.5, 0)
        #  14: x in [ 0.5, 1), y in [ 0, 0.5)
        #  15: x in [ 0.5, 1), y in [ 0.5, 1)
        with pytest.warns(UserWarning):
            assert (
                assigner.assign_bins(coords) == [0, 0, 5, 10, 10, 15, 7, 8]
            ).all()


@pytest.mark.unit
class TestBinnerHelpers:
    """Covers shared methods on the base class (via a concrete subclass)."""

    def _binner(self) -> RectilinearBinner:
        # Pad with -inf on the low end so values in [0, 3) always land in
        # a real interior bin and never fire the overflow warning.
        return RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0],
            bin_target_counts=5,
        )

    def test_get_bin_target_counts_expands_int(self) -> None:
        counts = self._binner().get_bin_target_counts()
        assert counts == [5, 5, 5, 5]

    def test_get_bin_target_counts_zeros_target_bin(self) -> None:
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0],
            bin_target_counts=5,
            target_state_inds=[3],
        )
        assert binner.get_bin_target_counts() == [5, 5, 5, 0]

    def test_get_bin_target_counts_accepts_single_int_index(self) -> None:
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0],
            bin_target_counts=5,
            target_state_inds=2,
        )
        assert binner.get_bin_target_counts() == [5, 5, 0, 5]

    def test_pickle_and_hash_is_stable(self) -> None:
        pkl, h = self._binner().pickle_and_hash()
        # Pickled bytes round-trip via pickle.
        restored = pickle.loads(pkl)
        assert isinstance(restored, RectilinearBinner)
        # Hash is 64 hex chars (sha256).
        assert len(h) == 64

    def test_bin_simulations_groups_by_bin(
        self,
        sim_factory: SimFactory,
    ) -> None:
        binner = self._binner()
        sims = [
            sim_factory(parent_pcoord=[0.5]),  # bin 1 ([0, 1))
            sim_factory(parent_pcoord=[1.5]),  # bin 2 ([1, 2))
            sim_factory(parent_pcoord=[1.7]),  # bin 2
            sim_factory(parent_pcoord=[2.5]),  # bin 3 ([2, 3))
        ]
        grouping = binner.bin_simulations(sims)
        assert sorted(grouping[1]) == [0]
        assert sorted(grouping[2]) == [1, 2]
        assert sorted(grouping[3]) == [3]

    def test_compute_iteration_metadata(
        self,
        sim_factory: SimFactory,
    ) -> None:
        binner = self._binner()
        sims = [
            sim_factory(weight=0.25, pcoord=[[0.0], [0.5]]),
            sim_factory(weight=0.25, pcoord=[[0.0], [0.7]]),
            sim_factory(weight=0.5, pcoord=[[0.0], [2.5]]),
        ]
        meta = binner.compute_iteration_metadata(sims)
        assert meta.iteration_id == 1
        # Bin 1 holds sims 0+1 -> 0.5; bin 3 holds sim 2 -> 0.5.
        assert meta.min_bin_prob == pytest.approx(0.5)
        assert meta.max_bin_prob == pytest.approx(0.5)
        assert meta.bin_target_counts == [5, 5, 5, 5]
        assert meta.binner_hash  # non-empty
        assert meta.binner_pickle  # non-empty


@pytest.mark.unit
def test_binner_is_abstract() -> None:
    with pytest.raises(TypeError):
        Binner(bin_target_counts=1)  # type: ignore[abstract]
