"""Tests for `deepdrivewe.resamplers`.

Focus: the split/merge/adjust primitives on the base `Resampler` and the
concrete policies (`SplitLowResampler`, `SplitHighResampler`,
`HuberKimResampler`, `LOFLowResampler`). Weight conservation is the
invariant worth guarding — a silent drift there corrupts a full WE run.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

from deepdrivewe.api import SimMetadata
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.recyclers import LowRecycler
from deepdrivewe.resamplers import HuberKimResampler
from deepdrivewe.resamplers import LOFLowResampler
from deepdrivewe.resamplers import Resampler
from deepdrivewe.resamplers import SplitHighResampler
from deepdrivewe.resamplers import SplitLowResampler

SimFactory = Callable[..., SimMetadata]


def _total_weight(sims: list[SimMetadata]) -> float:
    return sum(s.weight for s in sims)


class _IdentityResampler(Resampler):
    """Concrete resampler used to exercise the base-class helpers."""

    def resample(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
    ) -> tuple[list[SimMetadata], list[SimMetadata]]:
        return cur_sims, next_sims


@pytest.mark.unit()
class TestResamplerPrimitives:
    """Covers the split/merge/adjust helpers on the base class."""

    def test_get_next_sims_advances_iteration(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        cur = [
            sim_factory(
                simulation_id=0,
                iteration_id=4,
                pcoord=[[0.0], [1.2]],
            ),
            sim_factory(
                simulation_id=1,
                iteration_id=4,
                pcoord=[[0.0], [3.4]],
            ),
        ]
        nxt = resampler._get_next_sims(cur)
        assert [s.iteration_id for s in nxt] == [5, 5]
        # parent_pcoord picks up the last pcoord frame of the parent sim.
        assert nxt[0].parent_pcoord == [1.2]
        assert nxt[1].parent_pcoord == [3.4]
        # Fresh simulation_ids start from 0 each iteration.
        assert [s.simulation_id for s in nxt] == [0, 1]

    def test_split_sims_preserves_weight(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        sims = [
            sim_factory(weight=0.4, simulation_id=0),
            sim_factory(weight=0.2, simulation_id=1),
            sim_factory(weight=0.4, simulation_id=2),
        ]
        split = resampler.split_sims(sims, indices=[0], n_splits=4)
        # 2 untouched + 4 split fragments.
        assert len(split) == 6
        # Each fragment's weight is the original divided by n_splits.
        split_weights = [
            s.weight for s in split if s.weight == pytest.approx(0.4 / 4)
        ]
        assert len(split_weights) == 4
        assert _total_weight(split) == pytest.approx(_total_weight(sims))

    def test_merge_sims_sums_weight_and_marks_parents(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        cur = [
            sim_factory(weight=0.1, simulation_id=10),
            sim_factory(weight=0.2, simulation_id=11),
            sim_factory(weight=0.3, simulation_id=12),
        ]
        nxt = [
            sim_factory(
                weight=0.1,
                simulation_id=0,
                parent_simulation_id=10,
            ),
            sim_factory(
                weight=0.2,
                simulation_id=1,
                parent_simulation_id=11,
            ),
            sim_factory(
                weight=0.3,
                simulation_id=2,
                parent_simulation_id=12,
            ),
        ]
        random.seed(0)
        np.random.seed(0)
        merged = resampler.merge_sims(cur, nxt, indices=[0, 1])
        assert len(merged) == 2  # one survivor + the merged sim.
        # Merged sim's weight is the sum of the two merged parents.
        merged_weights = sorted(s.weight for s in merged)
        assert merged_weights == pytest.approx([0.3, 0.3])
        # The non-surviving parent has endpoint_type == 2.
        marked = [s for s in cur if s.endpoint_type == 2]
        assert len(marked) == 1

    def test_split_by_weight_only_splits_overweight(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        sims = [
            sim_factory(weight=0.1),
            sim_factory(weight=0.5),  # overweight
        ]
        out = resampler.split_by_weight(sims, ideal_weight=0.2)
        assert len(out) > len(sims)
        assert _total_weight(out) == pytest.approx(_total_weight(sims))

    def test_adjust_count_splits_when_too_few(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        cur = [sim_factory(weight=0.5)]
        nxt = [sim_factory(weight=0.5)]
        out = resampler.adjust_count(cur, nxt, target_count=3)
        assert len(out) == 3
        assert _total_weight(out) == pytest.approx(0.5)

    def test_adjust_count_merges_when_too_many(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = _IdentityResampler()
        cur = [sim_factory(weight=0.25, simulation_id=i) for i in range(4)]
        nxt = [
            sim_factory(weight=0.25, simulation_id=i + 10) for i in range(4)
        ]
        np.random.seed(0)
        out = resampler.adjust_count(cur, nxt, target_count=2)
        assert len(out) == 2
        assert _total_weight(out) == pytest.approx(1.0)


@pytest.mark.unit()
class TestSplitLowResampler:
    """Covers the "split the lowest-pcoord walker" policy."""

    def test_split_targets_lowest_pcoord(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = SplitLowResampler(num_resamples=1, n_split=2)
        nxt = [
            sim_factory(weight=0.25, parent_pcoord=[5.0]),
            sim_factory(weight=0.25, parent_pcoord=[1.0]),  # lowest
            sim_factory(weight=0.25, parent_pcoord=[3.0]),
            sim_factory(weight=0.25, parent_pcoord=[4.0]),
        ]
        out = resampler.split(nxt)
        # Two new sims with weight 0.125 are created from the lowest sim.
        fragment_count = sum(
            1 for s in out if s.weight == pytest.approx(0.125)
        )
        assert fragment_count == 2
        assert _total_weight(out) == pytest.approx(1.0)

    def test_full_resample_conserves_count_and_weight(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = SplitLowResampler(num_resamples=1, n_split=2)
        cur = [
            sim_factory(
                weight=0.25,
                simulation_id=i,
                pcoord=[[0.0], [float(i)]],
            )
            for i in range(4)
        ]
        nxt = [
            sim_factory(
                weight=0.25,
                simulation_id=i + 10,
                parent_pcoord=[float(i)],
            )
            for i in range(4)
        ]
        np.random.seed(0)
        _, resampled = resampler.resample(cur, nxt)
        assert len(resampled) == len(nxt)
        assert _total_weight(resampled) == pytest.approx(_total_weight(nxt))


@pytest.mark.unit()
class TestSplitHighResampler:
    """Covers the "split the highest-pcoord walker" policy."""

    def test_split_targets_highest_pcoord(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = SplitHighResampler(num_resamples=1, n_split=3)
        nxt = [
            sim_factory(weight=0.3, parent_pcoord=[1.0]),
            sim_factory(weight=0.3, parent_pcoord=[2.0]),
            sim_factory(weight=0.4, parent_pcoord=[5.0]),  # highest
        ]
        out = resampler.split(nxt)
        # n_split=3 means the winner is split into 3 fragments of 0.4/3.
        fragment = pytest.approx(0.4 / 3)
        assert sum(1 for s in out if s.weight == fragment) == 3

    def test_merge_targets_lowest_pcoord(
        self,
        sim_factory: SimFactory,
    ) -> None:
        # The `high` resampler merges the lowest-pcoord sims (opposite of the
        # splitting target) so the ensemble count stays stable.
        resampler = SplitHighResampler(num_resamples=1, n_split=2)
        cur = [sim_factory(weight=0.25, simulation_id=i) for i in range(4)]
        nxt = [
            sim_factory(
                weight=0.25,
                simulation_id=i + 10,
                parent_pcoord=[float(i)],
            )
            for i in range(4)
        ]
        np.random.seed(0)
        merged = resampler.merge(cur, nxt)
        # num_merges = num_resamples + 1 = 2 -> 4 sims become 3.
        assert len(merged) == 3
        assert _total_weight(merged) == pytest.approx(1.0)

    def test_full_resample_conserves_count_and_weight(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = SplitHighResampler(num_resamples=1, n_split=2)
        cur = [
            sim_factory(
                weight=0.25,
                simulation_id=i,
                pcoord=[[0.0], [float(i)]],
            )
            for i in range(4)
        ]
        nxt = [
            sim_factory(
                weight=0.25,
                simulation_id=i + 10,
                parent_pcoord=[float(i)],
            )
            for i in range(4)
        ]
        np.random.seed(0)
        _, resampled = resampler.resample(cur, nxt)
        assert len(resampled) == len(nxt)
        assert _total_weight(resampled) == pytest.approx(_total_weight(nxt))


@pytest.mark.unit()
class TestHuberKimResampler:
    """Covers the Huber-Kim policy through `Resampler.run`."""

    def test_run_keeps_sims_per_bin(
        self,
        sim_factory: SimFactory,
        basis_states,
    ) -> None:
        resampler = HuberKimResampler(sims_per_bin=3)
        # Pad with -inf to keep all pcoords in a real interior bin.
        binner = RectilinearBinner(
            bins=[-np.inf, 0.0, 1.0, 2.0, 3.0],
            bin_target_counts=3,
        )
        recycler = LowRecycler(
            basis_states=basis_states,
            target_threshold=-np.inf,  # never recycle
        )
        cur = [
            sim_factory(
                weight=0.25,
                simulation_id=i,
                pcoord=[[0.0], [0.5 + i * 0.1]],
            )
            for i in range(4)
        ]
        np.random.seed(0)
        random.seed(0)
        _, new_sims, meta = resampler.run(cur, binner, recycler)
        # All sims land in the first bin (pcoord ~0.5), so we should get
        # exactly `sims_per_bin` walkers after resampling.
        assert len(new_sims) == 3
        assert _total_weight(new_sims) == pytest.approx(_total_weight(cur))
        assert meta.iteration_id == 1


@pytest.mark.unit()
class TestLOFLowResampler:
    """Covers the LOF-based two-step resampler."""

    def test_raises_when_too_few_sims(
        self,
        sim_factory: SimFactory,
    ) -> None:
        resampler = LOFLowResampler(consider_for_resampling=4, max_resamples=1)
        cur = [sim_factory() for _ in range(4)]
        nxt = [sim_factory() for _ in range(4)]
        with pytest.raises(ValueError, match='too large'):
            resampler.resample(cur, nxt)

    def test_resample_preserves_weight(
        self,
        sim_factory: SimFactory,
    ) -> None:
        random.seed(0)
        np.random.seed(0)
        n = 12
        resampler = LOFLowResampler(
            consider_for_resampling=3,
            max_resamples=2,
            max_allowed_weight=1.0,
            min_allowed_weight=1e-10,
        )
        cur = [sim_factory(weight=1.0 / n, simulation_id=i) for i in range(n)]
        nxt = [
            sim_factory(
                weight=1.0 / n,
                simulation_id=i + 100,
                # Two pcoord channels: rmsd (idx 0), lof score (idx 1).
                parent_pcoord=[float(i), float(n - i)],
            )
            for i in range(n)
        ]
        _, resampled = resampler.resample(deepcopy(cur), nxt)
        assert _total_weight(resampled) == pytest.approx(1.0, abs=1e-12)
