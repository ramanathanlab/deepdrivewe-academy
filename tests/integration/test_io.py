"""Tests for `deepdrivewe.io` — the WESTPA HDF5 writer."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import h5py
import pytest

from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import TargetState
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.io import WEST_FILEFORMAT_VERSION
from deepdrivewe.io import WEST_ITER_PREC
from deepdrivewe.io import WestpaH5File

SimFactory = Callable[..., SimMetadata]


def _metadata_for(iteration: int) -> IterationMetadata:
    binner = RectilinearBinner(
        bins=[0.0, 1.0, 2.0],
        bin_target_counts=2,
    )
    pkl, h = binner.pickle_and_hash()
    return IterationMetadata(
        iteration_id=iteration,
        binner_pickle=pkl,
        binner_hash=h,
        min_bin_prob=0.3,
        max_bin_prob=0.7,
        bin_target_counts=[2, 2],
    )


@pytest.mark.integration
class TestWestpaH5File:
    """Covers creation and single-iteration append on the HDF5 writer."""

    def test_creates_file_with_expected_attrs(self, tmp_path: Path) -> None:
        path = tmp_path / 'west.h5'
        WestpaH5File(path)
        assert path.is_file()
        with h5py.File(path, 'r') as f:
            assert (
                f.attrs['west_file_format_version'] == WEST_FILEFORMAT_VERSION
            )
            assert f.attrs['west_iter_prec'] == WEST_ITER_PREC
            assert f.attrs['west_current_iteration'] == 1
            assert 'summary' in f
            assert 'iterations' in f

    def test_append_creates_iteration_group_and_summary_row(
        self,
        tmp_path: Path,
        sim_factory: SimFactory,
        basis_states: BasisStates,
    ) -> None:
        path = tmp_path / 'west.h5'
        writer = WestpaH5File(path)
        cur = [
            sim_factory(
                weight=0.25,
                simulation_id=i,
                pcoord=[[0.0], [0.5 + i * 0.1]],
            )
            for i in range(4)
        ]
        target_states = [TargetState(label='t', pcoord=[0.0])]
        meta = _metadata_for(iteration=1)
        writer.append(
            cur_sims=cur,
            basis_states=basis_states,
            target_states=target_states,
            metadata=meta,
        )

        with h5py.File(path, 'r') as f:
            # Summary row added for the new iteration.
            assert len(f['summary']) == 1
            summary = f['summary'][0]
            assert summary['n_particles'] == 4
            assert summary['norm'] == pytest.approx(1.0)
            assert summary['min_bin_prob'] == pytest.approx(0.3)
            assert summary['max_bin_prob'] == pytest.approx(0.7)

            # Iteration group was created with the right attrs.
            iter_group = f[
                '/iterations/iter_{:0{p}d}'.format(
                    1,
                    p=WEST_ITER_PREC,
                )
            ]
            assert iter_group.attrs['n_iter'] == 1
            assert 'seg_index' in iter_group
            assert 'pcoord' in iter_group
            assert 'bin_target_counts' in iter_group

            # Topology groups were populated.
            assert 'ibstates' in f
            assert 'tstates' in f
            assert 'bin_topologies' in f

    def test_append_rejects_mismatched_pcoord_dim(
        self,
        tmp_path: Path,
        sim_factory: SimFactory,
        basis_states: BasisStates,
    ) -> None:
        path = tmp_path / 'west.h5'
        writer = WestpaH5File(path)
        cur = [
            sim_factory(
                weight=1.0,
                simulation_id=0,
                parent_pcoord=[0.0, 0.0],  # 2D
                pcoord=[[1.0]],  # 1D -> mismatch
            ),
        ]
        with pytest.raises(ValueError):
            writer.append(
                cur_sims=cur,
                basis_states=basis_states,
                target_states=[],
                metadata=_metadata_for(iteration=1),
            )

    def test_append_rejects_mismatched_frame_counts(
        self,
        tmp_path: Path,
        sim_factory: SimFactory,
        basis_states: BasisStates,
    ) -> None:
        path = tmp_path / 'west.h5'
        writer = WestpaH5File(path)
        a = sim_factory(
            simulation_id=0,
            weight=0.5,
            pcoord=[[0.0], [1.0]],
        )
        b = sim_factory(
            simulation_id=1,
            weight=0.5,
            pcoord=[[0.0]],
        )
        with pytest.raises(ValueError):
            writer.append(
                cur_sims=[deepcopy(a), deepcopy(b)],
                basis_states=basis_states,
                target_states=[],
                metadata=_metadata_for(iteration=1),
            )
