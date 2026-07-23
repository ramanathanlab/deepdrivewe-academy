"""Tests for `deepdrivewe.checkpoint`."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.binners import RectilinearBinner
from deepdrivewe.checkpoint import EnsembleCheckpointer


def _populate_cur_sims(ensemble: WeightedEnsemble) -> list[SimMetadata]:
    """Copy basis-state sims and give them a pcoord frame so io can write."""
    cur = deepcopy(ensemble.basis_states.basis_states)
    for sim in cur:
        # io._append_pcoords dereferences pcoord[0], so we need at least
        # one frame that matches the parent_pcoord dimension.
        sim.pcoord = [list(sim.parent_pcoord)]
    return cur


@pytest.mark.integration
class TestEnsembleCheckpointer:
    """Covers checkpoint save/load round-trips and latest-file discovery."""

    def _build_metadata(self) -> IterationMetadata:
        # Produce a binner pickle/hash the checkpointer will pass to HDF5.
        binner = RectilinearBinner(
            bins=[0.0, 1.0, 2.0],
            bin_target_counts=2,
        )
        pkl, h = binner.pickle_and_hash()
        return IterationMetadata(
            iteration_id=1,
            binner_pickle=pkl,
            binner_hash=h,
            min_bin_prob=0.4,
            max_bin_prob=0.6,
            bin_target_counts=[2, 2],
        )

    def test_save_creates_checkpoint_dir_and_file(
        self,
        tmp_path: Path,
        weighted_ensemble: WeightedEnsemble,
    ) -> None:
        weighted_ensemble.metadata = self._build_metadata()
        # Seed cur_sims so io.py can write a summary row (needs min/max).
        weighted_ensemble.cur_sims = _populate_cur_sims(weighted_ensemble)
        checkpointer = EnsembleCheckpointer(tmp_path)
        checkpointer.save(weighted_ensemble)

        checkpoint_dir = tmp_path / 'checkpoints'
        assert checkpoint_dir.is_dir()
        assert (checkpoint_dir / 'checkpoint-000001.json').is_file()
        assert (tmp_path / 'west.h5').is_file()

    def test_roundtrip(
        self,
        tmp_path: Path,
        weighted_ensemble: WeightedEnsemble,
    ) -> None:
        weighted_ensemble.metadata = self._build_metadata()
        weighted_ensemble.cur_sims = _populate_cur_sims(weighted_ensemble)
        checkpointer = EnsembleCheckpointer(tmp_path)
        checkpointer.save(weighted_ensemble)
        loaded = checkpointer.load()
        assert loaded.iteration == weighted_ensemble.iteration
        assert len(loaded.basis_states) == len(weighted_ensemble.basis_states)
        assert len(loaded.cur_sims) == len(weighted_ensemble.cur_sims)

    def test_latest_checkpoint_picks_highest(
        self,
        tmp_path: Path,
    ) -> None:
        checkpointer = EnsembleCheckpointer(tmp_path)
        (checkpointer.checkpoint_dir / 'checkpoint-000001.json').write_text(
            '{}',
        )
        (checkpointer.checkpoint_dir / 'checkpoint-000003.json').write_text(
            '{}',
        )
        (checkpointer.checkpoint_dir / 'checkpoint-000002.json').write_text(
            '{}',
        )
        latest = checkpointer.latest_checkpoint()
        assert latest is not None
        assert latest.name == 'checkpoint-000003.json'

    def test_load_without_checkpoints_raises(self, tmp_path: Path) -> None:
        checkpointer = EnsembleCheckpointer(tmp_path)
        with pytest.raises(FileNotFoundError):
            checkpointer.load()
