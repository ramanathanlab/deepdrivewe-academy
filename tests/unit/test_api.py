"""Tests for `deepdrivewe.api` — data models and YAML I/O."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deepdrivewe.api import BaseModel
from deepdrivewe.api import BasisStateInitializer
from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import TargetState
from deepdrivewe.api import validate_and_resolve_file
from deepdrivewe.api import WeightedEnsemble


@pytest.mark.unit()
class TestValidateAndResolveFile:
    """Covers the small validator helper."""

    def test_returns_none_for_none(self) -> None:
        assert validate_and_resolve_file(None) is None

    def test_raises_for_nonexistent(self, tmp_path: Path) -> None:
        missing = tmp_path / 'no_such_file'
        with pytest.raises(FileNotFoundError):
            validate_and_resolve_file(missing)

    def test_raises_for_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_and_resolve_file(tmp_path)

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        target = tmp_path / 'exists.txt'
        target.write_text('hi')
        resolved = validate_and_resolve_file(target)
        assert resolved == target.resolve()


@pytest.mark.unit()
class TestSimMetadata:
    """Covers `SimMetadata` validation, helpers and mutations."""

    def _base_kwargs(self, restart_file: Path) -> dict[str, Any]:
        return {
            'weight': 0.5,
            'simulation_id': 7,
            'iteration_id': 3,
            'parent_restart_file': restart_file,
            'parent_pcoord': [0.1, 0.2],
            'pcoord': [[1.0, 2.0], [3.0, 4.0]],
            'parent_simulation_id': 2,
            'wtg_parent_ids': [2],
        }

    def test_weight_must_be_in_unit_interval(self, tmp_path: Path) -> None:
        kwargs = self._base_kwargs(tmp_path / 'r.ncrst')
        kwargs['weight'] = 1.5
        with pytest.raises(ValidationError):
            SimMetadata(**kwargs)

        kwargs['weight'] = -0.1
        with pytest.raises(ValidationError):
            SimMetadata(**kwargs)

    def test_iteration_id_is_1_indexed(self, tmp_path: Path) -> None:
        kwargs = self._base_kwargs(tmp_path / 'r.ncrst')
        kwargs['iteration_id'] = 0
        with pytest.raises(ValidationError):
            SimMetadata(**kwargs)

    def test_walltime_and_marks(self, tmp_path: Path) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        assert sim.walltime == pytest.approx(0.0)
        sim.mark_simulation_start()
        time.sleep(0.001)
        sim.mark_simulation_end()
        assert sim.walltime > 0

    def test_simulation_name_format(self, tmp_path: Path) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        assert sim.simulation_name == '000003/000007'

    def test_num_frames_tracks_pcoord(self, tmp_path: Path) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        assert sim.num_frames == 2

    def test_append_pcoord_appends_per_frame(self, tmp_path: Path) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        sim.append_pcoord([9.0, 10.0])
        assert sim.pcoord == [[1.0, 2.0, 9.0], [3.0, 4.0, 10.0]]

    def test_append_pcoord_mismatched_frames_raises(
        self,
        tmp_path: Path,
    ) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        with pytest.raises(ValueError, match='number of frames'):
            sim.append_pcoord([9.0, 10.0, 11.0])

    def test_endpoint_type_defaults_to_1(self, tmp_path: Path) -> None:
        sim = SimMetadata(**self._base_kwargs(tmp_path / 'r.ncrst'))
        assert sim.endpoint_type == 1


@pytest.mark.unit()
class TestIterationMetadata:
    """Covers `IterationMetadata` defaults and serialization."""

    def test_defaults(self) -> None:
        meta = IterationMetadata()
        assert meta.iteration_id == 1
        assert meta.binner_pickle == b''
        assert meta.binner_hash == ''
        assert meta.bin_target_counts == []

    def test_iteration_id_minimum(self) -> None:
        with pytest.raises(ValidationError):
            IterationMetadata(iteration_id=0)

    def test_bin_prob_floating_point_slack(self) -> None:
        # Values up to 1.001 are allowed to absorb floating-point drift.
        IterationMetadata(max_bin_prob=1.001)
        with pytest.raises(ValidationError):
            IterationMetadata(max_bin_prob=1.01)

    def test_binner_pickle_excluded_from_json(self) -> None:
        meta = IterationMetadata(binner_pickle=b'\x00\x01\x02')
        payload = meta.model_dump_json()
        assert 'binner_pickle' not in payload
        assert 'binner_hash' in payload


@pytest.mark.unit()
class TestBaseModelYAML:
    """YAML dump/load round-trip on the shared `BaseModel`."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        state = TargetState(label='bound', pcoord=[0.1, 0.2])
        yaml_path = tmp_path / 'state.yaml'
        state.dump_yaml(yaml_path)
        loaded: TargetState = BaseModel.from_yaml.__func__(  # type: ignore[attr-defined]
            TargetState,
            yaml_path,
        )
        assert loaded.label == 'bound'
        assert loaded.pcoord == [0.1, 0.2]

    def test_roundtrip_via_classmethod(self, tmp_path: Path) -> None:
        state = TargetState(label='bound', pcoord=[0.5])
        yaml_path = tmp_path / 'state.yaml'
        state.dump_yaml(yaml_path)
        loaded = TargetState.from_yaml(yaml_path)
        assert loaded == state


@pytest.mark.unit()
class TestBasisStates:
    """Covers basis state loading, iteration and uniform init."""

    def test_rejects_non_directory(self, tmp_path: Path) -> None:
        bogus = tmp_path / 'not_a_dir'
        bogus.write_text('x')
        # Pydantic's field_validator lets NotADirectoryError bubble through
        # rather than wrapping it in ValidationError.
        with pytest.raises((ValidationError, NotADirectoryError)):
            BasisStates(
                basis_state_dir=bogus,
                initial_ensemble_members=2,
            )

    def test_loads_and_assigns_uniform_weight(
        self,
        basis_state_dir: Path,
        simple_initializer: BasisStateInitializer,
    ) -> None:
        states = BasisStates(
            basis_state_dir=basis_state_dir,
            initial_ensemble_members=6,
        )
        states.load_basis_states(simple_initializer)

        # Six ensemble members, three unique files — cycle wraps.
        assert len(states) == 6
        assert states.num_basis_files == 3
        expected_weight = 1.0 / 6
        for sim in states:
            assert sim.weight == pytest.approx(expected_weight)
            assert sim.iteration_id == 1
            # Negative parent_simulation_id flags basis-state origin.
            assert sim.parent_simulation_id is not None
            assert sim.parent_simulation_id < 0

    def test_unique_basis_states_slice(
        self,
        basis_state_dir: Path,
        simple_initializer: BasisStateInitializer,
    ) -> None:
        states = BasisStates(
            basis_state_dir=basis_state_dir,
            initial_ensemble_members=6,
        )
        states.load_basis_states(simple_initializer)
        unique = states.unique_basis_states
        assert len(unique) == 3
        assert {s.simulation_id for s in unique} == {0, 1, 2}

    def test_empty_directory_raises(
        self,
        tmp_path: Path,
        simple_initializer: BasisStateInitializer,
    ) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        states = BasisStates(
            basis_state_dir=empty,
            initial_ensemble_members=2,
        )
        with pytest.raises(ValueError, match='No basis states'):
            states.load_basis_states(simple_initializer)

    def test_subdir_without_matching_file_raises(
        self,
        tmp_path: Path,
        simple_initializer: BasisStateInitializer,
    ) -> None:
        root = tmp_path / 'bad'
        (root / 'system1').mkdir(parents=True)
        (root / 'system1' / 'wrong.ext').write_text('x')
        states = BasisStates(
            basis_state_dir=root,
            initial_ensemble_members=1,
        )
        with pytest.raises(FileNotFoundError):
            states.load_basis_states(simple_initializer)

    def test_random_initialization_is_seed_deterministic(
        self,
        basis_state_dir: Path,
        simple_initializer: BasisStateInitializer,
    ) -> None:
        def _load(seed: int) -> list[int]:
            states = BasisStates(
                basis_state_dir=basis_state_dir,
                initial_ensemble_members=2,
                randomly_initialize=True,
                random_seed=seed,
            )
            states.load_basis_states(simple_initializer)
            return [int(s.parent_pcoord[0]) for s in states]

        assert _load(42) == _load(42)

    def test_getitem_and_iter(
        self,
        basis_states: BasisStates,
    ) -> None:
        assert basis_states[0] == next(iter(basis_states))
        assert len(list(iter(basis_states))) == len(basis_states)


@pytest.mark.unit()
class TestWeightedEnsemble:
    """Covers `WeightedEnsemble` lifecycle helpers."""

    def test_iteration_property(
        self,
        weighted_ensemble: WeightedEnsemble,
    ) -> None:
        assert weighted_ensemble.iteration == 1

    def test_initialize_basis_states_populates_next_sims(
        self,
        basis_states: BasisStates,
        target_states: list[TargetState],
        simple_initializer: BasisStateInitializer,
    ) -> None:
        # basis_states fixture already loaded, but round-trip via the
        # WeightedEnsemble entry point to exercise the wrapper.
        fresh = BasisStates(
            basis_state_dir=basis_states.basis_state_dir,
            initial_ensemble_members=4,
        )
        ensemble = WeightedEnsemble(
            basis_states=fresh,
            target_states=target_states,
        )
        ensemble.initialize_basis_states(simple_initializer)
        assert len(ensemble.next_sims) == 4
        # deepcopy guarantee: mutation doesn't leak back to basis_states.
        ensemble.next_sims[0].weight = 0.0
        assert ensemble.basis_states[0].weight != 0.0

    def test_advance_iteration(
        self,
        weighted_ensemble: WeightedEnsemble,
    ) -> None:
        cur = deepcopy(weighted_ensemble.basis_states.basis_states[:2])
        nxt = deepcopy(weighted_ensemble.basis_states.basis_states[:2])
        meta = IterationMetadata(iteration_id=2, min_bin_prob=0.1)
        weighted_ensemble.advance_iteration(
            cur_sims=cur,
            next_sims=nxt,
            metadata=meta,
        )
        assert weighted_ensemble.iteration == 2
        assert weighted_ensemble.cur_sims == cur
        assert weighted_ensemble.next_sims == nxt


@pytest.mark.unit()
def test_target_state_requires_pcoord() -> None:
    with pytest.raises(ValidationError):
        TargetState.model_validate({'label': 'unbound'})
