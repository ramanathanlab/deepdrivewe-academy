"""Unit tests for compute-config worker-slot accounting."""

from __future__ import annotations

import pytest

from deepdrivewe.parsl import LocalConfig
from deepdrivewe.parsl import PolarisConfig
from deepdrivewe.parsl import VistaConfig
from deepdrivewe.parsl import WorkstationConfig


def test_local_num_workers_matches_max_workers() -> None:
    """Local slots equal the configured worker count."""
    assert LocalConfig(max_workers_per_node=3).num_workers == 3


def test_workstation_num_workers_from_int() -> None:
    """An integer accelerator count is returned as-is."""
    assert WorkstationConfig(available_accelerators=2).num_workers == 2


def test_workstation_num_workers_from_device_list() -> None:
    """A device-id list is counted by length."""
    config = WorkstationConfig(available_accelerators=['0', '1', '2', '3'])
    assert config.num_workers == 4


def test_vista_num_workers_is_one_per_node() -> None:
    """VISTA uses one GH accelerator per node."""
    assert VistaConfig(num_nodes=3).num_workers == 3


def test_polaris_num_workers_is_four_per_node() -> None:
    """Polaris has four GPUs per node."""
    assert PolarisConfig(num_nodes=2).num_workers == 8


def test_polaris_gpus_per_node_is_not_a_config_field() -> None:
    """The per-node GPU count is a fixed constant, not configurable."""
    assert 'gpus_per_node' not in PolarisConfig.model_fields


@pytest.mark.parametrize(
    'config',
    (
        LocalConfig(max_workers_per_node=1),
        WorkstationConfig(available_accelerators=1),
        VistaConfig(num_nodes=3),
        PolarisConfig(num_nodes=1),
    ),
)
def test_num_workers_is_positive(config: object) -> None:
    """Every compute config reports at least one worker slot."""
    assert config.num_workers >= 1  # type: ignore[attr-defined]
