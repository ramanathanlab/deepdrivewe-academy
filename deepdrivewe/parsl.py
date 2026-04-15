"""Utilities to build Parsl configurations."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import WrappedLauncher
from parsl.providers import LocalProvider
from pydantic import BaseModel
from pydantic import Field


class BaseComputeConfig(BaseModel, ABC):
    """Compute config (HPC platform, number of GPUs, etc)."""

    # Name of the platform to uniquely identify it
    name: Literal[''] = ''

    @abstractmethod
    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : str | Path
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class LocalConfig(BaseComputeConfig):
    """Local compute config."""

    name: Literal['local'] = 'local'  # type: ignore[assignment]

    max_workers_per_node: int = Field(
        default=1,
        description='Number of workers to use.',
    )
    cores_per_worker: float = Field(
        default=1.0,
        description='Number of cores per worker.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    label: str = Field(
        default='cpu_htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for local execution."""
        return Config(
            run_dir=str(run_dir),
            strategy=None,
            executors=[
                HighThroughputExecutor(
                    address='127.0.0.1',
                    label=self.label,
                    max_workers_per_node=self.max_workers_per_node,
                    cores_per_worker=self.cores_per_worker,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class WorkstationConfig(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal['workstation'] = 'workstation'  # type: ignore[assignment]

    available_accelerators: int | Sequence[str] = Field(
        default=1,
        description='Number of GPU accelerators to use.',
    )
    worker_port_range: tuple[int, int] = Field(
        default=(10000, 20000),
        description='Port range for the workers.',
    )
    address: str = Field(
        default='127.0.0.1',
        description='Address for the workers to connect to.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )
    label: str = Field(
        default='htex',
        description='Label for the executor.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration for workstation execution."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    address=self.address,
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                ),
            ],
        )


class VistaConfig(BaseComputeConfig):
    """VISTA compute config.

    https://tacc.utexas.edu/systems/vista/
    """

    name: Literal['vista'] = 'vista'  # type: ignore[assignment]

    num_nodes: int = Field(
        ge=3,
        description='Number of nodes to use (must use at least 3 nodes).',
    )

    # We have a long idletime to ensure train/inference executors are not
    # shut down (to enable warmstarts) while simulations are running.
    max_idletime: float = Field(
        default=60.0 * 10,
        description='The maximum idle time allowed for an executor before '
        'strategy could shut down unused blocks. Default is 10 minutes.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration."""
        return Config(
            run_dir=str(run_dir),
            max_idletime=self.max_idletime,
            executors=[
                HighThroughputExecutor(
                    label='htex',
                    available_accelerators=1,  # 1 GH per node
                    cores_per_worker=72,
                    cpu_affinity='alternating',
                    prefetch_capacity=0,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            prepend=(
                                'srun -l'
                                ' --ntasks-per-node=1'
                                f' --nodes={self.num_nodes}'
                            ),
                        ),
                        cmd_timeout=120,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        max_blocks=1,
                    ),
                ),
            ],
        )


class PolarisConfig(BaseComputeConfig):
    """Compute config for a workstation."""

    name: Literal['polaris'] = 'polaris'  # type: ignore[assignment]

    num_nodes: int = Field(
        ge=1,
        description='Number of nodes to use.',
    )
    retries: int = Field(
        default=1,
        description='Number of retries for the task.',
    )
    max_idletime: float = Field(
        default=60.0 * 10,
        description='The maximum idle time allowed for an executor before '
        'strategy could shut down unused blocks. Default is 10 minutes.',
    )

    def get_parsl_config(self, run_dir: str | Path) -> Config:
        """Generate a Parsl configuration."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            max_idletime=self.max_idletime,
            executors=[
                HighThroughputExecutor(
                    label='htex',
                    cpu_affinity='block-reverse',
                    available_accelerators=4,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            prepend='mpiexec'
                            f' -n {self.num_nodes}'
                            ' --ppn 1'
                            ' --depth=64'
                            ' --cpu-bind depth',
                        ),
                        cmd_timeout=120,
                        nodes_per_block=self.num_nodes,
                        init_blocks=1,
                        max_blocks=1,
                    ),
                ),
            ],
        )


ComputeConfigTypes = (
    LocalConfig | WorkstationConfig | VistaConfig | PolarisConfig
)
