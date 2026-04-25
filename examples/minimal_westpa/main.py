"""Minimal WESTPA example showing agent subclassing.

Demonstrates how to subclass ``SimulationAgent`` and
``WestpaAgent`` to inject custom simulation and inference
logic with stateful agents.

For simple stateless cases, see ``run_westpa_workflow``
which accepts plain callables instead of subclasses.

Usage
-----
Run locally (default, no authentication required)::

    python examples/minimal_westpa/main.py

Run via the Academy Exchange Cloud (requires Globus
authentication)::

    python examples/minimal_westpa/main.py --exchange globus
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager
from pydantic import BaseModel
from pydantic import Field

from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.workflows.westpa import run_westpa_workflow
from deepdrivewe.workflows.westpa import SimulationAgent
from deepdrivewe.workflows.westpa import WestpaAgent

EXCHANGE_ADDRESS = 'https://exchange.academy-agents.org'


class MockSimAgent(SimulationAgent):
    """Simulation agent with stateful RNG.

    Demonstrates how ``agent_on_startup`` can initialize
    expensive state (e.g., an ML model) that persists across
    simulation calls.
    """

    def __init__(
        self,
        westpa_handle: Handle[WestpaAgent],
        restart_dir: Path,
        logfile: Path | None = None,
    ) -> None:
        super().__init__(westpa_handle, logfile=logfile)
        self.restart_dir = Path(restart_dir)

    async def agent_on_startup(self) -> None:
        """Initialize the RNG on startup."""
        await super().agent_on_startup()
        self.rng = np.random.default_rng()

    def run_simulation(
        self,
        metadata: SimMetadata,
    ) -> SimResult:
        """Mock simulation: generate a random pcoord.

        A real restart file is materialized on disk so the next
        iteration's ``wait_for_file`` resolves immediately rather
        than burning through the (8-retry, 1s base delay) backoff.
        """
        pcoord_value = float(self.rng.uniform(0.5, 10.0))
        metadata.pcoord = [[pcoord_value]]
        restart_file = (
            self.restart_dir
            / f'iter_{metadata.iteration_id}_sim_{metadata.simulation_id}.rst'
        )
        restart_file.touch()
        metadata.restart_file = restart_file
        return SimResult(
            data={'pcoord': np.array([[pcoord_value]])},
            metadata=metadata,
        )


class MockWestpaAgent(WestpaAgent):
    """WESTPA agent with mock resampling logic.

    In a real workflow, override ``run_inference`` to use
    ``RectilinearBinner``, ``LowRecycler``, and
    ``HuberKimResampler``.
    """

    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Propagate sims to the next iteration."""
        cur_sims = [r.metadata for r in sim_results]
        next_sims = []
        for idx, sim in enumerate(cur_sims):
            assert sim.restart_file is not None
            next_sims.append(
                SimMetadata(
                    weight=sim.weight,
                    simulation_id=idx,
                    iteration_id=sim.iteration_id + 1,
                    parent_restart_file=sim.restart_file,
                    parent_pcoord=sim.pcoord[-1],
                    parent_simulation_id=sim.simulation_id,
                    wtg_parent_ids=[sim.simulation_id],
                ),
            )
        metadata = IterationMetadata(
            iteration_id=cur_sims[0].iteration_id,
        )
        return cur_sims, next_sims, metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Minimal WESTPA pattern example.',
    )
    parser.add_argument(
        '--exchange',
        choices=['local', 'globus'],
        default='local',
        help='Exchange type (default: local).',
    )
    return parser.parse_args()


def create_exchange_factory(
    exchange_type: str,
) -> LocalExchangeFactory | HttpExchangeFactory:
    """Create the exchange factory based on the type."""
    if exchange_type == 'local':
        return LocalExchangeFactory()
    return HttpExchangeFactory(url=EXCHANGE_ADDRESS, auth_method='globus')


class WestpaConfig(BaseModel):
    """Configuration for the minimal WESTPA example."""

    max_iterations: int = Field(
        default=5,
        description='Number of WE iterations to run.',
    )
    num_simulations: int = Field(
        default=4,
        description='Number of simulation walkers.',
    )


async def main() -> None:
    """Run the minimal WESTPA workflow."""
    args = parse_args()
    init_logging('INFO')

    config = WestpaConfig()

    # Use a temp dir for both basis states and restart files so the
    # example is self-contained and runs from any working directory.
    # `wait_for_file` would otherwise burn ~4 minutes of retry backoff
    # on missing parent_restart_files.
    with tempfile.TemporaryDirectory(prefix='minimal-westpa-') as tmp:
        restart_dir = Path(tmp)

        # Seed the initial parent_restart_files on disk.
        initial_sims: list[SimMetadata] = []
        for idx in range(config.num_simulations):
            seed = restart_dir / f'initial_{idx}.rst'
            seed.touch()
            initial_sims.append(
                SimMetadata(
                    weight=1.0 / config.num_simulations,
                    simulation_id=idx,
                    iteration_id=1,
                    parent_restart_file=seed,
                    parent_pcoord=[0.0],
                ),
            )

        # Create a mock ensemble with synthetic initial sims.
        # In a real workflow, use ensemble.initialize_basis_states()
        # to load from actual simulation restart files.
        ensemble = WeightedEnsemble(
            basis_states=BasisStates(
                basis_state_dir=restart_dir,
                initial_ensemble_members=config.num_simulations,
            ),
            target_states=[TargetState(pcoord=[0.0])],
            next_sims=initial_sims,
        )

        async with await Manager.from_exchange_factory(
            factory=create_exchange_factory(args.exchange),
            executors=ThreadPoolExecutor(),
        ) as manager:
            await run_westpa_workflow(
                manager=manager,
                sim_agent_type=MockSimAgent,
                westpa_agent_type=MockWestpaAgent,
                max_iterations=config.max_iterations,
                ensemble=ensemble,
                sim_agent_kwargs={'restart_dir': restart_dir},
            )


if __name__ == '__main__':
    asyncio.run(main())
