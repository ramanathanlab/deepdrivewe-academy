"""Minimal example of using Academy to implement DeepDriveWE pattern.

Usage
-----
Run locally (default, no authentication required)::

    python examples/minimal_pattern/main.py

Run via the Academy Exchange Cloud (requires Globus authentication)::

    python examples/minimal_pattern/main.py --exchange globus
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from academy.agent import action
from academy.agent import Agent
from academy.agent import loop
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.handle import Handle
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from pydantic import BaseModel
from pydantic import Field

from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult

EXCHANGE_ADDRESS = 'https://exchange.academy-agents.org'


class SimulationAgent(Agent):
    """Agent for simulation."""

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    def __init__(
        self,
        train_handle: Handle[TrainingAgent],
        inference_handle: Handle[InferenceAgent],
    ) -> None:
        super().__init__()
        self.train_handle = train_handle
        self.inference_handle = inference_handle

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def simulate(self, sim_metadata: SimMetadata) -> None:
        """Run a simulation and send results to training/inference."""
        self.__logger.info(
            f'running simulation {sim_metadata.simulation_id} '
            f'iteration {sim_metadata.iteration_id}',
        )

        # Create mock simulation result with empty data products
        result = SimResult(
            data={
                'contact_maps': np.array([]),
                'rmsd': np.array([]),
            },
            metadata=sim_metadata,
        )

        self.__logger.info(
            f'simulation {sim_metadata.simulation_id} complete, '
            f'sending results',
        )

        # Send the result to the training and inference agents.
        await self.train_handle.receive_simulation_data(result)
        await self.inference_handle.receive_simulation_data(result)


class TrainingAgent(Agent):
    """Agent for training."""

    # A queue for receiving data from the simulation agent.
    __queue: asyncio.Queue[SimResult]

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    def __init__(self, inference_handle: Handle[InferenceAgent]) -> None:
        super().__init__()
        self.inference_handle = inference_handle

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]
        self.__queue = asyncio.Queue()

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def receive_simulation_data(self, result: SimResult) -> None:
        """Receive simulation result data for training."""
        self.__logger.info(
            f'received simulation data for sim '
            f'{result.metadata.simulation_id} '
            f'iteration {result.metadata.iteration_id}.',
        )
        await self.__queue.put(result)

    @loop
    async def train(self, shutdown: asyncio.Event) -> None:
        """Train on simulation results as they arrive."""
        while not shutdown.is_set():
            result = await self.__queue.get()
            self.__logger.info(
                f'training on sim {result.metadata.simulation_id} data',
            )

            # Mock training: produce a model weights path
            model_weights_path = (
                f'weights/iter{result.metadata.iteration_id}/model.pt'
            )
            self.__logger.info(
                f'trained model: {model_weights_path}',
            )
            self.__queue.task_done()

            # Send the model weights path to the inference agent.
            await self.inference_handle.receive_model_weights(
                model_weights_path,
            )


class InferenceAgent(Agent):
    """Agent for inference."""

    # An integer to keep track of the iteration.
    iteration: int

    # The number of simulations to collect before running inference.
    num_simulations: int

    # The maximum number of iterations to run before shutting down.
    max_iterations: int

    # The handles of the simulation agents to send new simulation metadata to.
    simulation_handles: list[Handle[SimulationAgent]]

    # The model weights path.
    model_weights_path: str

    # Collected simulation results for the current iteration.
    __batch: list[SimResult]

    # An event to signal that a full batch is ready for inference.
    __batch_ready: asyncio.Event

    # A lock to ensure that the model weights are not updated while
    # inference is happening.
    __model_lock: asyncio.Lock

    # A logger for logging exceptions in background tasks.
    __logger: logging.Logger

    def __init__(
        self,
        num_simulations: int,
        max_iterations: int,
        simulation_handles: list[Handle[SimulationAgent]],
    ) -> None:
        super().__init__()
        self.num_simulations = num_simulations
        self.max_iterations = max_iterations
        self.simulation_handles = simulation_handles

    async def agent_on_startup(self) -> None:
        """Startup."""
        self.__logger = logging.getLogger(__class__.__name__)  # type: ignore[name-defined]
        self.__batch = []
        self.__batch_ready = asyncio.Event()
        self.__model_lock = asyncio.Lock()

        # Initialize the iteration counter.
        self.iteration = 1

        # Load the initial model weights.
        self.model_weights_path = 'path/to/model/weights.pt'

        # Log that the agent has started after initializing the logger.
        self.__logger.info('started')

    @action
    async def receive_simulation_data(
        self,
        result: SimResult,
    ) -> None:
        """Receive a simulation result and buffer it."""
        self.__logger.info(
            f'received result for sim '
            f'{result.metadata.simulation_id} '
            f'iteration {result.metadata.iteration_id}. '
            f'batch: {len(self.__batch) + 1}'
            f'/{self.num_simulations}',
        )
        self.__batch.append(result)

        # Signal the infer loop when all results are collected.
        if len(self.__batch) >= self.num_simulations:
            self.__batch_ready.set()

    @action
    async def receive_model_weights(
        self,
        model_weights_path: str,
    ) -> None:
        """Receive updated model weights from the training agent."""
        self.__logger.info(
            f'received model weights: {model_weights_path}.',
        )
        async with self.__model_lock:
            self.model_weights_path = model_weights_path

    @loop
    async def infer(self, shutdown: asyncio.Event) -> None:
        """Wait for a full batch then run inference."""
        while not shutdown.is_set():
            # Wait until all simulation results are collected.
            await self.__batch_ready.wait()
            self.__batch_ready.clear()

            # Grab the batch and reset for the next iteration.
            batch = self.__batch
            self.__batch = []

            self.__logger.info(
                f'running inference on {len(batch)} results '
                f'with model {self.model_weights_path}',
            )

            # Mock inference: use model weights on collected data.
            async with self.__model_lock:
                for result in batch:
                    self.__logger.info(
                        f'  infer sim '
                        f'{result.metadata.simulation_id}: '
                        f'rmsd={result.data.get("rmsd")}',
                    )

            # Update the iteration counter.
            self.iteration += 1

            # Check if we've reached the max iteration for this example.
            if self.iteration > self.max_iterations:
                self.__logger.info(
                    f'reached max iterations ({self.max_iterations}), '
                    f'shutting down.',
                )
                shutdown.set()
                # TODO: Do we also need to call self.agent_shutdown()?
                return

            self.__logger.info(
                f'inference complete for iteration {self.iteration - 1}, '
                f'kicking off iteration {self.iteration}.',
            )

            # Kick off the next iteration of simulations by
            # sending new SimMetadata to each simulation agent.
            for idx, sim_handle in enumerate(
                self.simulation_handles,
            ):
                next_metadata = SimMetadata(
                    weight=1.0 / self.num_simulations,
                    simulation_id=idx,
                    iteration_id=self.iteration,
                    parent_restart_file=Path(
                        f'restart/{self.iteration}/{idx}.rst',
                    ),
                    parent_pcoord=[0.0],
                )
                await sim_handle.simulate(next_metadata)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Minimal DeepDriveWE pattern example.',
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
    """Create the exchange factory based on the factory type."""
    if exchange_type == 'local':
        return LocalExchangeFactory()

    # NOTE: If using the cloud exchange, run the authentication prior to
    # submitting a batch job script. This will cache a Globus auth session
    # token on the machine that will be reused.

    # Use the HttpExchangeFactory to connect to the Academy Exchange Cloud.
    # This makes all agents talk to each other through the cloud, which
    # allows them to run on different machines with easier setup.
    return HttpExchangeFactory(url=EXCHANGE_ADDRESS, auth_method='globus')


class DeepDriveWeConfig(BaseModel):
    """Configuration for DeepDriveWE pattern."""

    max_iterations: int = Field(
        default=7,
        description='Number of iterations to run the pattern for.',
    )

    num_simulations: int = Field(
        default=2,
        description='Number of simulation agents to launch.',
    )


async def main() -> None:
    """Run the main function."""
    args = parse_args()

    # Load the configuration
    config = DeepDriveWeConfig()

    async with await Manager.from_exchange_factory(
        factory=create_exchange_factory(args.exchange),
        executors=ThreadPoolExecutor(),
        log_config=recommended_logging('INFO'),
    ) as manager:
        # Register the agents with the manager (this will create the
        # mailboxes for the agents).
        reg_inference_agent = await manager.register_agent(InferenceAgent)
        reg_training_agent = await manager.register_agent(TrainingAgent)
        reg_simulation_agents = await asyncio.gather(
            *[
                manager.register_agent(SimulationAgent)
                for _ in range(config.num_simulations)
            ],
        )

        print('num simulation agents registered:', len(reg_simulation_agents))

        # Get the handle of each agent from the manager.
        inference_handle = manager.get_handle(reg_inference_agent)
        training_handle = manager.get_handle(reg_training_agent)
        simulation_handles = [
            manager.get_handle(reg_simulation_agent)
            for reg_simulation_agent in reg_simulation_agents
        ]

        print('num simulation handles:', len(simulation_handles))

        # Launch the agents (this will start the agent_on_startup method of
        # each agent).
        inference_handle = await manager.launch(
            InferenceAgent,
            registration=reg_inference_agent,
            args=(
                config.num_simulations,
                config.max_iterations,
                simulation_handles,
            ),
        )

        training_handle = await manager.launch(
            TrainingAgent,
            registration=reg_training_agent,
            args=(inference_handle,),
        )

        simulation_agents = await asyncio.gather(
            *[
                manager.launch(
                    SimulationAgent,
                    registration=reg_simulation_agent,
                    args=(training_handle, inference_handle),
                )
                for reg_simulation_agent in reg_simulation_agents
            ],
        )

        # Kick off the first iteration of simulations
        initial_metadata = [
            SimMetadata(
                weight=1.0 / config.num_simulations,
                simulation_id=idx,
                iteration_id=1,
                parent_restart_file=Path(
                    f'restart/1/{idx}.rst',
                ),
                parent_pcoord=[0.0],
            )
            for idx in range(config.num_simulations)
        ]
        await asyncio.gather(
            *[
                agent.simulate(meta)
                for agent, meta in zip(
                    simulation_agents,
                    initial_metadata,
                    strict=True,
                )
            ],
        )

        # Wait until the inference agent signals that it is done by shutting
        # down.
        await manager.wait((inference_handle,))


if __name__ == '__main__':
    asyncio.run(main())
