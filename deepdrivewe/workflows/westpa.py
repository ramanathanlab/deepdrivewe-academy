"""WESTPA workflow agents for the Academy framework.

Provides abstract base agents for running weighted ensemble
(WESTPA) workflows. Users subclass ``SimulationAgent`` and
``WestpaAgent`` to inject their own simulation and inference
logic, with full access to agent state (e.g., cached ML
models, loaded configurations).

Example
-------
::

    class MySimAgent(SimulationAgent):
        def __init__(self, westpa_handle, model):
            super().__init__(westpa_handle)
            self.model = model

        def run_simulation(self, metadata):
            return simulate(metadata, self.model)

    class MyWestpaAgent(WestpaAgent):
        def __init__(self, *args, basis, **kwargs):
            super().__init__(*args, **kwargs)
            self.basis = basis

        def run_inference(self, sim_results):
            return resample(sim_results, self.basis)

    await run_westpa_workflow(
        manager=mgr,
        sim_agent_type=MySimAgent,
        westpa_agent_type=MyWestpaAgent,
        max_iterations=100,
        ensemble=ensemble,
        num_sim_agents=num_gpus,
        sim_agent_kwargs={'model': my_model},
        westpa_agent_kwargs={'basis': my_basis},
    )
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import aiohttp
from academy.agent import action
from academy.agent import Agent
from academy.agent import loop
from academy.handle import Handle
from academy.manager import Manager

from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.utils import wait_for_file


async def dispatch_round_robin(
    handles: list[Handle[SimulationAgent]],
    sims: list[SimMetadata],
    max_retries: int = 3,
) -> None:
    """Dispatch simulations to agents round-robin.

    Retries each send up to ``max_retries`` times on transient errors
    (e.g. exchange timeout or connection drop).
    """

    async def _send(handle: Handle[SimulationAgent], sim: SimMetadata) -> None:
        for attempt in range(max_retries):
            try:
                await handle.simulate(sim)
                return
            except Exception as exc:
                if attempt == max_retries - 1 or not isinstance(
                    exc,
                    (
                        aiohttp.ClientConnectionError,
                        aiohttp.ClientPayloadError,
                        asyncio.TimeoutError,
                    ),
                ):
                    raise
                await asyncio.sleep(2.0**attempt)

    await asyncio.gather(
        *[_send(handles[i % len(handles)], sim) for i, sim in enumerate(sims)],
    )


class SimulationAgent(Agent, ABC):
    """Base agent for running simulations.

    Subclass and override ``run_simulation`` to provide custom
    simulation logic. Use ``agent_on_startup`` to initialize
    expensive state (e.g., load an ML model).

    The ``simulate`` action offloads ``run_simulation`` to a
    thread pool via ``agent_run_sync`` since MD simulations
    are typically blocking.

    Parameters
    ----------
    westpa_handle : Handle[WestpaAgent]
        Handle to send results to the WESTPA agent.
    """

    westpa_handle: Handle[WestpaAgent]
    logger: logging.Logger

    def __init__(
        self,
        westpa_handle: Handle[WestpaAgent],
    ) -> None:
        super().__init__()
        self.westpa_handle = westpa_handle

    async def agent_on_startup(self) -> None:
        """Initialize the agent.

        Override to add custom startup logic (e.g., loading a
        model). Always call ``await super().agent_on_startup()``
        first.

        Logging is configured by the ``Manager`` via its
        ``log_config`` and propagates to every agent it launches,
        including those in remote worker processes.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.info('started')

    @abstractmethod
    def run_simulation(self, metadata: SimMetadata) -> SimResult:
        """Run a simulation for the given metadata.

        Override this method in a subclass to provide custom
        simulation logic. Has access to ``self`` for any
        state initialized in ``__init__`` or
        ``agent_on_startup``.

        Parameters
        ----------
        metadata : SimMetadata
            The simulation metadata (walker weight, restart
            file, parent progress coordinate, etc.).

        Returns
        -------
        SimResult
            The simulation result with data products and
            updated metadata.
        """
        ...

    @action
    async def simulate(self, sim_metadata: SimMetadata) -> None:
        """Run the simulation and send result."""
        self.logger.info(
            f'running sim {sim_metadata.simulation_id} '
            f'iteration {sim_metadata.iteration_id}',
        )

        # Wait for the restart file to be available before running the
        # simulation. Handles NFS caching issues where the file may not be
        # immediately visible
        await wait_for_file(sim_metadata.parent_restart_file, self.logger)

        # Run the simulation in a thread to avoid blocking the event loop
        result = await self.agent_run_sync(self.run_simulation, sim_metadata)

        self.logger.info(f'sim {sim_metadata.simulation_id} complete')
        await self.westpa_handle.receive_simulation_data(result)


class WestpaAgent(Agent, ABC):
    """Base agent for orchestrating the WESTPA iteration cycle.

    Subclass and override ``run_inference`` to provide custom
    resampling logic. The base class handles result collection,
    ensemble state management, checkpointing, and dispatching.

    Simulations are distributed to agents round-robin, so the
    number of walkers can differ from the number of agents
    (e.g., after resampling changes the walker count).

    Parameters
    ----------
    simulation_handles : list[Handle[SimulationAgent]]
        Handles to dispatch simulations to.
    max_iterations : int
        Stop after this many WE iterations.
    ensemble : WeightedEnsemble
        Ensemble state to maintain across iterations.
        Provides ``basis_states``, ``target_states``,
        the initial batch size via ``next_sims``, and the
        starting iteration via ``iteration``.
    checkpointer : EnsembleCheckpointer, optional
        Checkpointer for saving ensemble state.
    """

    max_iterations: int
    simulation_handles: list[Handle[SimulationAgent]]
    ensemble: WeightedEnsemble
    checkpointer: EnsembleCheckpointer | None
    logger: logging.Logger

    _batch: list[SimResult]
    _batch_ready: asyncio.Event

    def __init__(
        self,
        simulation_handles: list[Handle[SimulationAgent]],
        max_iterations: int,
        ensemble: WeightedEnsemble,
        checkpointer: EnsembleCheckpointer | None = None,
    ) -> None:
        super().__init__()
        self.simulation_handles = simulation_handles
        self.max_iterations = max_iterations
        self.ensemble = ensemble
        self.checkpointer = checkpointer

    @property
    def iteration(self) -> int:
        """Return the current iteration from the ensemble."""
        return self.ensemble.iteration

    @property
    def basis_states(self) -> BasisStates:
        """Return basis states from the ensemble."""
        return self.ensemble.basis_states

    @property
    def target_states(self) -> list[TargetState]:
        """Return target states from the ensemble."""
        return self.ensemble.target_states

    async def agent_on_startup(self) -> None:
        """Initialize the agent.

        Override to add custom startup logic. Always call
        ``await super().agent_on_startup()`` first.

        Logging is configured by the ``Manager`` via its
        ``log_config`` and propagates to every agent it launches,
        including those in remote worker processes.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self._batch = []
        self._batch_ready = asyncio.Event()
        self.logger.info(
            f'started (iteration={self.iteration}, max={self.max_iterations})',
        )

    @abstractmethod
    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Run inference/resampling on the batch of results.

        Override this method in a subclass to provide custom
        resampling logic. Has access to ``self`` for any
        state initialized in ``__init__`` or
        ``agent_on_startup``.

        Parameters
        ----------
        sim_results : list[SimResult]
            The simulation results for the current iteration.

        Returns
        -------
        tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]
            A tuple of ``(cur_sims, next_sims, metadata)``.
        """
        ...

    @action
    async def receive_simulation_data(self, result: SimResult) -> None:
        """Buffer a simulation result."""
        self.logger.info(
            f'received sim '
            f'{result.metadata.simulation_id} '
            f'iter {result.metadata.iteration_id}. '
            f'batch: {len(self._batch) + 1}'
            f'/{len(self.ensemble.next_sims)}',
        )
        self._batch.append(result)

        if len(self._batch) >= len(self.ensemble.next_sims):
            self._batch_ready.set()

    @loop
    async def run_westpa(self, shutdown: asyncio.Event) -> None:
        """Run the WESTPA iteration loop."""
        while not shutdown.is_set():
            await self._batch_ready.wait()
            self._batch_ready.clear()

            batch = self._batch
            self._batch = []

            self.logger.info(
                f'running inference on {len(batch)} '
                f'results for iteration {self.iteration}',
            )

            # Run the user's inference/resampling
            cur_sims, next_sims, metadata = await self.agent_run_sync(
                self.run_inference,
                batch,
            )

            # Update ensemble state and checkpoint
            self.ensemble.advance_iteration(
                cur_sims=cur_sims,
                next_sims=next_sims,
                metadata=metadata,
            )
            if self.checkpointer is not None:
                self.checkpointer.save(self.ensemble)

            self.logger.info(
                f'iteration {self.iteration} complete. '
                f'{len(next_sims)} walkers next.',
            )

            if self.iteration >= self.max_iterations:
                self.logger.info(
                    f'reached max iterations '
                    f'({self.max_iterations}), '
                    f'shutting down.',
                )
                shutdown.set()
                return

            # Dispatch next iteration round-robin
            self.logger.info(f'dispatching iteration {self.iteration}')
            await dispatch_round_robin(self.simulation_handles, next_sims)


async def run_westpa_workflow(  # noqa: PLR0913
    manager: Manager[Any],
    sim_agent_type: type[SimulationAgent],
    westpa_agent_type: type[WestpaAgent],
    max_iterations: int,
    ensemble: WeightedEnsemble,
    num_sim_agents: int,
    checkpointer: EnsembleCheckpointer | None = None,
    sim_agent_kwargs: dict[str, Any] | None = None,
    westpa_agent_kwargs: dict[str, Any] | None = None,
    sim_executor: str | None = None,
    westpa_executor: str | None = None,
) -> None:
    """Run a WESTPA workflow with user-defined agent types.

    Registers and launches all agents, dispatches the first
    iteration of simulations from ``ensemble.next_sims``,
    and waits for the workflow to complete. Simulations are
    distributed round-robin across ``num_sim_agents`` agents,
    so each agent may handle multiple simulations sequentially.

    Parameters
    ----------
    manager : Manager
        The Academy manager (within ``async with`` context).
        Configure logging by passing a ``log_config`` (e.g.
        ``recommended_logging(...)``) when creating the manager;
        it propagates to every agent, including remote workers.
    sim_agent_type : type[SimulationAgent]
        Concrete ``SimulationAgent`` subclass that implements
        ``run_simulation``.
    westpa_agent_type : type[WestpaAgent]
        Concrete ``WestpaAgent`` subclass that implements
        ``run_inference``.
    max_iterations : int
        Maximum number of WE iterations.
    ensemble : WeightedEnsemble
        Ensemble state to track across iterations.
        ``ensemble.next_sims`` provides the initial batch.
    checkpointer : EnsembleCheckpointer, optional
        Save ensemble state each iteration.
    sim_agent_kwargs : dict, optional
        Extra keyword arguments for ``SimulationAgent``
        subclass ``__init__`` (e.g., simulation config).
    westpa_agent_kwargs : dict, optional
        Extra keyword arguments for ``WestpaAgent``
        subclass ``__init__`` (e.g., inference config).
    sim_executor : str, optional
        Named executor for simulation agents (e.g., GPU).
    westpa_executor : str, optional
        Named executor for the WESTPA agent (e.g., CPU).
    num_sim_agents : int
        Number of simulation agents to launch. Must not exceed the
        number of available executor slots (e.g., GPUs); set it to
        the slot count so agents are reused across simulations
        (round-robin) rather than queued indefinitely. Launching one
        agent per walker deadlocks whenever walkers exceed slots
        (e.g., 4 GPUs, 72 walkers), so this is required rather than
        defaulted.

    Raises
    ------
    ValueError
        If ``num_sim_agents`` is less than 1.
    """
    if num_sim_agents < 1:
        raise ValueError(
            f'num_sim_agents must be >= 1, got {num_sim_agents}.',
        )

    initial_sims = ensemble.next_sims

    # Register agents with the manager
    reg_westpa = await manager.register_agent(westpa_agent_type)
    reg_sims = await asyncio.gather(
        *[
            manager.register_agent(sim_agent_type)
            for _ in range(num_sim_agents)
        ],
    )

    # Get handles for inter-agent communication
    westpa_handle = manager.get_handle(reg_westpa)
    sim_handles = [manager.get_handle(reg) for reg in reg_sims]

    # Launch the WestpaAgent
    westpa_handle = await manager.launch(
        westpa_agent_type,
        registration=reg_westpa,
        args=(sim_handles,),
        kwargs={
            'max_iterations': max_iterations,
            'ensemble': ensemble,
            'checkpointer': checkpointer,
            **(westpa_agent_kwargs or {}),
        },
        executor=westpa_executor,
    )

    # Launch the SimulationAgents
    sim_agents = await asyncio.gather(
        *[
            manager.launch(
                sim_agent_type,
                registration=reg,
                args=(westpa_handle,),
                kwargs=sim_agent_kwargs,
                executor=sim_executor,
            )
            for reg in reg_sims
        ],
    )

    # Dispatch first iteration round-robin
    await dispatch_round_robin(sim_agents, initial_sims)

    # Wait for the WestpaAgent to finish
    await manager.wait((westpa_handle,))
