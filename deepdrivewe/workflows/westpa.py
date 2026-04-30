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
        sim_agent_kwargs={'model': my_model},
        westpa_agent_kwargs={'basis': my_basis},
    )
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import aiohttp
import aiohttp.client
from academy.agent import action
from academy.agent import Agent
from academy.agent import loop
from academy.handle import Handle
from academy.logging import init_logging
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
) -> None:
    """Dispatch simulations to agents round-robin."""
    await asyncio.gather(
        *[
            handles[i % len(handles)].simulate(sim)
            for i, sim in enumerate(sims)
        ],
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
        logfile: Path | None = None,
    ) -> None:
        super().__init__()
        # Patch the aiohttp default session timeout so the SSE listener
        # survives long-running iterations.  Academy creates the exchange
        # ClientSession (step 1 of run_until_complete) AFTER __init__, so
        # setting DEFAULT_TIMEOUT here is the earliest reliable hook.
        # This is especially important for sim agents running inside Parsl
        # worker subprocesses: those processes never execute main.py, so
        # without this patch they inherit aiohttp's default total=300 s and
        # the SSE listener times out after 5 minutes, causing agents to stop
        # receiving new simulation tasks.
        aiohttp.client.DEFAULT_TIMEOUT = aiohttp.ClientTimeout(
            total=None,
            sock_connect=30,
            sock_read=None,
        )
        self.westpa_handle = westpa_handle
        self.logfile = logfile

    async def agent_on_startup(self) -> None:
        """Initialize the agent.

        Override to add custom startup logic (e.g., loading a
        model). Always call ``await super().agent_on_startup()``
        first.
        """
        if self.logfile is not None:
            init_logging('INFO', logfile=self.logfile)
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
        logfile: Path | None = None,
    ) -> None:
        super().__init__()
        self.simulation_handles = simulation_handles
        self.max_iterations = max_iterations
        self.ensemble = ensemble
        self.checkpointer = checkpointer
        self.logfile = logfile

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
        """
        if self.logfile is not None:
            init_logging('INFO', logfile=self.logfile)
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
    manager: Manager,
    sim_agent_type: type[SimulationAgent],
    westpa_agent_type: type[WestpaAgent],
    max_iterations: int,
    ensemble: WeightedEnsemble,
    checkpointer: EnsembleCheckpointer | None = None,
    sim_agent_kwargs: dict[str, Any] | None = None,
    westpa_agent_kwargs: dict[str, Any] | None = None,
    sim_executor: str | None = None,
    westpa_executor: str | None = None,
    logfile: Path | None = None,
) -> None:
    """Run a WESTPA workflow with user-defined agent types.

    Registers and launches all agents, dispatches the first
    iteration of simulations from ``ensemble.next_sims``,
    and waits for the workflow to complete. One
    ``SimulationAgent`` is launched per initial simulation;
    simulations are distributed round-robin.

    Parameters
    ----------
    manager : Manager
        The Academy manager (within ``async with`` context).
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
    logfile : Path, optional
        Log file path passed to each agent. Agents call
        ``init_logging`` in ``agent_on_startup`` so that
        workers in separate processes get logging configured.
    """
    initial_sims = ensemble.next_sims
    # TODO: Generalize this so we don't have to assume one agent per sim.
    # This is the case were we reuse the same hardware for multiple sims.
    num_agents = len(initial_sims)

    # Register agents with the manager
    reg_westpa = await manager.register_agent(westpa_agent_type)
    reg_sims = await asyncio.gather(
        *[manager.register_agent(sim_agent_type) for _ in range(num_agents)],
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
            'logfile': logfile,
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
                kwargs={
                    'logfile': logfile,
                    **(sim_agent_kwargs or {}),
                },
                executor=sim_executor,
            )
            for reg in reg_sims
        ],
    )

    # Dispatch first iteration round-robin
    await dispatch_round_robin(sim_agents, initial_sims)

    # Wait for the WestpaAgent to finish
    await manager.wait((westpa_handle,))
