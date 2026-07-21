from __future__ import annotations

import asyncio
import logging
import pickle

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

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
from deepdrivewe.api import TrainResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.utils import wait_for_file

from deepdrivewe.workflows.westpa import SimulationAgent, WestpaAgent, dispatch_round_robin, run_westpa_workflow


class TrainingAgent(Agent, ABC):
    """Base agent for running DDWE training.

    Subclass and override ``run_training`` to provide custom
    training logic. Use ``agent_on_startup`` to initialize
    expensive state (e.g., a model held across iterations).

    The ``train`` action offloads ``run_training`` to a thread
    pool via ``agent_run_sync`` since training is typically
    blocking, and returns the result directly to the caller.

    Parameters
    ----------
    logfile : Path, optional
        Log file path. ``agent_on_startup`` calls
        ``init_logging`` so workers in separate processes get
        logging configured.
    """

    logger: logging.Logger

    def __init__(self, logfile: Path | None = None) -> None:
        super().__init__()
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
    def run_training(
        self,
        sim_results: list[SimResult],
    ) -> TrainResult:
        """Run training on the batch of results.

        Override this method in a subclass to provide custom
        training logic. Has access to ``self`` for any state
        initialized in ``__init__`` or ``agent_on_startup``.

        Parameters
        ----------
        sim_results : list[SimResult]
            The simulation results for the current iteration.

        Returns
        -------
        TrainResult
            The output of the training process.
        """
        ...

    @action
    async def train(self, batch_path: Path) -> TrainResult:
        """Load the batch from shared storage and run training.

        The batch is passed by path rather than by value: a full
        batch of simulation data exceeds the message size limit of
        the exchange transport.
        """
        await wait_for_file(batch_path, self.logger)
        sim_results = pickle.loads(batch_path.read_bytes())

        self.logger.info(
            f'running training on {len(sim_results)} results',
        )
        result = await self.agent_run_sync(
            self.run_training,
            sim_results,
        )
        self.logger.info('training complete')
        return result


class InferenceAgent(Agent, ABC):
    """Base agent for running DDWE inference/resampling.

    Subclass and override ``run_inference`` to provide custom
    resampling logic. Use ``agent_on_startup`` to initialize
    expensive state (e.g., a model held across iterations).

    The ``infer`` action offloads ``run_inference`` to a thread
    pool via ``agent_run_sync`` since inference is typically
    blocking, and returns the result directly to the caller.

    Parameters
    ----------
    logfile : Path, optional
        Log file path. ``agent_on_startup`` calls
        ``init_logging`` so workers in separate processes get
        logging configured.
    """

    logger: logging.Logger

    def __init__(self, logfile: Path | None = None) -> None:
        super().__init__()
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
    def run_inference(
        self,
        sim_results: list[SimResult],
        train_output: TrainResult,
        basis_states: BasisStates,
        target_states: list[TargetState],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Run inference/resampling on the batch of results.

        Override this method in a subclass to provide custom
        resampling logic. Has access to ``self`` for any state
        initialized in ``__init__`` or ``agent_on_startup``.

        Parameters
        ----------
        sim_results : list[SimResult]
            The simulation results for the current iteration.
        train_output : TrainResult
            The output of the training process.
        basis_states : BasisStates
            The basis states for the ensemble.
        target_states : list[TargetState]
            The target states for recycling.

        Returns
        -------
        tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]
            A tuple of ``(cur_sims, next_sims, metadata)``.
        """
        ...

    @action
    async def infer(
        self,
        batch_path: Path,
        train_output: TrainResult,
        basis_states: BasisStates,
        target_states: list[TargetState],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Load the batch from shared storage and run inference.

        The batch is passed by path rather than by value: a full
        batch of simulation data exceeds the message size limit of
        the exchange transport.
        """
        await wait_for_file(batch_path, self.logger)
        sim_results = pickle.loads(batch_path.read_bytes())

        self.logger.info(
            f'running inference on {len(sim_results)} results',
        )
        result = await self.agent_run_sync(
            self.run_inference,
            sim_results,
            train_output,
            basis_states,
            target_states,
        )
        self.logger.info('inference complete')
        return result


class DDWEAgent(WestpaAgent, ABC):
    """Orchestrator agent for the DDWE iteration cycle.

    Delegates training and inference to separate
    ``TrainingAgent``/``InferenceAgent`` instances via handles,
    instead of running them in-process.
    """

    use_stale_model: bool
    _train_output: TrainResult | None
    _pending_train: asyncio.Task[TrainResult] | None

    def __init__(
            self,
            simulation_handles: list[Handle[SimulationAgent]],
            training_handle: Handle[TrainingAgent],
            inference_handle: Handle[InferenceAgent],
            max_iterations: int,
            ensemble: WeightedEnsemble,
            checkpointer: EnsembleCheckpointer | None = None,
            logfile: Path | None = None,
            use_stale_model: bool = False,
            output_dir: Path = Path('ddwe_output'),
    ) -> None:
        super().__init__(simulation_handles, max_iterations, ensemble, checkpointer, logfile)
        self.training_handle = training_handle
        self.inference_handle = inference_handle
        self.use_stale_model = use_stale_model
        self.output_dir = output_dir
        self._train_output = None
        self._pending_train = None

    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        """Unused; required by ``WestpaAgent``'s abstract interface.

        Inference is delegated to ``inference_handle`` instead;
        see ``run`` below.
        """
        raise NotImplementedError(
            'DDWEAgent delegates inference to InferenceAgent via '
            'inference_handle; this method is not used.',
        )

    async def update_ensemble(
        self,
        cur_sims: list[SimMetadata],
        next_sims: list[SimMetadata],
        metadata: IterationMetadata,
        shutdown: asyncio.Event,
    ) -> None:
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

    @loop
    async def run(self, shutdown: asyncio.Event) -> None:
        """Run the DDWE iteration loop."""
        while not shutdown.is_set():
            await self._batch_ready.wait()
            self._batch_ready.clear()

            batch = self._batch
            self._batch = []

            # Get correct iteration number
            batch_iteration = batch[0].metadata.iteration_id

            # Spill the batch to shared storage and pass it to the
            # training and inference agents by path. The batch is too
            # large to send through the exchange as a message, and
            # routing bulk data through the message bus would
            # bottleneck at scale.
            batch_path = (
                self.output_dir / f'{batch_iteration:06d}' / 'batch.pkl'
            )
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            batch_path.write_bytes(
                pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL),
            )

            # Harvest the training launched on a previous iteration.
            # It runs in the background while simulations execute, so
            # by now it has usually already finished.
            if self._pending_train is not None:
                self._train_output = await self._pending_train
                self._pending_train = None

            # Stale model: launch training in the background and run
            # inference immediately against the previous iteration's
            # model, so training overlaps the next round of
            # simulations rather than blocking them.
            if self.use_stale_model and self._train_output is not None:
                self.logger.info(
                    'using stale model, training in the background',
                )

                self._pending_train = asyncio.create_task(
                    self.training_handle.train(batch_path),
                )

                cur_sims, next_sims, metadata = (
                    await self.inference_handle.infer(
                        batch_path,
                        self._train_output,
                        self.basis_states,
                        self.target_states,
                    )
                )
            else:  # Sequential: train first, then infer
                self.logger.info(
                    f'running training on {len(batch)} '
                    f'results for iteration {batch_iteration}',
                )

                # Run training on the training agent
                self._train_output = await self.training_handle.train(
                    batch_path,
                )

                self.logger.info(
                    f'running inference on {len(batch)} '
                    f'results for iteration {batch_iteration}',
                )

                # Run inference/resampling on the inference agent
                cur_sims, next_sims, metadata = (
                    await self.inference_handle.infer(
                        batch_path,
                        self._train_output,
                        self.basis_states,
                        self.target_states,
                    )
                )

            # Update ensemble state and checkpoint
            await self.update_ensemble(cur_sims, next_sims, metadata, shutdown)


async def run_ddwe_workflow(  # noqa: PLR0913
    manager: Manager,
    sim_agent_type: type[SimulationAgent],
    training_agent_type: type[TrainingAgent],
    inference_agent_type: type[InferenceAgent],
    ddwe_agent_type: type[DDWEAgent],
    max_iterations: int,
    ensemble: WeightedEnsemble,
    checkpointer: EnsembleCheckpointer | None = None,
    sim_agent_kwargs: dict[str, Any] | None = None,
    training_agent_kwargs: dict[str, Any] | None = None,
    inference_agent_kwargs: dict[str, Any] | None = None,
    ddwe_agent_kwargs: dict[str, Any] | None = None,
    sim_executor: str | None = None,
    training_executor: str | None = None,
    inference_executor: str | None = None,
    ddwe_executor: str | None = None,
    logfile: Path | None = None,
    num_sim_agents: int | None = None,
) -> None:
    """Run a DDWE workflow with user-defined agent types.

    Registers and launches the training and inference agents,
    then delegates simulation and orchestrator setup to
    ``run_westpa_workflow``, threading the training/inference
    handles through its ``westpa_agent_kwargs``.

    Parameters
    ----------
    manager : Manager
        The Academy manager (within ``async with`` context).
    sim_agent_type : type[SimulationAgent]
        Concrete ``SimulationAgent`` subclass.
    training_agent_type : type[TrainingAgent]
        Concrete ``TrainingAgent`` subclass.
    inference_agent_type : type[InferenceAgent]
        Concrete ``InferenceAgent`` subclass.
    ddwe_agent_type : type[DDWEAgent]
        Concrete ``DDWEAgent`` subclass.
    max_iterations : int
        Maximum number of WE iterations.
    ensemble : WeightedEnsemble
        Ensemble state to track across iterations.
    checkpointer : EnsembleCheckpointer, optional
        Save ensemble state each iteration.
    sim_agent_kwargs : dict, optional
        Extra keyword arguments for the simulation agent.
    training_agent_kwargs : dict, optional
        Extra keyword arguments for the training agent.
    inference_agent_kwargs : dict, optional
        Extra keyword arguments for the inference agent.
    ddwe_agent_kwargs : dict, optional
        Extra keyword arguments for the DDWE orchestrator agent.
    sim_executor : str, optional
        Named executor for simulation agents.
    training_executor : str, optional
        Named executor for the training agent.
    inference_executor : str, optional
        Named executor for the inference agent.
    ddwe_executor : str, optional
        Named executor for the DDWE orchestrator agent.
    logfile : Path, optional
        Log file path passed to each agent.
    num_sim_agents : int, optional
        Number of simulation agents to launch.
    """
    # Register training/inference agents and get their handles
    # up front, mirroring how sim_handles is built in
    # run_westpa_workflow before those agents are launched.
    reg_training = await manager.register_agent(training_agent_type)
    reg_inference = await manager.register_agent(inference_agent_type)
    training_handle = manager.get_handle(reg_training)
    inference_handle = manager.get_handle(reg_inference)

    # Launch the training/inference agents.
    await asyncio.gather(
        manager.launch(
            training_agent_type,
            registration=reg_training,
            kwargs={'logfile': logfile, **(training_agent_kwargs or {})},
            executor=training_executor,
        ),
        manager.launch(
            inference_agent_type,
            registration=reg_inference,
            kwargs={'logfile': logfile, **(inference_agent_kwargs or {})},
            executor=inference_executor,
        ),
    )

    # Delegate sim + orchestrator setup to the existing,
    # unmodified run_westpa_workflow, threading the handles
    # through its westpa_agent_kwargs passthrough.
    await run_westpa_workflow(
        manager=manager,
        sim_agent_type=sim_agent_type,
        westpa_agent_type=ddwe_agent_type,
        max_iterations=max_iterations,
        ensemble=ensemble,
        checkpointer=checkpointer,
        sim_agent_kwargs=sim_agent_kwargs,
        westpa_agent_kwargs={
            'training_handle': training_handle,
            'inference_handle': inference_handle,
            **(ddwe_agent_kwargs or {}),
        },
        sim_executor=sim_executor,
        westpa_executor=ddwe_executor,
        logfile=logfile,
        num_sim_agents=num_sim_agents,
    )
