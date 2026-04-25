"""Tests for `deepdrivewe.workflows.westpa`.

The WESTPA workflow is the orchestration glue that wires together the
simulation agents and the resampling agent through Academy's
inter-agent messaging. These tests exercise the full loop end-to-end
using `LocalExchangeFactory` + `ThreadPoolExecutor`, with mock agent
subclasses modeled on `examples/minimal_westpa/main.py`.

The mock `SimulationAgent` deliberately creates a real restart file on
disk for each completed sim so the next iteration's `wait_for_file`
returns immediately without burning through the (8-retry, 1s base
delay) backoff schedule.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager

from deepdrivewe.api import BasisStates
from deepdrivewe.api import IterationMetadata
from deepdrivewe.api import SimMetadata
from deepdrivewe.api import SimResult
from deepdrivewe.api import TargetState
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.workflows.westpa import dispatch_round_robin
from deepdrivewe.workflows.westpa import run_westpa_workflow
from deepdrivewe.workflows.westpa import SimulationAgent
from deepdrivewe.workflows.westpa import WestpaAgent


class _MockSimAgent(SimulationAgent):
    """Mock simulation agent that materializes a restart file per call.

    The base ``SimulationAgent.simulate`` action calls ``wait_for_file``
    on ``parent_restart_file`` before invoking ``run_simulation``. By
    creating the next iteration's restart file inside ``run_simulation``
    we guarantee subsequent iterations don't stall waiting for a missing
    file (which would burn ~4 minutes of retry backoff).
    """

    def __init__(
        self,
        westpa_handle: Any,
        restart_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(westpa_handle, **kwargs)
        self.restart_dir = Path(restart_dir)

    def run_simulation(self, metadata: SimMetadata) -> SimResult:
        new_restart = (
            self.restart_dir
            / f'iter_{metadata.iteration_id}_sim_{metadata.simulation_id}.rst'
        )
        new_restart.touch()
        metadata.pcoord = [[1.0]]
        metadata.restart_file = new_restart
        return SimResult(data={'pcoord': np.array([[1.0]])}, metadata=metadata)


class _MockWestpaAgent(WestpaAgent):
    """Mock resampler that simply propagates each walker forward."""

    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[
        list[SimMetadata],
        list[SimMetadata],
        IterationMetadata,
    ]:
        cur_sims = [r.metadata for r in sim_results]
        next_sims: list[SimMetadata] = []
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
        metadata = IterationMetadata(iteration_id=cur_sims[0].iteration_id)
        return cur_sims, next_sims, metadata


def _build_ensemble(restart_dir: Path, num_sims: int = 2) -> WeightedEnsemble:
    """Build a tiny ensemble with on-disk seed restart files."""
    initial: list[SimMetadata] = []
    for idx in range(num_sims):
        # Seed the parent_restart_file the first iteration's wait_for_file
        # will look for.
        seed = restart_dir / f'initial_{idx}.rst'
        seed.touch()
        initial.append(
            SimMetadata(
                weight=1.0 / num_sims,
                simulation_id=idx,
                iteration_id=1,
                parent_restart_file=seed,
                parent_pcoord=[0.0],
            ),
        )
    return WeightedEnsemble(
        basis_states=BasisStates(
            basis_state_dir=restart_dir,
            initial_ensemble_members=num_sims,
        ),
        target_states=[TargetState(pcoord=[0.0])],
        next_sims=initial,
    )


class _RecordingHandle:
    """Stand-in for `Handle[SimulationAgent]` that records dispatch calls."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def simulate(self, sim: SimMetadata) -> None:
        self.calls.append(sim.simulation_id)


@pytest.mark.integration
class TestDispatchRoundRobin:
    """Covers the round-robin dispatcher."""

    async def test_distributes_sims_evenly_across_handles(self) -> None:
        handles = [_RecordingHandle() for _ in range(3)]
        sims = [
            SimMetadata(
                weight=0.1,
                simulation_id=i,
                iteration_id=1,
                parent_restart_file=Path(f'/tmp/dummy_{i}'),
                parent_pcoord=[0.0],
            )
            for i in range(7)
        ]
        await dispatch_round_robin(handles, sims)
        # 7 sims across 3 handles -> [3, 2, 2]
        assert handles[0].calls == [0, 3, 6]
        assert handles[1].calls == [1, 4]
        assert handles[2].calls == [2, 5]

    async def test_more_handles_than_sims_only_uses_first(self) -> None:
        handles = [_RecordingHandle() for _ in range(5)]
        sims = [
            SimMetadata(
                weight=0.5,
                simulation_id=i,
                iteration_id=1,
                parent_restart_file=Path(f'/tmp/dummy_{i}'),
                parent_pcoord=[0.0],
            )
            for i in range(2)
        ]
        await dispatch_round_robin(handles, sims)
        assert handles[0].calls == [0]
        assert handles[1].calls == [1]
        for h in handles[2:]:
            assert h.calls == []


@pytest.mark.integration
class TestRunWestpaWorkflow:
    """End-to-end smoke test exercising SimulationAgent + WestpaAgent."""

    async def test_runs_to_max_iterations(self, tmp_path: Path) -> None:
        ensemble = _build_ensemble(tmp_path, num_sims=2)
        async with await Manager.from_exchange_factory(
            factory=LocalExchangeFactory(),
            executors=ThreadPoolExecutor(),
        ) as manager:
            # asyncio.wait_for guards against a hang if anything in the
            # workflow wedges (e.g. wait_for_file backoff on a missing file).
            await asyncio.wait_for(
                run_westpa_workflow(
                    manager=manager,
                    sim_agent_type=_MockSimAgent,
                    westpa_agent_type=_MockWestpaAgent,
                    max_iterations=2,
                    ensemble=ensemble,
                    sim_agent_kwargs={'restart_dir': tmp_path},
                ),
                timeout=30.0,
            )

        # IterationMetadata.iteration_id is set to cur_sims[0].iteration_id
        # inside run_inference, so ensemble.iteration tracks the iteration
        # that *just completed*. After max_iterations=2 it sits at 2.
        assert ensemble.iteration == 2
        assert len(ensemble.cur_sims) == 2
        for sim in ensemble.cur_sims:
            assert sim.iteration_id == 2
            assert sim.restart_file is not None
            assert sim.restart_file.exists()
        # next_sims point at iteration 3 since the loop dispatches before
        # the shutdown check would reset things — we don't dispatch them
        # but they're built by run_inference.
        assert len(ensemble.next_sims) == 2
        for sim in ensemble.next_sims:
            assert sim.iteration_id == 3

    async def test_single_iteration_workflow(self, tmp_path: Path) -> None:
        ensemble = _build_ensemble(tmp_path, num_sims=2)
        async with await Manager.from_exchange_factory(
            factory=LocalExchangeFactory(),
            executors=ThreadPoolExecutor(),
        ) as manager:
            await asyncio.wait_for(
                run_westpa_workflow(
                    manager=manager,
                    sim_agent_type=_MockSimAgent,
                    westpa_agent_type=_MockWestpaAgent,
                    max_iterations=1,
                    ensemble=ensemble,
                    sim_agent_kwargs={'restart_dir': tmp_path},
                ),
                timeout=30.0,
            )
        assert ensemble.iteration == 1
        assert len(ensemble.cur_sims) == 2
