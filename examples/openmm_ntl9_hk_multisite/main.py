r"""OpenMM NTL9 Huber-Kim multisite example using Academy agents.

Runs the WE orchestrator locally and dispatches OpenMM simulation
agents to a remote HPC site via a pre-configured Globus Compute
endpoint, communicating through the Academy Exchange Cloud.

Usage
-----
::

    # Two-site run (orchestrator local, sims on HPC endpoint)
    python main.py -c config.yaml --exchange globus

    # Single-host smoke test (sims in a local thread pool)
    python main.py -c config.yaml --exchange local
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from argparse import ArgumentParser
from collections.abc import MutableMapping
from concurrent.futures import Executor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from workflow import ExperimentSettings
from workflow import HuberKimWestpaAgent
from workflow import OpenMMSimAgent

from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.workflows.westpa import run_westpa_workflow

# --- Configuration ---------------------------------------------------


EXCHANGE_ADDRESS = 'https://exchange.academy-agents.org'


def create_exchange_factory(
    exchange_type: str,
) -> LocalExchangeFactory | HttpExchangeFactory:
    """Create the exchange factory."""
    if exchange_type == 'local':
        return LocalExchangeFactory()
    return HttpExchangeFactory(url=EXCHANGE_ADDRESS, auth_method='globus')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument(
        '--exchange',
        choices=['local', 'globus'],
        default='globus',
        help='"globus" dispatches sims to the Globus Compute endpoint '
        'via the Academy Exchange Cloud; "local" runs everything '
        'in-process for smoke testing.',
    )
    return parser.parse_args()


def create_executors(
    exchange: str,
    simulation_endpoint_id: str,
    inference_endpoint_id: str | None,
) -> MutableMapping[str, Executor | None]:
    """Return the ``{executor_name: Executor}`` map for the Manager.

    ``local`` uses ThreadPools on the orchestrator host for smoke
    tests. ``globus`` dispatches each agent type to its own
    pre-configured Globus Compute endpoint via
    ``globus_compute_sdk.Executor``:

    - ``sim_executor`` -> the simulation endpoint (OpenMMSimAgent).
    - ``westpa_executor`` -> the inference endpoint
      (HuberKimWestpaAgent). If ``inference_endpoint_id`` is ``None``
      the WESTPA agent runs locally on the orchestrator via a
      ThreadPoolExecutor instead.
    """
    if exchange == 'local':
        return {
            'sim_executor': ThreadPoolExecutor(max_workers=4),
            'westpa_executor': ThreadPoolExecutor(max_workers=1),
        }
    # Imported lazily so the local smoke test does not require
    # globus-compute-sdk to be installed.
    from globus_compute_sdk import Executor as GCExecutor

    sim_executor = GCExecutor(endpoint_id=simulation_endpoint_id)

    if inference_endpoint_id is None:
        logging.warning(
            'No inference endpoint ID provided; WESTPA agent will run locally',
        )
        westpa_executor: Executor = ThreadPoolExecutor(max_workers=1)
    else:
        westpa_executor = GCExecutor(endpoint_id=inference_endpoint_id)

    return {'sim_executor': sim_executor, 'westpa_executor': westpa_executor}


def _export_pythonpath() -> None:
    """Add this directory to PYTHONPATH for Parsl workers."""
    example_dir = str(Path(__file__).resolve().parent)
    pythonpath = os.environ.get('PYTHONPATH', '')
    if example_dir not in pythonpath:
        os.environ['PYTHONPATH'] = example_dir + os.pathsep + pythonpath


async def main() -> None:
    """Run the OpenMM WESTPA workflow across two sites."""
    _export_pythonpath()
    args = parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    init_logging('INFO', logfile=cfg.output_dir / 'runtime.log')

    # Seed ensemble for the orchestrator. This determines how many
    # sim agents to launch (one per initial walker). Checkpoint
    # resume happens on the inference host in agent_on_startup.
    ensemble = WeightedEnsemble(
        basis_states=cfg.basis_states,
        target_states=cfg.target_states,
    )
    ensemble.initialize_basis_states(cfg.basis_state_initializer)

    logging.info(f'Basis states: {ensemble.basis_states}')
    logging.info(f'Target states: {ensemble.target_states}')

    # Create the compute executors for each agent type
    executors = create_executors(
        args.exchange,
        cfg.globus_compute.simulation_endpoint_id,
        cfg.globus_compute.inference_endpoint_id,
    )

    try:
        async with await Manager.from_exchange_factory(
            factory=create_exchange_factory(args.exchange),
            executors=executors,
            default_executor='sim_executor',
        ) as manager:
            await run_westpa_workflow(
                manager=manager,
                sim_agent_type=OpenMMSimAgent,
                westpa_agent_type=HuberKimWestpaAgent,
                max_iterations=cfg.num_iterations,
                ensemble=ensemble,
                checkpointer=None,
                sim_agent_kwargs={
                    'sim_config': cfg.simulation_config,
                    'output_dir': cfg.sim_output_dir,
                },
                westpa_agent_kwargs={
                    'inference_config': cfg.inference_config,
                },
                sim_executor='sim_executor',
                westpa_executor='westpa_executor',
                logfile=cfg.output_dir / 'runtime.log',
            )
    finally:
        for ex in executors.values():
            if ex is not None:
                ex.shutdown(wait=False)


if __name__ == '__main__':
    asyncio.run(main())
