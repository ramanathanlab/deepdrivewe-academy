r"""OpenMM NTL9 Huber-Kim WESTPA example using Academy agents.

Adapted from examples/openmm_ntl9_hk to use the Academy
multi-agent framework instead of Colmena/Parsl.

Usage
-----
::

    python -m examples.openmm_ntl9_hk_academy.main -c config.yaml

Or with the Academy Exchange Cloud::

    python -m examples.openmm_ntl9_hk_academy.main \
        -c config.yaml --exchange globus
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager
from parsl.concurrent import ParslPoolExecutor
from workflow import ExperimentSettings
from workflow import HuberKimWestpaAgent
from workflow import OpenMMSimAgent

from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.workflows.westpa import run_westpa_workflow

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
        default='local',
    )
    return parser.parse_args()


def _export_pythonpath() -> None:
    """Add this directory to PYTHONPATH for Parsl workers."""
    example_dir = str(Path(__file__).resolve().parent)
    pythonpath = os.environ.get('PYTHONPATH', '')
    if example_dir not in pythonpath:
        os.environ['PYTHONPATH'] = example_dir + os.pathsep + pythonpath


async def main() -> None:
    """Run the OpenMM WESTPA workflow."""
    _export_pythonpath()
    args = parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    init_logging('INFO', logfile=cfg.output_dir / 'runtime.log')

    # Create Parsl configuration from compute config
    parsl_config = cfg.compute_config.get_parsl_config(
        cfg.output_dir / 'run-info',
    )

    # Initialize or resume ensemble
    checkpointer = EnsembleCheckpointer(output_dir=cfg.output_dir)
    checkpoint = checkpointer.latest_checkpoint()

    if checkpoint is None:
        ensemble = WeightedEnsemble(
            basis_states=cfg.basis_states,
            target_states=cfg.target_states,
        )
        ensemble.initialize_basis_states(cfg.basis_state_initializer)
    else:
        ensemble = checkpointer.load(checkpoint)
        logging.info(f'Loaded ensemble from checkpoint {checkpoint}')

    logging.info(f'Basis states: {ensemble.basis_states}')
    logging.info(f'Target states: {ensemble.target_states}')

    # Create the Parsl executor outside the Manager context so we
    # can guarantee cleanup even if the process is interrupted.
    gpu_executor = ParslPoolExecutor(parsl_config)

    # Safety net: atexit fires on normal exit and unhandled exceptions.
    atexit.register(gpu_executor.shutdown, wait=False)

    try:
        async with await Manager.from_exchange_factory(
            factory=create_exchange_factory(args.exchange),
            executors={
                'gpu': gpu_executor,
                'cpu': ThreadPoolExecutor(max_workers=1),
            },
            default_executor='gpu',
        ) as manager:
            await run_westpa_workflow(
                manager=manager,
                sim_agent_type=OpenMMSimAgent,
                westpa_agent_type=HuberKimWestpaAgent,
                max_iterations=cfg.num_iterations,
                ensemble=ensemble,
                checkpointer=checkpointer,
                sim_agent_kwargs={
                    'sim_config': cfg.simulation_config,
                    'output_dir': cfg.output_dir / 'simulation',
                },
                westpa_agent_kwargs={
                    'inference_config': cfg.inference_config,
                },
                sim_executor='gpu',
                westpa_executor='cpu',
                logfile=cfg.output_dir / 'runtime.log',
            )
    finally:
        gpu_executor.shutdown(wait=False)


if __name__ == '__main__':
    asyncio.run(main())
