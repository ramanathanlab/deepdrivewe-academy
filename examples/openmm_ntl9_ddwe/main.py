r"""OpenMM NTL9 CVAE-based DeepDriveWE example using Academy agents.

Adapted from examples/openmm_ntl9_hk to use the Academy
multi-agent framework instead of Colmena/Parsl.

Usage
-----
::

    python -m examples.openmm_ntl9_ddwe.main -c config.yaml

Or with the Academy Exchange Cloud::

    python -m examples.openmm_ntl9_ddwe.main \
        -c config.yaml --exchange globus
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import parsl
from academy.exchange import ExchangeFactory
from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.local import LocalExchangeFactory
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from parsl.concurrent import ParslPoolExecutor
from parsl.config import Config
from workflow import CVAEInferAgent
from workflow import CVAETrainAgent
from workflow import ExperimentSettings

from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.checkpoint import EnsembleCheckpointer
from deepdrivewe.workflows.ddwe import DDWEAgent
from deepdrivewe.workflows.ddwe import run_ddwe_workflow
from examples.openmm_ntl9_hk.workflow import OpenMMSimAgent


def create_exchange_factory(
    exchange_type: str,
) -> ExchangeFactory[Any]:
    """Create the exchange factory."""
    if exchange_type == 'local':
        return LocalExchangeFactory()
    return HttpExchangeFactory(auth_method='globus')


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
    """Run the OpenMM DeepDriveWE workflow."""
    _export_pythonpath()
    args = parse_args()
    cfg = ExperimentSettings.from_yaml(args.config)
    cfg.dump_yaml(cfg.output_dir / 'params.yaml')

    # Build one Parsl Config with three labeled executors (sim,
    # train, infer). parsl.load is a per-process singleton, so a
    # single DFK must back all GPU pools. Train and infer each get
    # one dedicated GPU carved from ddwe_compute_config.
    accelerators = cfg.ddwe_compute_config.available_accelerators
    train_cfg = cfg.ddwe_compute_config.model_copy(
        update={
            'available_accelerators': [accelerators[0]],
            'label': 'train_htex',
        },
    )
    infer_cfg = cfg.ddwe_compute_config.model_copy(
        update={
            'available_accelerators': [accelerators[1]],
            'label': 'infer_htex',
        },
    )
    run_dir = cfg.output_dir / 'run-info'
    combined_config = Config(
        run_dir=str(run_dir),
        executors=[
            cfg.sim_compute_config.get_parsl_config(run_dir).executors[0],
            train_cfg.get_parsl_config(run_dir).executors[0],
            infer_cfg.get_parsl_config(run_dir).executors[0],
        ],
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

    # Load the single shared DFK once, then wrap each labeled
    # executor in its own ParslPoolExecutor restricted to that
    # pool. These wrappers do not own the DFK (config=None), so we
    # tear it down ourselves via dfk.cleanup() below.
    dfk = parsl.load(combined_config)
    sim_executor = ParslPoolExecutor(dfk=dfk, executors=['htex'])
    train_executor = ParslPoolExecutor(dfk=dfk, executors=['train_htex'])
    infer_executor = ParslPoolExecutor(dfk=dfk, executors=['infer_htex'])

    # Handle `kill <pid>` (SIGTERM). Parsl workers survive the main
    # process dying, and normal interpreter shutdown hangs after
    # atexit cleans up the DFK. Using os._exit() after cleanup
    # sidesteps the hang while still tearing down workers cleanly.
    def _handle_sigterm(*_: object) -> None:
        dfk.cleanup()
        os._exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        async with await Manager.from_exchange_factory(
            factory=create_exchange_factory(args.exchange),
            executors={
                'sim': sim_executor,
                'train': train_executor,
                'infer': infer_executor,
                'orchestrator': ThreadPoolExecutor(max_workers=1),
            },
            default_executor='orchestrator',
            log_config=recommended_logging(
                'INFO',
                logfile=cfg.output_dir / 'runtime.log',
            ),
        ) as manager:
            await run_ddwe_workflow(
                manager=manager,
                sim_agent_type=OpenMMSimAgent,
                training_agent_type=CVAETrainAgent,
                inference_agent_type=CVAEInferAgent,
                ddwe_agent_type=DDWEAgent,
                max_iterations=cfg.num_iterations,
                ensemble=ensemble,
                checkpointer=checkpointer,
                sim_agent_kwargs={
                    'sim_config': cfg.simulation_config,
                    'output_dir': cfg.output_dir / 'simulation',
                },
                training_agent_kwargs={
                    'train_config': cfg.train_config,
                    'output_dir': cfg.output_dir / 'DeepDriveWE_output',
                },
                inference_agent_kwargs={
                    'inference_config': cfg.inference_config,
                    'output_dir': cfg.output_dir / 'DeepDriveWE_output',
                },
                ddwe_agent_kwargs={
                    'use_stale_model': cfg.use_stale_model,
                    'output_dir': cfg.output_dir / 'DeepDriveWE_output',
                },
                sim_executor='sim',
                training_executor='train',
                inference_executor='infer',
                ddwe_executor='orchestrator',
                num_sim_agents=len(
                    cfg.sim_compute_config.available_accelerators,
                ),
            )
    finally:
        dfk.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
