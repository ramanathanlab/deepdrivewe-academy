# Quickstart

This guide walks through the minimal steps to run a weighted ensemble
workflow with DeepDriveWE-Academy.

## Concepts

A DeepDriveWE workflow has two agent types:

1. **SimulationAgent** -- runs one MD simulation per call and returns a
   `SimResult`.
2. **WestpaAgent** -- collects results from all walkers in an iteration,
   applies binning/recycling/resampling, and dispatches the next round.

You subclass each agent to inject your own simulation and inference
logic, then hand them to `run_westpa_workflow` which handles
registration, launching, and communication.

## Minimal Pattern

The simplest workflow defines `run_simulation` and `run_inference`:

```python
from deepdrivewe.api import SimMetadata, SimResult, IterationMetadata
from deepdrivewe.workflows import SimulationAgent, WestpaAgent


class MySimAgent(SimulationAgent):
    def run_simulation(self, metadata: SimMetadata) -> SimResult:
        # Your simulation logic here
        ...
        return SimResult(data={'pcoord': pcoord}, metadata=metadata)


class MyWestpaAgent(WestpaAgent):
    def run_inference(
        self,
        sim_results: list[SimResult],
    ) -> tuple[list[SimMetadata], list[SimMetadata], IterationMetadata]:
        # Your binning + resampling logic here
        ...
        return cur_sims, next_sims, metadata
```

## Running the Workflow

Wire everything together with `run_westpa_workflow`:

```python
import asyncio
from academy.exchange.local import LocalExchangeFactory
from academy.manager import Manager
from deepdrivewe.api import WeightedEnsemble
from deepdrivewe.workflows import run_westpa_workflow


async def main():
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        await run_westpa_workflow(
            manager=manager,
            sim_agent_type=MySimAgent,
            westpa_agent_type=MyWestpaAgent,
            max_iterations=10,
            ensemble=ensemble,  # WeightedEnsemble with basis/target states
        )

asyncio.run(main())
```

## Next Steps

- See the [OpenMM NTL9 tutorial](../tutorials/openmm-ntl9-hk.md) for a
  complete protein-folding example with GPU-distributed MD.
- Read the [Architecture](../concepts/architecture.md) page to
  understand the agent communication model.
- Browse the [API Reference](../reference/) for full class
  documentation.
