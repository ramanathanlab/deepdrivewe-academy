---
name: academy-agent-framework
description: Use this skill when building, extending, or debugging agents with the Academy multi-agent framework (academy-py). Covers Agent subclassing, @action/@loop decorators, Manager lifecycle, Handle proxies, exchange backends, and executor strategies.
---

# Academy Agent Framework

Reference guide for the **Academy** framework (`academy-py`), a modular middleware for building and deploying stateful actors and autonomous agents across distributed systems and federated research infrastructure.

## When to Activate

- Creating or modifying an `Agent` subclass
- Wiring agents together with `Handle` references
- Configuring a `Manager` and choosing an exchange backend
- Choosing between `ThreadPoolExecutor`, `ProcessPoolExecutor`, `ParslPoolExecutor`, or `GlobusComputeExecutor`
- Debugging deadlocks, startup ordering, or blocking-in-async issues

## Core Concepts

### Agent
Defined by subclassing `academy.agent.Agent`. An agent has:
- **State** - instance attributes set in `__init__`
- **Actions** - async methods decorated with `@action` (remotely invocable)
- **Loops** - async methods decorated with `@loop` (autonomous, non-terminating; receive `shutdown: asyncio.Event`)
- **Lifecycle callbacks** - `agent_on_startup()` and `agent_on_shutdown()`

```python
from academy.agent import Agent, action, loop
import asyncio

class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.count = 0

    @action
    async def increment(self) -> int:
        self.count += 1
        return self.count

    @loop
    async def heartbeat(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            await asyncio.sleep(5)
            print(f"count={self.count}")
```

### Handle
A client reference to a remote agent. Translates method calls into async messages. Awaiting an action on a handle blocks until the result is returned.

```python
handle = await manager.launch(MyAgent)
result = await handle.increment()  # blocks until result
await handle.shutdown()
```

### Manager
High-level coordinator for launching agents. Created with `Manager.from_exchange_factory()` as an async context manager.

```python
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from concurrent.futures import ThreadPoolExecutor

async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
    executors=ThreadPoolExecutor(),
) as manager:
    handle = await manager.launch(MyAgent)
    ...
```

### Exchange (Message Routing)
Configures how agents communicate. Choose based on deployment:

| Factory | Use Case |
|---|---|
| `LocalExchangeFactory` | Single process, in-memory (dev/testing) |
| `RedisExchangeFactory` | Multi-node distributed, resilient |
| `HttpExchangeFactory` | Cloud-hosted, REST API, supports auth |
| `HybridExchangeFactory` | Direct TCP + Redis fallback |

All are imported from `academy.exchange`.

### Executor (Agent Execution)
Configures where agents run:

| Executor | Use Case |
|---|---|
| `ThreadPoolExecutor` | Same process, threads (simple) |
| `ProcessPoolExecutor` | Multiple processes, same machine |
| `ParslPoolExecutor` | HPC clusters (SLURM, HTCondor) |
| `GlobusComputeExecutor` | FaaS cloud (remote HPC) |

## Key Rules & Best Practices

1. **Never communicate in `__init__`** — use `agent_on_startup()` for any inter-agent communication at startup.
2. **Avoid blocking calls in async methods** — use `await self.agent_run_sync(blocking_fn)` to run blocking code in a thread.
3. **Beware circular startup requests** — agents awaiting each other's startup callbacks will deadlock.
4. **Use `asyncio.Task` for concurrent action invocation** — don't `await` multiple handles sequentially if you want parallelism; create tasks.
5. **Configure logging on the Manager** — pass `log_config=recommended_logging(...)` to `Manager.from_exchange_factory()`; it propagates to every agent the Manager launches, including remote workers. (In academy < 0.5 you instead called the removed `init_logging()` and passed an executor `initializer`.)

## Examples

### Example 1: Stateful Actor (examples/01-actor-client)
State is initialized in `agent_on_startup()`, not `__init__`. Actions are invoked sequentially from the client.

```python
from academy.agent import action, Agent
from academy.exchange import LocalExchangeFactory
from academy.logging.recommended import recommended_logging
from academy.manager import Manager
from concurrent.futures import ThreadPoolExecutor
import asyncio

class Counter(Agent):
    count: int

    async def agent_on_startup(self) -> None:
        self.count = 0

    @action
    async def increment(self, value: int = 1) -> None:
        self.count += value

    @action
    async def get_count(self) -> int:
        return self.count

async def main() -> None:
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
        log_config=recommended_logging('INFO'),
    ) as manager:
        agent = await manager.launch(Counter)
        await agent.increment()
        count = await agent.get_count()  # → 1

asyncio.run(main())
```

### Example 2: Agent with Control Loop (examples/02-agent-loop)
A `@loop` runs autonomously in the background until the agent shuts down.

```python
class Counter(Agent):
    count: int

    async def agent_on_startup(self) -> None:
        self.count = 0

    @loop
    async def increment(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            await asyncio.sleep(1)
            self.count += 1

    @action
    async def get_count(self) -> int:
        return self.count

# After launch, wait for loops to run, then read state:
agent = await manager.launch(Counter)
await asyncio.sleep(2)
count = await agent.get_count()  # ≥ 1
```

### Example 3: Multi-Agent Pipeline (examples/03-agent-agent)
Pass handles as constructor arguments to wire agents together into a pipeline.

```python
from academy.handle import Handle

class Coordinator(Agent):
    def __init__(self, lowerer: Handle[Lowerer], reverser: Handle[Reverser]) -> None:
        super().__init__()
        self.lowerer = lowerer
        self.reverser = reverser

    @action
    async def process(self, text: str) -> str:
        text = await self.lowerer.lower(text)
        text = await self.reverser.reverse(text)
        return text

class Lowerer(Agent):
    @action
    async def lower(self, text: str) -> str:
        return text.lower()

class Reverser(Agent):
    @action
    async def reverse(self, text: str) -> str:
        return text[::-1]

# Launch workers first, then pass their handles to the coordinator:
async with await Manager.from_exchange_factory(...) as manager:
    lowerer = await manager.launch(Lowerer)
    reverser = await manager.launch(Reverser)
    coordinator = await manager.launch(Coordinator, args=(lowerer, reverser))
    result = await coordinator.process('DEADBEEF')  # → 'feebdaed'
```

### Example 4: Distributed Execution with HTTP Exchange (examples/04-execution)
Use `spawn_http_exchange` + `ProcessPoolExecutor` to run agents in separate processes.
The Manager's `log_config` propagates to those worker processes, so no
executor `initializer` is needed for logging.

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from academy.exchange.cloud import spawn_http_exchange
from academy.logging.recommended import recommended_logging

async def main() -> None:
    with spawn_http_exchange('localhost', 5346) as factory:
        mp_context = multiprocessing.get_context('spawn')
        executor = ProcessPoolExecutor(
            max_workers=3,
            mp_context=mp_context,
        )
        async with await Manager.from_exchange_factory(
            factory=factory,
            executors=executor,
            log_config=recommended_logging('INFO'),  # propagates to workers
        ) as manager:
            lowerer = await manager.launch(Lowerer)
            reverser = await manager.launch(Reverser)
            coordinator = await manager.launch(Coordinator, args=(lowerer, reverser))
            result = await coordinator.process('DEADBEEF')
```

### Example 5: LLM Orchestrator Agent with pydantic-ai
Wrap academy handles as pydantic-ai tools and use them inside an Orchestrator agent.
Initialize the AI agent in `agent_on_startup()` so the handle is available at construction time.

```python
from pydantic_ai import Agent as AIAgent
from academy.handle import Handle

class SimAgent(Agent):
    """Agent for running computational tools on molecules."""

    @action
    async def compute_ionization_energy(self, smiles: str) -> float:
        """Compute the ionization energy for the given molecule."""
        return 0.5  # replace with real computation

class Orchestrator(Agent):
    def __init__(self, model: str, simulators: list[Handle[SimAgent]]) -> None:
        super().__init__()
        self.model = model
        self.simulators = simulators

    async def agent_on_startup(self) -> None:
        # Close over handles to build async tool functions for pydantic-ai
        sim = self.simulators[0]

        async def compute_ionization_energy(smiles: str) -> float:
            """Compute the ionization energy of a molecule given its SMILES string."""
            return await sim.compute_ionization_energy(smiles)

        self.ai_agent = AIAgent(self.model, tools=[compute_ionization_energy])

    @action
    async def answer(self, goal: str) -> str:
        """Use simulator agents to answer questions about molecules."""
        result = await self.ai_agent.run(goal)
        return result.data

# Wire them up:
async with await Manager.from_exchange_factory(factory=LocalExchangeFactory()) as manager:
    simulator = await manager.launch(SimAgent)
    orchestrator = await manager.launch(
        Orchestrator,
        kwargs={'model': 'openai:gpt-4o', 'simulators': [simulator]},
    )
    result = await orchestrator.answer('What is the ionization energy of benzene?')
```

## Common Patterns

### Self-Termination
```python
await self.agent_shutdown()
```

### Deferred Initialization
Launch the agent process before the agent object is constructed (useful when initialization needs exchange context):
```python
handle = await manager.launch(MyAgent, args=(...), deferred=True)
await handle.agent_start()
```

### Distributed with Redis
```python
from academy.exchange import RedisExchangeFactory
from concurrent.futures import ProcessPoolExecutor

async with await Manager.from_exchange_factory(
    factory=RedisExchangeFactory('<REDIS HOST>', port=6379),
    executors=ProcessPoolExecutor(max_workers=4),
) as manager:
    ...
```

### Running the HTTP Exchange Server
```bash
python -m academy.exchange.cloud --config exchange.toml
```

## Installation
```bash
pip install academy-py
```

## Project Layout
```
academy/
├── academy/
│   ├── agent.py        # Agent base class, @action, @loop
│   ├── handle.py       # Handle protocol
│   ├── manager.py      # Manager
│   ├── runtime.py      # Agent runtime
│   ├── exchange/       # LocalExchangeFactory, RedisExchangeFactory, etc.
│   ├── message.py      # Message types
│   └── logging/        # LogConfig, recommended_logging()
├── docs/               # MkDocs documentation
├── examples/           # Runnable examples (01-actor-client through 08-discussion)
└── tests/
    ├── unit/
    └── integration/
```

## Resources
- Docs: https://docs.academy-agents.org
- GitHub: https://github.com/academy-agents/academy
- Paper: https://arxiv.org/abs/2505.05428
