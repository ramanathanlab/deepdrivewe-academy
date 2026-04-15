# Claude Skill: Build a DeepDriveWE Example

This project ships a [Claude Code](https://claude.com/claude-code)
skill that walks Claude through creating a new DeepDriveWE example
end-to-end. Install it once, then ask Claude something like
*"build a new DeepDriveWE example for ligand binding with AMBER"* --
Claude automatically loads the skill and follows the project's
conventions (directory layout, agent subclassing, binner/resampler
choices, Parsl compute configs, verification steps).

## What the Skill Covers

- Required project layout (`main.py`, `workflow.py`, `config.yaml`,
  `common_files/`, `inputs/bstates/`).
- Why Parsl-serializable classes must live in `workflow.py`, not
  `main.py`.
- How to subclass
  [`SimulationAgent`][deepdrivewe.workflows.westpa.SimulationAgent]
  and [`WestpaAgent`][deepdrivewe.workflows.westpa.WestpaAgent].
- Selection guide for binners, recyclers, resamplers, and compute
  configs.
- Checkpoint/resume pattern and SIGTERM handling.
- Verification steps before scaling up.

## Where to Install It

Claude Code loads skills from two locations:

=== "User-level (recommended)"

    ```
    ~/.claude/skills/deepdrivewe-example/SKILL.md
    ```

    Available in every project you work on. Install this way if you
    plan to build DeepDriveWE examples across multiple checkouts or
    downstream projects.

=== "Project-level"

    ```
    .claude/skills/deepdrivewe-example/SKILL.md
    ```

    Committed to this repo and automatically available whenever you
    open it in Claude Code. No install step needed -- it's already
    here.

## Install (User-Level)

=== "curl"

    ```bash
    mkdir -p ~/.claude/skills/deepdrivewe-example
    curl -fsSL \
      https://raw.githubusercontent.com/braceal/deepdrivewe-academy/main/.claude/skills/deepdrivewe-example/SKILL.md \
      -o ~/.claude/skills/deepdrivewe-example/SKILL.md
    ```

=== "From a local checkout"

    ```bash
    mkdir -p ~/.claude/skills/deepdrivewe-example
    cp .claude/skills/deepdrivewe-example/SKILL.md \
       ~/.claude/skills/deepdrivewe-example/SKILL.md
    ```

=== "wget"

    ```bash
    mkdir -p ~/.claude/skills/deepdrivewe-example
    wget -O ~/.claude/skills/deepdrivewe-example/SKILL.md \
      https://raw.githubusercontent.com/braceal/deepdrivewe-academy/main/.claude/skills/deepdrivewe-example/SKILL.md
    ```

Once installed, Claude Code will see `deepdrivewe-example` in its
skill list on the next session start.

## How to Trigger It

Just ask in natural language. Claude matches the user request against
the skill's description and loads the skill automatically.

Example prompts:

> "Create a new DeepDriveWE example for NTL9 folding using AMBER
> instead of OpenMM."

> "Build a new weighted ensemble workflow to study ligand unbinding
> from a kinase. Use a distance-based progress coordinate."

> "Set up a minimal DDWE example that runs on a single laptop CPU
> with a synthetic simulation."

## Contents Preview

The skill is ~200 lines of structured guidance. A condensed outline:

1. **Before You Start** -- requirements gathering (system, MD engine,
   progress coordinate, target direction, compute, starting point).
2. **Project Layout** -- required files and the critical rule about
   where serializable classes must live.
3. **Step-by-Step** (6 steps) -- config models, agent subclassing,
   `main.py` pattern, `config.yaml` pattern, verification.
4. **Component Selection Guide** -- tables for picking the right
   binner, recycler, resampler, and compute config.
5. **Common Extension Points** -- custom pcoords, data products,
   resamplers, ML-driven workflows.
6. **Gotchas** -- project conventions and easy-to-miss pitfalls.

View the full source at
[.claude/skills/deepdrivewe-example/SKILL.md](https://github.com/braceal/deepdrivewe-academy/blob/main/.claude/skills/deepdrivewe-example/SKILL.md).

## Updating the Skill

If you improve the skill locally, please open a PR against `develop`
so everyone benefits. See the [Contributing](../contributing.md) guide.
