# Skills

Reusable task guides for AI coding agents working in this repo.

Skills are defined centrally here and symlinked into the agentic directories
that need them, keeping a single source of truth without polluting personal
`.claude/skills/` setups.

## Usage

Skills are synced to the global Claude Code skill registry via CI and are
available to AI agents as invocable slash commands without any extra flags.

To invoke a skill manually, use `/<skill-name>` in your Claude Code session.

## Available skills

| Skill | Description |
|---|---|
| `model-onboarding` | Onboard a new model family (LLM, VLM, MoE, etc.) |
| `recipe-development` | Create and modify training/eval recipes |
| `parity-testing` | Verify numerical correctness against references |
| `distributed-training` | FSDP2, HSDP, pipeline/context parallelism |
| `launcher-config` | Slurm and SkyPilot job submission |
| `linting-and-formatting` | ruff rules, type hints, docstrings, copyright headers, code review checklist |
| `cicd` | Commit/PR workflow, CI trigger mechanism, failure investigation |
| `build-and-dependency` | Container setup, uv package management, environment variables, CLI usage |
| `testing` | Unit and functional test layout, tier semantics (L0/L1/L2), adding tests |
| `bump-dependency` | Bump a pinned dependency, regenerate `uv.lock` + `uv-pytorch.lock`, open a PR, and watchdog `CICD NeMo` to green |
