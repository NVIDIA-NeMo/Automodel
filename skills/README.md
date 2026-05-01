# Skills

Reusable task guides for AI coding agents working in this repo.

Shared skills live here (`skills/`) so they don't conflict with anyone's
personal `.claude/skills/` directory.

## Usage

### Claude Code

Launch Claude Code with the `--add-dir` flag to auto-register the shared
skills as slash commands:

```bash
claude --add-dir skills
```

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