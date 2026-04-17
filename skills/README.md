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
| `developer-guide` | Environment setup and day-to-day dev workflow |
| `recipe-development` | Create and modify training/eval recipes |
| `parity-testing` | Verify numerical correctness against references |
| `distributed-training` | FSDP2, HSDP, pipeline/context parallelism |
| `launcher-config` | Slurm and SkyPilot job submission |