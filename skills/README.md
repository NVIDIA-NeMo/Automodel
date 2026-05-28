# Public Skills

Customer-facing task guides for AI coding agents using NeMo AutoModel.

Only skills intended for the public catalog live under `skills/`. Contributor
workflow skills live under `.agents/contributor-skills/` so they remain
available in this repository without being synced externally.

## Usage

These public skills are synced to the global Claude Code skill registry via CI
and are available to AI agents as invocable slash commands without any extra
flags.

To invoke a skill manually, use `/<skill-name>` in your Claude Code session.

## Available skills

| Skill | Description |
|---|---|
| `model-onboarding` | Onboard a new model family (LLM, VLM, MoE, etc.) |
| `recipe-development` | Create and modify training/eval recipes |
| `distributed-training` | FSDP2, HSDP, pipeline/context parallelism |
| `launcher-config` | Slurm and SkyPilot job submission |
