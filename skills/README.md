# Skills

Task guides for AI coding agents using or developing NeMo AutoModel.

Skills are split into two locations:

- `skills/` contains customer-facing skills that can sync to the public catalog.
- `.agents/contributor-skills/` contains contributor-facing workflow skills that
  stay local to this repository and are not synced externally.

## Usage

Public skills are synced to the global Claude Code skill registry via CI and
are available to AI agents as invocable slash commands without any extra flags.

To invoke a skill manually, use `/<skill-name>` in your Claude Code session.

## Public Catalog Skills

| Skill | Description |
|---|---|
| `nemo-automodel-model-onboarding` | Onboard a new model family (LLM, VLM, MoE, etc.) |
| `nemo-automodel-recipe-development` | Create and modify training/eval recipes |
| `nemo-automodel-distributed-training` | FSDP2, HSDP, pipeline/context parallelism |
| `nemo-automodel-launcher-config` | Slurm and SkyPilot job submission |

## Contributor Skills

See [.agents/contributor-skills](../.agents/contributor-skills/README.md) for
the contributor-facing skills.

| Skill | Description |
|---|---|
| `build-and-dependency` | Container setup, uv package management, environment variables, CLI usage |
| `cicd` | Commit/PR workflow, CI trigger mechanism, failure investigation |
| `fern-docs` | Maintain the Fern docs site under `fern/`: pages, slugs, redirects, version aliases, library reference |
| `linting-and-formatting` | ruff rules, type hints, docstrings, copyright headers, code review checklist |
| `parity-testing` | Verify numerical correctness against references |
| `testing` | Unit and functional test layout, tier semantics (L0/L1/L2), adding tests |
