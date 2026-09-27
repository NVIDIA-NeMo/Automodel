# `examples/` — recipe configs

Adds to the repository root `AGENTS.md`.

**Load the `nemo-automodel-recipe-development` skill before adding or changing
a recipe config.**

Every YAML here is user-facing documentation and a CI test input at the same
time. `.github/workflows/validate-recipe-configs.yml` lints these files on
every PR via `tools/lint_example_yamls.py`; run it locally before pushing:

```bash
python tools/lint_example_yamls.py --automodel-dir .
```

## Structural rules the linter enforces

- A top-level `recipe:` section must come **first**.
- A top-level `ci:` section must come **last**.
- No duplicate top-level keys.
- A `wandb:` block must set `enable: false`. Examples are run by people without
  W&B credentials; logging is opt-in. Omitting the block entirely is also fine.

## The `ci:` block

`ci:` configures how nemo-ci runs this recipe. Keys map to CI variables in
`CI_KEY_TO_VAR` in `tests/ci_tests/utils/generate_ci_tests.py`:

| Key | Sets |
|---|---|
| `time` | `TIME` (job wall clock) |
| `nodes`, `node_multiplier` | `TEST_NODE_COUNT`, `NODE_MULTIPLIER` |
| `max_steps` | `MAX_STEPS` |
| `local_batch_size`, `ep_size`, `nproc_per_node` | `LOCAL_BATCH_SIZE`, `EP_SIZE`, `CONFIG_NPROC_PER_NODE` |
| `cluster_tag` | `RESERVED_CLUSTER_TAG` |
| `recipe_owner` | `RECIPE_OWNER` |
| `env_vars` | Extra environment variables on the job |

`ci.time` is the job's wall clock. A recipe that needs longer than the default
must raise `time` — `dist_env.timeout_minutes` is a distributed-init timeout
and does not extend the job.

## Keep configs honest

`_target_` values are import paths nothing type-checks. When a class moves,
update every config that names it in the same change.

A config committed here is expected to run. If it needs hardware the CI
clusters do not have, say so in the PR rather than leaving a config that fails
for everyone who tries it.
