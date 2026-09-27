# `nemo_automodel/recipes/` — training entry points

Adds to the repository root `AGENTS.md`.

**Load the `nemo-automodel-recipe-development` skill before adding or changing
a recipe.**

Recipes under `llm/`, `vlm/`, `multimodal/`, `diffusion/`, `dllm/`, and
`retrieval/` are the primary training and eval entry points, built on
`base_recipe.py` with shared helpers in `_dist_utils.py`, `_typed_config.py`,
and `kd_utils.py`.

## Recipes are thin

A recipe assembles a model, optimizer, dataloader, and trainer from its YAML
config and runs the loop. It composes `config.build(...)` results through
public component APIs and owns no component construction logic of its own.

If a recipe needs behavior a component does not expose, add it to the component
and its typed config — do not inline it into the recipe, and do not reach into
component internals. Logic that lands in a recipe is invisible to every other
recipe and to every test that does not run that recipe end to end.

## Configuration reaches recipes only through config objects

Do not read environment variables or global state inside a recipe to change
training behavior. Every knob is a config field so that the run is reproducible
from its YAML alone.

## Parallelism plumbing

Recipes construct the device mesh and hand it to components as an explicit
`build(...)` argument. When adding a parallelism dimension, check that every
recipe that supports it validates the topology up front — a misconfigured mesh
should fail at setup with a clear message, not deadlock at step 12.
