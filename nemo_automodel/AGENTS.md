# `nemo_automodel/` — package rules

Adds to the repository root `AGENTS.md`.

The package has three top-level entry surfaces: `cli/` (the `automodel` CLI and
launcher dispatch), `recipes/` (end-to-end training workflows), and
`components/` (self-contained, dependency-light modules). `_transformers/` and
`_diffusers/` are the HuggingFace bridges. See `docs/repository-structure.mdx`
for the full tour.

## Config pattern

Every component config is a typed Python dataclass. When adding a field, provide
a backward-compatible default and keep consumers on the typed object — do not
thread raw dicts through component APIs.

YAML configs use the `_target_` key to name the class or function to
instantiate, the same pattern as Hydra/OmegaConf:

```yaml
model:
  _target_: nemo_automodel.components.models.llama.model.LlamaForCausalLM
  config:
    hidden_size: 4096
    num_attention_heads: 32
```

`_target_` strings are import paths that no compiler checks. When you move or
rename a class, grep `examples/` and `tests/` for the old path in the same
change.

## Construction

The root file's config-owned-construction and no-free-standing-builder rules
apply to everything here. Concretely, inside this package:

- A `build(...)` method reads declarative fields off `self` and takes
  runtime-only values as explicit typed arguments.
- Runtime objects (process groups, meshes, tokenizers, devices) are never
  stored back onto the config — the config must stay serializable after
  `build(...)` returns.

## Optional dependencies

Many components import optional backends (Transformer Engine, DeepEP, Megatron,
tilelang). Guard every such import with `safe_import()` from
`nemo_automodel.shared.import_utils` so that importing a module never fails on
a machine that lacks the backend. Import failures must surface at use time with
an actionable message, not at module load.
