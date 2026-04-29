# NeMo AutoModel -- Guide for AI Agents

NeMo AutoModel is a PyTorch-native training framework for LLMs, VLMs, diffusion
models, and retrieval models. It integrates with HuggingFace Transformers via
custom `NeMoAuto*` wrapper classes, uses YAML-driven recipe configs, and relies
on FSDP2/HSDP/DDP/DTensor/DeepEP for distributed training.

This document is the top-level reference for any AI agent working in this
repository. Read it first, then consult the relevant skill file for the task at
hand.

---

## Coding Style

- **Explicit over implicit.** Inline logic where possible; avoid hiding behavior
  behind unnecessary layers of indirection.
- **No speculative abstractions.** Do not add features, parameters, or
  generalization beyond what is explicitly asked for.
- **Formatter:** `ruff` with a line length of 120 and double quotes.
  Run `ruff format .` then `ruff check --fix .` before committing.
- **Type hints** are required on all public API signatures (functions, methods,
  class attributes exposed in `__init__.py`).
- **Docstrings** follow Google style.
- **Optional dependencies** must be guarded with `safe_import()` from
  `nemo_automodel.shared.import_utils`. Never let an optional import crash
  module loading.
- **Copyright header.** Every Python file must start with the NVIDIA copyright
  block. Do not remove or modify it.
- **Package management.** The project uses `uv`. Do not introduce `pip install`
  commands in scripts or docs, instead use `uv`.
- **Python version.** 3.10+ required. PyTorch 2.6+.

---

## Architecture Overview

```
automodel <command> <domain> -c <config.yaml>
    |
    v
_cli/app.py          -- routes command + domain to recipe scripts
    |
    v
recipes/             -- main training / eval entry points
  llm/
  vlm/
  diffusion/
  retrieval/
    |
    v
components/          -- modular building blocks
  models/            -- 27+ model families (LLM, VLM, MoE, ...)
  datasets/          -- LLM, VLM, diffusion data pipelines
  distributed/       -- FSDP2, HSDP, DDP utilities
  checkpoint/        -- async DCP, SafeTensors
  quantization/      -- FP8, QAT, calibration
  _peft/             -- LoRA, QLoRA adapters
  launcher/          -- Slurm, SkyPilot job submission
    |
    v
_transformers/       -- HuggingFace bridge
  auto_model.py      -- NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText, ...
  registry.py        -- MODEL_ARCH_MAPPING (model registration)
  capabilities.py    -- per-model feature detection flags
  infrastructure.py  -- device mesh setup for distributed training

_diffusers/          -- diffusion pipeline wrapper
  NeMoAutoDiffusionPipeline
```

### Entry Point

`_cli/app.py` parses `automodel <command> <domain>` and dispatches to the
matching recipe script. The `-c` flag points to a YAML config that drives all
component construction.

### Recipes

Files under `recipes/` are the primary training entry points. Each recipe
assembles a model, optimizer, dataloader, and trainer from its YAML config,
then runs the training loop.

### Components

Everything under `components/` is a self-contained building block. Components
are composed by recipes, never by each other (no hidden cross-component
imports).

### Transformers Bridge

`_transformers/` is the integration layer with HuggingFace:

- `auto_model.py` -- defines the `NeMoAuto*` classes that wrap
  `PreTrainedModel` with NeMo-specific functionality (distributed init,
  checkpoint hooks, backend dispatch).
- `registry.py` -- `MODEL_ARCH_MAPPING` maps architecture strings to model
  classes. Every new model must be registered here.
- `capabilities.py` -- declares per-model feature flags (supports_fp8,
  supports_moe, has_combined_qkv, etc.). These flags drive conditional logic
  throughout the framework.
- `infrastructure.py` -- builds the device mesh for FSDP2/HSDP and manages
  process-group lifecycle.

### Diffusers Bridge

`_diffusers/` wraps HuggingFace diffusion pipelines via
`NeMoAutoDiffusionPipeline`, providing the same recipe-driven config and
distributed training interface used by LLM/VLM recipes.

---

## Model Conventions

### Directory Layout

Each model lives under `components/models/<name>/` and contains:

| File                    | Purpose                                           |
|-------------------------|---------------------------------------------------|
| `model.py`             | Model class (inherits `PreTrainedModel` + `HFCheckpointingMixin`) |
| `state_dict_adapter.py`| Weight key mapping between HF and NeMo formats    |
| `config.py` (optional) | Custom config class if HF config is insufficient  |
| `layers.py` (optional) | Custom layer implementations                      |
| `rope_utils.py` (optional) | Model-specific RoPE variants                  |

### Inheritance

- All models inherit from `PreTrainedModel` and `HFCheckpointingMixin`.
- MoE models additionally inherit `MoEFSDPSyncMixin` for correct expert
  gradient synchronization under FSDP2.

### Registration

Every model must be added to `MODEL_ARCH_MAPPING` in
`_transformers/registry.py`. Without this entry the `NeMoAuto*` classes will
not find the model.

### Combined Projections

Combined projections (fused QKV, fused GateUp) use **interleaved layout** so
that tensor-parallel sharding splits evenly across heads/experts. Do not change
the interleave order without understanding the TP implications.

### Backend System

`BackendConfig` controls which kernel implementations are used for attention,
linear layers, normalization, RoPE, and expert dispatch. Backend selection is
set in the YAML config and threaded through model construction; individual
layers should never hard-code a backend choice.

---

## Config Pattern

### YAML and `_target_`

All YAML configs use the `_target_` key to specify the Python class or function
to instantiate. This is the same pattern used by Hydra/OmegaConf:

```yaml
model:
  _target_: nemo_automodel.components.models.llama.model.LlamaForCausalLM
  config:
    hidden_size: 4096
    num_attention_heads: 32
```

### Dataclass Configs

Every component config is a Python dataclass that exposes `to_dict()` and
`from_dict()` for serialization round-tripping. When adding a new config field,
always provide a default value and add it to both methods.

### Recipe Builder Functions

Recipes use a standard set of builder functions to construct components from
config dicts:

- `build_model()` -- instantiate and shard the model
- `build_optimizer()` -- create optimizer and LR scheduler
- `build_dataloader()` -- set up dataset, sampler, and DataLoader
- `build_trainer()` -- assemble the training loop

These builders read their parameters from the YAML config. Do not bypass them
with manual construction unless there is a strong reason.

---

## Available Skills

The `skills/` directory contains focused guides for common tasks. Each skill
file gives step-by-step instructions an AI agent can follow.

| # | Skill                    | Directory                  | Description                                      |
|---|--------------------------|----------------------------|--------------------------------------------------|
| 1 | model-onboarding         | `model-onboarding`         | Onboard a new LLM, VLM, OMNI, MoE, dLLM, text-to-image, text-to-video model family     |
| 2 | developer-guide          | `developer-guide`          | Environment setup and day-to-day dev workflow     |
| 3 | recipe-development       | `recipe-development`       | Create and modify training/eval recipes           |
| 4 | distributed-training     | `distributed-training`     | FSDP2, HSDP, pipeline parallelism, context parallelism |
| 5 | parity-testing           | `parity-testing`           | Verify numerical correctness against reference implementations |
| 6 | launcher-config          | `launcher-config`          | Slurm and SkyPilot job submission setup           |
| 7 | automodel-researcher     | `automodel-researcher`     | Autonomous fine-tuning experiment loops for minimizing validation loss |

Use `skills/automodel-researcher/` when the task is to run NemoClaw/autoresearch-style autonomous tuning for an AutoModel fine-tuning notebook and config. Start with `skills/automodel-researcher/README.md` for the high-level workflow and artifacts, then follow `skills/automodel-researcher/program.md` for the experiment budget, logging schema, GPU saturation pass, optimization loop, and final best-run notebook deliverable.

When starting a task, identify which skill applies and read that file for
domain-specific guidance before writing any code.
