# `components/distributed` -- Guide for AI Agents

Additive to the repository root `AGENTS.md`. Only the rules specific to this
directory are listed here.

## Never import a concrete model class at module scope

Do **not** add a module-scope import of:

- `transformers.models.<arch>.modeling_<arch>` (any `…ForCausalLM`,
  `…ForConditionalGeneration`, `…ForSequenceClassification`), or
- `nemo_automodel.components.models.<arch>.model`

to any file in this directory. `parallelizer.py` and `optimized_tp_plans.py`
are imported by nearly everything, so an import added here is paid everywhere.

### Why

Importing one modeling module drags the whole model zoo in behind it:

```
transformers.models.gemma3.modeling_gemma3
  -> transformers.generation.candidate_generator -> sklearn -> pandas, scipy
  -> transformers.processing_utils -> image_utils -> torchvision
  -> transformers.generation.continuous_batching -> opentelemetry
```

That single line measured **+2.24 s** on top of `import transformers` -- more
than `import torch` (1.74 s) costs by itself. Removing these imports took
`import nemo_automodel.components.distributed.parallelizer` from **7.40 s to
4.12 s** (local, py3.10 / torch 2.10).

The cost is paid by every process that touches the module: pytest startup, the
unit-test collection phase, and -- worst -- each `mp.spawn` child in the test
suite, which re-imports from scratch. In one measured CPU test that was 10.90 s
of imports for 2.27 s of actual work.

It is also a cascade: each of `parallelizer.py`, `optimized_tp_plans.py` and the
`components/models/*` modules it reaches independently pins the same graph, so
fixing one layer alone shows **zero** improvement. Re-measure after each change
rather than assuming (see below).

### What to do instead

Pick whichever fits the use:

**Per-model contracts** -- declare them on the model class, not in this directory:

```python
# components/models/<name>/model.py
class NewModelForCausalLM(HFCheckpointingMixin, nn.Module):
    parallel_spec: ParallelSpec = ParallelSpec(tp_plan=_new_model_tp_plan)
```

`ParallelSpec` (`parallel_spec.py`) is pure data: the TP plan and its sequence-parallel
overlay as dictionaries, layer groups, the `sharded_output_only` constraint, the `shard_by_dtype`
opt-in (dtype-aware per-block FSDP2 driven by the model's `_keep_in_fp32_modules_strict`) and the
strategy override (the one place for instance-dependent behaviour; `DefaultParallelizationStrategy`
exposes hooks such as `select_activation_checkpointing_layers` so a strategy narrows one step instead
of re-implementing the flow). It never points at a model attribute; what must come from the instance
(HF `_tp_plan`, input embedding, text config) is read through the model's own API, and
`ParallelSpec.from_hf_model` is the constructor for that case. Layer groups are derived when not
declared: containers of the blocks the model names in `_no_split_modules`, with the role taken from
the multimodal tower they live under, else `language` for a model with `get_decoder()` and
`backbone` otherwise (`_derive_layer_groups`; the same `_no_split_modules` feed the MegatronFSDP unit
derivation). A wrapper that declares nothing resolves through the child its `base_model_prefix`
names. `query_parallel_spec` (in `parallel_spec.py`) only ever reads the `parallel_spec` class
attribute; activation checkpointing has its own `ActivationCheckpointingSpec`
(`activation_checkpointing.py`, read from the separate `activation_checkpointing_spec` attribute by
`query_activation_checkpointing_spec`). Architectures the repository does not re-implement -- stock
`transformers` classes, `trust_remote_code` checkpoints, `diffusers` transformers -- get a
`components/models/<family>/parallelization.py` of their own only when they need to declare
something; the loader resolver (`_transformers/model_init.py::model_specs_for`, shared by the
diffusion pipeline) derives `<family>` from the class and binds every declared spec onto the
wrapper. No table of model names exists anywhere, and nothing in this directory names a model.

**Type annotations** -- put the import under `if TYPE_CHECKING:`. That needs
`from __future__ import annotations` at the top of the file so annotations are
never evaluated at runtime; `optimized_tp_plans.py` already has it, add it if the
file you are editing does not.

**Runtime use** (`isinstance`, attribute access) -- import inside the function.
After the first call it is a `sys.modules` lookup.

**Keep the compiler out of the import path too.** `torch._dynamo` (+ triton, +0.5 s) enters
through `torch.distributed.pipelining` and `torch._functorch.partitioners`; both are deferred
on purpose -- `pipelining/__init__.py` resolves `AutoPipeline` lazily, and
`activation_checkpointing.py` builds the selective-AC save-set on first use. Do not add a
module-scope import of either, or a module-scope call that reaches them.
`tests/unit_tests/test_import_hygiene.py` fails if they load.

### Verifying a change

```bash
# where the time actually goes
python -X importtime -c "import nemo_automodel.components.distributed.parallelizer" 2>&1 | sort -t'|' -k2 -rn | head -20

# wall clock
python -c "import nemo_automodel.components.distributed.parallelizer"
```

If you touch a `parallel_spec` (or the bridge tables), prove equivalence by serializing
the resolved spec fields for every affected class before and after your change and
diffing them -- they must be identical.
