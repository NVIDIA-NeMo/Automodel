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

**Registry keys** -- use the literal qualname string, not the class object:

```python
# Good
"transformers.models.gemma3.modeling_gemma3.Gemma3ForCausalLM": _parallelize_gemma3,

# Bad -- forces the import
_get_class_qualname(Gemma3ForCausalLM): _parallelize_gemma3,
```

Generate the literal by running `_get_class_qualname` on the real class rather
than writing it by hand; aliased imports resolve to their true module path (e.g.
`CustomLlamaForCausalLM` -> `nemo_automodel.components.models.llama.model.LlamaForCausalLM`).

**Type annotations** -- put the import under `if TYPE_CHECKING:`. That needs
`from __future__ import annotations` at the top of the file so annotations are
never evaluated at runtime; `optimized_tp_plans.py` already has it, add it if the
file you are editing does not.

**Runtime use** (`isinstance`, attribute access) -- import inside the function.
After the first call it is a `sys.modules` lookup.

### Verifying a change

```bash
# where the time actually goes
python -X importtime -c "import nemo_automodel.components.distributed.parallelizer" 2>&1 | sort -t'|' -k2 -rn | head -20

# wall clock
python -c "import nemo_automodel.components.distributed.parallelizer"
```

If you touch `PARALLELIZE_FUNCTIONS` or `_get_model_layer_group_specs`, prove
equivalence by serializing both before and after your change and diffing them --
they must be byte-identical.
