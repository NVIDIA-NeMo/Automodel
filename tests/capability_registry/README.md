# Model Capability Registry

A standalone CLI that **queries** which parallelism capabilities (TP, CP, PP, EP) a
NeMo AutoModel model claims to support, then **validates** each implemented and
supported capability by running a standardized parity test (no-parallelism
reference vs. with-parallelism variant) and checking that the per-token KL
divergence on post-training logits stays below a fixed threshold.

If the variant's logits diverge from the reference by more than the threshold,
the framework concludes **the registry lied** for that capability and exits
non-zero.

## Quick start

```bash
# Query only — no GPUs needed. Reports the supports_* flags.
python tests/capability_registry/query_and_validate_model_registry.py \
    --model_id meta-llama/Llama-3.1-8B --query_only

# Full validation — runs TP and CP tests on 2 GPUs, auto-spawning torchrun.
python tests/capability_registry/query_and_validate_model_registry.py \
    --model_id meta-llama/Llama-3.1-8B

# Validate just one capability:
python tests/capability_registry/query_and_validate_model_registry.py \
    --model_id meta-llama/Llama-3.1-8B --capabilities tp
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--model_id` | (required) | HF model id, e.g. `meta-llama/Llama-3.1-8B` |
| `--capabilities` | all supported+implemented | Subset of `{tp, cp, pp, ep}` to validate |
| `--kl_threshold` | `1e-2` | Max allowed per-token KL divergence |
| `--dtype` | `bfloat16` | Model dtype (`bfloat16` or `float32`) |
| `--world_size` | `2` | `nproc_per_node` for each torchrun spawn |
| `--num_steps` | `4` | K-1 train steps + 1 forward-capture step |
| `--local_batch_size` | `1` | Per-rank batch size |
| `--query_only` | off | Skip validation, just print the registry |
| `--report_json PATH` | none | Optional structured JSON output |

## How it works

1. **Query phase** (single-process, no GPU). Loads the model on the meta
   device via `AutoConfig` → `AutoModelForCausalLM.from_config` and inspects
   `model.supports_*` via NeMo's
   `nemo_automodel/_transformers/capabilities.py::ModelSupports`.
2. **Validation phase**. For each capability that is *both* supported by the
   model and implemented by this suite, the parent process spawns a
   `torchrun --nproc_per_node=<world_size>` child that runs the standardized
   test for that capability and writes a JSON result file. The parent reads
   the result and renders the final summary.
3. Each test internally:
   - All ranks load the reference (no-parallelism) model so NeMo's internal
     cross-rank barriers don't deadlock.
   - Rank 0 trains the reference for `num_steps - 1` steps with SGD on a
     small slice of the recipe dataset (HellaSwag by default) and captures
     final-step logits in eval mode.
   - All ranks build the variant with the parallelism plan applied (TP via
     `parallelize_module`, CP via `context_parallel`/`attach_cp_sdpa_hooks`),
     train the same steps on the same broadcast batches, and capture
     final-step logits.
   - Compute KL divergence and reduce MAX across ranks.

## Adding a new capability test

1. Create `tests/capability_registry/standardized_tests/test_<name>.py` exporting
   a class with the `CapabilityTest` shape (see `_base.py`):

   ```python
   class XTest:
       name = "x"
       implemented = True   # set False for SKIP stubs
       world_size = 2
       def run(self, *, model_id, dtype, kl_threshold, num_steps,
               local_batch_size) -> CapabilityTestResult: ...
   ```
2. Register it in `tests/capability_registry/_runner.py`:

   ```python
   from .standardized_tests.test_x import XTest
   CAPABILITY_TESTS["x"] = XTest()
   ```

   That's it — no other file changes needed. The CLI keys off
   `model.supports_<name>` and dispatches automatically.

## Adding a per-model dataset override

By default the suite uses HellaSwag (`rowan/hellaswag`), which works for any
causal LM. For models whose primary recipe uses a different dataset, add an
entry to `_MODEL_DATASET_OVERRIDES` in `_dataset_utils.py`.

## Known limitations

- **CP for HF Llama-3.1 via SDPA**: NeMo's introspection-based `supports_cp`
  flag reports `True` for any HF model with `_supports_sdpa=True`, but the
  end-to-end CP path through `torch.distributed.tensor.experimental.context_parallel`
  + HF SDPA produces logits that diverge significantly from the no-CP reference
  for this model. Production NeMo recipes that use CP rely on the TE attention
  backend (`backend.attn=te`) rather than vanilla SDPA, and the published Llama
  pretrain configs still pin `cp_size: 1` pending packed-sequence CP support.
  The framework correctly reports this as a "registry lied" result for
  Llama-3.1-8B.
- **Memory**: Llama-3.1-8B at bf16 takes ~16 GB per rank for params plus
  gradients; SGD is used (no optimizer state) so it fits comfortably on
  2×80 GB H100. For 2×40 GB cards consider reducing seq_len or enabling
  activation checkpointing.

## File layout

```
tests/capability_registry/
├── query_and_validate_model_registry.py   # CLI entry point
├── _kl_utils.py                           # KL divergence + DTensor gather
├── _distributed_utils.py                  # torchrun detect / spawn / barriers
├── _capability_query.py                   # registry: model_id -> supports_* dict
├── _dataset_utils.py                      # recipe-dataset batch builder
├── _runner.py                             # capability -> CapabilityTest dispatcher
├── standardized_tests/
│   ├── _base.py                           # CapabilityTest protocol + result dataclass
│   ├── test_tp.py                         # TP=2 vs TP=1 (implemented)
│   ├── test_cp.py                         # CP=2 vs CP=1 (implemented)
│   ├── test_pp.py                         # SKIP stub
│   └── test_ep.py                         # SKIP stub
└── README.md
```
