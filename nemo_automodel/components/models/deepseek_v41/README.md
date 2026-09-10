# DeepSeek V4.1 Flash

This implementation follows the [released report, configuration, and inference
code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40)
at revision `df42c109f1defefcbfcedbe7d905718a12266e40`. The numerical oracle is
the unchanged HF-hosted `inference/` implementation; the dump does not provide
a Transformers modeling implementation.

The model contains a 40-layer causal encoder/decoder backbone with 384 routed
experts and one shared expert per layer, top-6 routing, CSA2 Full/Reindex/Reuse
attention, hierarchical candidate selection, single-pass mHC, and Engram at
layers 1 and 14. The approximately 552B backbone and 196B Engram tables remain
separate parameter groups in the architecture. Optional image support uses
the released vision encoder, spatial merger, image delimiters and visual
router bias. DSpark draft weights under `mtp.*` are excluded.

## Implementation and precision

- [model.py](model.py) owns the backbone loop, per-layer CSA2 state snapshots,
  model-owned Engram sharding, image insertion and FSDP precision contract.
- [layers.py](layers.py) implements compression, index reuse, candidate masks,
  mHC and attention, reusing the shared DeepSeek V4 sparse kernels.
- [engram.py](engram.py) supplies V4.1 hashes and gates; its embedding reuses
  the Qwen3.8 Flash Next owner-sharded table and autograd communication.
- [state_dict_adapter.py](state_dict_adapter.py) maps the released layouts and
  supports bounded direct initialization from safetensors.
- [processing.py](processing.py) and [vision.py](vision.py) provide ordinary
  text/image processing and the reused DeepSeek V4 vision modules.

CUDA model construction defaults to `torch_mm` experts with `hybridep`
dispatch, `torch` linears, `torch_fp32` normalization and `tilelang` attention;
the validation recipe selects these explicitly. CPU construction uses local
PyTorch experts/dispatch and SDPA. The language-model head computes and
returns FP32 logits without rounding them back to BF16.
Shared MoE retains the FP32 combine contract: individual routed contributions
return to their source ranks, accumulate in expert order, and combine with
the shared expert before the final BF16 cast. This requires additional
communication compared with the usual BF16 HybridEP combine.

The default MoE `gate_bias_update_factor` is zero, preserving the released
router correction biases during fine-tuning. An explicit positive override
enables the shared MoE's routing-bias updates.

`torch_linear` is an optional explicit expert backend using separate eager
projections with the same shared parameter storage and dispatcher. It is not
the selected validation backend. Its historical bitwise reference results
must not be attributed to `torch_mm`, whose grouped GEMMs and fused pointwise
operations have different rounding boundaries.

Released FP8 dense matrices, packed FP4 experts and rowwise FP8 Engram tables
initialize floating-point training parameters. With `kv_cache_fake_quant: true`,
the forward retains FP8 window KV, NVFP4 compressed KV and MXFP4 index query/key
quantize/dequantize boundaries. Training uses straight-through gradients.
Indexer parameters remain frozen because discrete top-k does not provide a
language-model gradient; indexer distillation is outside this implementation.

## Training recipe and ownership

[The EP64 recipe](../../../../examples/llm_finetune/deepseek_v41/deepseek_v41_flash_hellaswag_ep64_16nodes.yaml)
targets 16 nodes with four GB200 GPUs each in one verified NVLink domain:
WORLD=64, EP=64, 64 Engram owners, and TP1/PP1/CP1. Dense parameters use FSDP2;
experts have no additional FSDP shard axis. Global batch 64 and local batch 1
give one microbatch per optimizer update. AC and reshard-after-forward are
enabled. The schedule requests 100 updates with validation and checkpoints
after updates 50 and 100.

Engram tables are enabled but frozen by default in the model config. This
recipe explicitly sets `engram_trainable: true` for both tables. Each table
is a registered global row-sharded DTensor, excluded from FSDP all-gathers.
Its owner mesh defaults to WORLD and must match the dense FSDP ranks and
ordering. Owner gradients receive the owner divisor once before global
clipping. Disabling Engram changes the model function.

The recipe uses TE FusedAdam with FP32 moments and int16 remainders for BF16
master weights. It preserves FP32 mHC coefficients across FSDP boundaries with
`output_dtype: null` and `cast_forward_inputs: false`. This is fine-tuning,
not a reproduction of the report's Engram pretraining optimizer.

Supply a shared local snapshot of the pinned revision and an output directory
outside the source checkout through the launcher's overrides documented in
the YAML header. The launcher owns rank placement, communication setup and
reusable compilation caches. The recipe does not establish memory fit: the
original full-resident 16-node run completed update 0 and then OOMed during
the next backward, despite AC and resharding.

## Checkpoint APIs

For normal recipe initialization, set
`checkpoint.dequantize_base_checkpoint: true`. Save trained parameters without
quantization; the adapter exports released names and retains required FP32
weights. Engram save/load trims logical rows and restores only allocation
padding. Removing the original quantization metadata is appropriate for these
decoded floating-point checkpoints; quantized re-export is not established.

The explicit loader accepts an already materialized model with its final
FSDP/expert/Engram ownership and is used outside an active training graph:

```python
audit = model.state_dict_adapter.load_from_checkpoint(model, checkpoint_path)
```

It copies bounded slices directly into model-owned storage and rejects missing
keys, incompatible decoded shapes, non-aliasing expert destinations, and
incorrect local storage for strict FP32 parameters. It neither gathers an
Engram table nor materializes the complete checkpoint. The returned
`CheckpointLoadAudit` contains `loaded_keys`, `source_bytes`, `loaded_bytes`,
`max_chunk_source_bytes` and `max_chunk_output_bytes`; scratch counters exclude
model storage. `get_hf_state_dict_keys(state_dict)` discovers rank-independent
released names before owner DTensor preparation. The ordinary DCP path
retains its existing conversion/reconstruction behavior.

## Inputs and limits

Text supports padded batches and the PR's packed THD path with document
metadata. Engram hashing requires `input_ids` and the pinned tokenizer;
configuration-owned tokenizer construction uses the resolved checkpoint
commit. Checkpoint-free callers can supply a tokenizer explicitly, and
materialization restores the tokenizer-derived map and deterministic hash
buffers after meta initialization. `inputs_embeds` alone cannot supply those
hashes. The standalone `labels` API accepts unpacked `[batch, sequence]`
labels; packed text uses the external recipe loss. TP, PP, CP, inference KV
caches, bounded decoder replay and DSpark training are not implemented.
Million-token training has not been validated.

Omitting vision configuration creates a zero-layer tower; loading the released
vision metadata enables its declared layers. Both HellaSwag recipes explicitly
disable the tower. `bias_vl` remains present even in text-only gates.
`DeepseekV41Processor` provides standard text/image conversations and
save/reload. Image batches require unpacked `input_ids [batch, sequence]`,
`pixel_values [all_patches, 3, patch_size, patch_size]`,
`image_grid_hws [images, 2]`, and `vision_token_types [batch, sequence]`.
Packed images are rejected. Use the official encoder for specialized tool or
reasoning formatting before passing rendered text to the processor.

## Validation provenance

CPU regression tests cover reference arithmetic, packed text, image processing,
FP32 logits and loss, and real Gloo owner gradients and clipping. Distributed
checkpoint tests exercise quantized initialization and trained SafeTensors
restore with EP2 and EP2 plus an inner FSDP shard axis. These tests use small
tensors and do not execute the CUDA HybridEP or TileLang kernels.

**Full GPU parity and full-model training validation of this migration remain
pending.** The following measurements came from the original implementation
before migration and are not a pass for this branch.

At original commit `0d5919a5a506dd35dc4f55769410feb98ae854b8`, the continuous
40-layer, 4,096-token, full-129,280-vocabulary comparison used the unchanged
official TP1 oracle and native EP4/Engram-owner4 with `torch_mm`, HybridEP and
TileLang. Native rank 0 received the valid sample; the other ranks participated
with masked inputs. Native execution streamed one production block at a time,
carrying its own residual streams, mHC coefficients and CSA2 state; shared
embedding, norm and head weights were replicated.

| Metric | Original `torch_mm` streamed-40 result |
|---|---:|
| Mean KL | 0.0017376351 |
| P95 KL | 0.0005163713 |
| Maximum KL | 0.8451328874 |
| Logit cosine | 0.9946222187 |
| Top-1 agreement | 99.58496094% |

The fixed mean-KL and cosine gates failed. This was a streamed
full-backbone forward comparison, not full-resident training or convergence.
The reference used BF16 weight compute while retaining its released KV/index
quantization math; it was not a quantized-GEMM performance comparison.

Earlier `torch_linear` comparisons achieved exact logits in their recorded
scopes. A separate reduced-width 40-layer EP2/expert-FSDP2 experiment at
original commit `69a192aaa183efc6aae317255f6d8f0a3a906499` used workspace-only
wrappers to verify real interruption and fresh-process restore of model,
TE optimizer including int16 masters, RNG, loader and progress. These results
do not replace migration validation. Earlier trajectory probes were debugging
evidence, not final proof of Engram gradient scaling; TileLang continuation
was not bitwise reproducible. No successful full-resident 100-update run or
full-model convergence result is claimed.
