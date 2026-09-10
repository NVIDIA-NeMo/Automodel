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
- [attention.py](attention.py) implements CSA2 attention, compression, index
  reuse and candidate masks, reusing the shared DeepSeek V4 sparse kernels.
- [quantization.py](quantization.py) implements the KV and index quantization
  boundaries; [layers.py](layers.py) provides RMSNorm and mHC.
- [config.py](config.py) defines the nested text and vision configurations.
- [engram.py](engram.py) supplies V4.1 hashes and gates; its embedding reuses
  the Qwen3.8 Flash Next owner-sharded table and autograd communication.
- [state_dict_adapter.py](state_dict_adapter.py) maps the released layouts and
  supports bounded direct initialization from safetensors.
- [processing.py](processing.py) and [vision.py](vision.py) provide ordinary
  text/image processing and the reused DeepSeek V4 vision modules.

Model construction defaults to `torch_linear` experts with `hybridep`
dispatch, `torch` linears, `torch_fp32` normalization and `tilelang` attention.
There is no automatic CPU backend substitution. Both validation recipes
explicitly select `torch_mm` experts with HybridEP and TileLang. The
language-model head computes and returns FP32 logits without rounding them
back to BF16.
Shared MoE retains the FP32 combine contract: individual routed contributions
return to their source ranks, accumulate in expert order, and combine with
the shared expert before the final BF16 cast. This requires additional
communication compared with the usual BF16 HybridEP combine.

The default MoE `gate_bias_update_factor` is zero, preserving the released
router correction biases during fine-tuning.

`torch_linear` uses separate eager
projections with the same shared parameter storage and dispatcher. It is not
the selected validation backend. Its historical bitwise reference results
must not be attributed to `torch_mm`, whose grouped GEMMs and fused pointwise
operations have different rounding boundaries.

Released FP8 dense matrices, packed FP4 experts and rowwise FP8 Engram tables
initialize floating-point training parameters. The forward always retains
FP8 window KV, NVFP4 compressed KV and MXFP4 index query/key quantize/dequantize
boundaries. Training uses straight-through gradients; there is no QAT toggle.
Indexer parameters remain frozen because discrete top-k does not provide a
language-model gradient; indexer distillation is outside this implementation.

## Training recipes and ownership

[The EP64 recipe](../../../../examples/llm_finetune/deepseek_v41/deepseek_v41_flash_hellaswag_ep64_16nodes.yaml)
targets 16 nodes with four GB200 GPUs each in one verified NVLink domain:
WORLD=64, EP=64, 64 Engram owners, and TP1/PP1/CP1. Dense parameters use FSDP2;
experts have no additional FSDP shard axis. Global batch 64 and local batch 1
give one microbatch per optimizer update. AC and reshard-after-forward are
enabled. The schedule requests 100 updates with validation after updates 50
and 100; checkpoint saving is disabled. TE FusedAdam uses BF16 moments and
FP32 master weights stored as int16 remainders. Saving is disabled because TE
optimizer export expands those moments to FP32 on the GPU. The recipe enables
online W&B and requests a one-hour allocation.

[The EP128 recipe](../../../../examples/llm_finetune/deepseek_v41/deepseek_v41_flash_hellaswag_ep128.yaml)
targets 32 nodes with four GB200 GPUs each: WORLD=128, EP=128, 128 Engram
owners, TP1/PP1/CP1, global batch 128 and local batch 1. It uses FP32 optimizer
moments with int16 master remainders and enables checkpoint saving after
updates 50 and 100. Both recipes use the full 40-layer model without gradient
accumulation or an additional expert FSDP shard axis.

Both Engram tables are trainable. Configured `engram_layer_ids` determine
which layers contain a table; there is no freezing switch. Each table is a
registered global row-sharded DTensor, excluded from FSDP all-gathers. The
distributed model defaults its owner mesh to WORLD, matching the dense FSDP
ranks and ordering. A standalone Engram module without a process group keeps
a local table. Owner gradients receive the owner divisor once before global
clipping. Disabling Engram changes the model function.

Both recipes preserve FP32 mHC coefficients across FSDP boundaries with
`output_dtype: null` and `cast_forward_inputs: false`. This is fine-tuning,
not a reproduction of the report's Engram pretraining optimizer.

For the EP128 recipe spanning partial NVL72 domains, export
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4` on every node so HybridEP uses
equal node-local groups. The EP64 recipe instead requires verified placement
within one 64-GPU NVLink domain; do not inherit the four-rank override.

Set `DS41_CHECKPOINT` to the shared local snapshot of pinned revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, using the same directory for parity
and training. Pass both overrides to the recipe launcher:

```bash
--model.config.pretrained_model_name_or_path="$DS41_CHECKPOINT" \
--model.config.name_or_path="$DS41_CHECKPOINT"
```

Keep outputs outside the source checkout. The launcher owns rank placement,
communication setup and reusable compilation caches. Neither recipe
establishes memory fit: the earlier full-resident 16-node run with FP32
moments completed update 0 and then OOMed during the next backward, despite
AC and resharding. The BF16-moment EP64 run and the EP128 run remain pending.

## Checkpoint APIs

Retain the released config's quantization metadata when initializing from
the original checkpoint: the NeMo model loader uses it to infer base-weight
dequantization. Both recipes also set
`checkpoint.dequantize_base_checkpoint: true`. Trained parameters are saved
without quantization; the adapter exports released names and retains required
FP32 weights. Engram save/load trims logical rows and restores only allocation
padding. The adapter removes original quantization metadata from decoded
floating-point checkpoint configs; quantized re-export is not established.

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

Configuration uses `DeepseekV41Config` with a nested `text_config`
(`DeepseekV41TextConfig`) and `vision_config`. Text accepts unpacked
`input_ids [batch, sequence]` with right padding and zero-based positions;
packed THD is unsupported. Enabled Engram requires a real fast tokenizer at
construction. Configuration-owned tokenizer loading uses the resolved
checkpoint revision, or callers can supply a tokenizer explicitly.
Materialization restores the tokenizer-derived map and deterministic hash
buffers after meta initialization. The standalone `labels` API accepts
unpacked `[batch, sequence]` labels. TP, PP, CP, inference KV caches, bounded
decoder replay and DSpark training are not implemented. Million-token
training has not been validated.

The default configuration retains the released vision tower. Both HellaSwag
recipes explicitly disable it. `bias_vl` remains present even in text-only gates.
`DeepseekV41Processor` provides standard text/image conversations and
save/reload. Image batches require unpacked `input_ids [batch, sequence]`,
`pixel_values [all_patches, 3, patch_size, patch_size]`,
`image_grid_hws [images, 2]`, and `vision_token_types [batch, sequence]`.
Packed images are rejected. Use the official encoder for specialized tool or
reasoning formatting before passing rendered text to the processor.

## Validation provenance

**Final CPU/Gloo tests, GPU parity and full-model training validation of this
migration remain pending.** The following measurements came from the original
implementation before migration and are not a pass for this branch. See the
[model coverage page](../../../../docs/model-coverage/llm/deepseek-ai/deepseek-v41-flash.mdx)
for the recipe and validation scope.

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
