# V4.1 fused indexer validation

The TileLang attention backend now also fuses the frozen indexer's scoring in
`deepseek_v41/indexer.py`. Query/key dot products, ReLU, head weighting and head
reduction stay inside each tile; only `[batch, local_sequence, global_keys]`
scores are written to device memory. Existing visibility metadata and
document-local top-k/candidate selection are preserved. Eager/SDPA retain the
original PyTorch scoring. No DSV4 implementation files changed.

The kernel accumulates in FP32 and preserves the released BF16 rounding after
the dot product, weighted head product and final head reduction. It accepts the
existing visibility mask for global CP causality, valid compression groups and
packed document isolation. It removes the large per-head scoring intermediates;
it does not remove the indexer's necessary visibility mask or top-k output.

## Replacement correctness

Validated on GB200 interactive allocations 7117760 and 7117995. The second
allocation replaced the first only after its 30-minute limit expired.

- 44 existing attention/packing/CP unit tests passed.
- 5 GPU functional cases passed. Real NCCL CP2 tests cover six attention roles,
  including compressed KV/index-key reuse, candidate generation/reindexing,
  odd-length documents, padding and a document crossing the CP boundary, with
  per-block activation checkpointing both enabled and disabled.
- Selected indices, state, outputs, input gradients and non-sink parameter
  gradients matched the original scoring exactly. FP32 sink gradients passed
  the existing atomic-reduction tolerance (atol 2e-6, rtol 2e-5).
- Ruff formatting and checks passed.

The real pretrained first four layers used 4K input, full hidden width/vocabulary,
full owner-sharded Engram, native DCP loading, EP4 and TileLang attention.
All four ranks produced logits **bitwise identical** to the same-model original
scoring. The rank-0 logits also matched the frozen historical native artifact
bitwise. This is a forward comparison, not full-model training validation.

| Comparison | Mean KL | p95 KL | Logit cosine | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Fused vs original scoring | 0 | 0 | 1 | 100% |
| Original vs official reference | 0.0062182555 | 0.0357103683 | 0.999393508 | 94.5068359% |
| Fused vs official reference | 0.0062182555 | 0.0357103683 | 0.999393508 | 94.5068359% |

The pre-existing strict official-reference gates still fail (mean KL <= 0.001,
p95 KL <= 0.002 and top-1 >= 98%). This replacement introduces no new deviation;
the unchanged failures are not relabeled as passing.

The first4 harness encountered a loading-time allocator stall. Its successful
run deferred explicit `empty_cache()` calls only inside the loader scope, then
restored normal allocator behavior before both forwards. A CPU rendezvous
preceded DCP. Native checkpoint conversion, dtype handling, source-value audits
and forward arithmetic were retained. Earlier interrupted/stale-harness attempts
and their stack traces are preserved in the experiment directory.

## Isolated 32K scoring

Synthetic quantized/dequantized MXFP4 BF16 inputs: B=1, local Q=4096, global
K=32768, H=32, D=128, ratio-1 CP8 rank-7 visibility. Nine CUDA-event samples:

| Scoring | Median time | Additional peak allocated memory |
| --- | ---: | ---: |
| Original PyTorch | 14.875 ms | 24.00 GiB |
| Fused TileLang | 5.759 ms | 0.25 GiB |

Visibility and top-512 indices matched exactly. Five finite scores differed;
maximum absolute difference was 0.001953125, within the explicit one-BF16-ULP
relative gate. Different FP32 head-reduction order can affect final rounding.
Do not describe all scores as bitwise identical. Partial query/key tile tests
and a separate random-visibility 32K probe also passed.

These are scoring measurements, excluding projection, communication, top-k and
full-model training. Packed CP8 32K training has not been rerun at this checkpoint.

## Evidence

All artifacts remain under
`/lustre/fsw/portfolios/coreai/users/huiyingl/ds41/logs/`:

- `fused_indexer_first4_ab_7117995.json` and its rank/load/storage/logit artifacts
- `fused_indexer_functional_7117760.log`, `fused_indexer_unit_7117760.log`
- `fused_indexer_probe.json`, `fused_indexer_benchmark_7117760.json`
- `fused_indexer_first4_validation_attempts.json`
- `fused_indexer_ruff_7117760.log`, `fused_indexer_test_ruff_7117995.log`
