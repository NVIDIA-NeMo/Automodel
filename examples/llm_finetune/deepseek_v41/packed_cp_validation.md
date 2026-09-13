# DeepSeek V4.1 packed context parallelism

The training input has one representation: token rows [batch, sequence] from
packed_sequence_thd_collater, with seq_lens for real document lengths and
seq_lens_padded for physical spans. It does not require flattened cu_seqlens
or a second packing implementation.

pack_dataset(..., cp_size=1, pad_to_multiple_of=2) counts compression alignment
inside the fixed 4096-token budget. The model-owned sharder prepares both CP1
and CP2, then CP2 keeps contiguous halves. Labels are shifted within each source
sample before packing and are only moved by the sharder. A valid final EOS
target remains supervised.

Every document has local RoPE positions and an explicit document ID. Window
attention, compressed-key selection, and Engram history all respect those IDs.
Compression groups start at document boundaries; incomplete groups are invalid.
Top-k and candidate-block selection use each document's own key interval so
masked rows from neighboring documents cannot change tie selection. Source
state carries global compressed keys and document IDs through encoder reuse
and the encoder-to-decoder handoff. Reindex layers retain the source candidates.
Checkpointing receives immutable per-block state.

The direct model API also accepts seq_lens/seq_lens_padded and restores caller
token coordinates. Its HF-style labels are unshifted and receive a document
boundary mask. Prepared sharded batches use the recipe's already-shifted labels
and external loss. Image packing and preflattened cu_seqlens input are rejected.

## Correctness checks

tests/unit_tests/models/deepseek_v41/test_packing.py covers:

- Packed versus independently unpadded documents, including lengths 1, 3, 9 and
  17. It compares logits, every captured layer, supervised-token mean loss,
  every parameter gradient, clipping norm, and an SGD update. The strict FP32
  model preserves all 40 layers, sources 2/8/14/20, decoder reindex layers, and
  Engram layers 1/14 while reducing parameter dimensions.
- Changing one document does not affect other documents. A random upstream
  gradient restricted to one document produces exactly zero input gradients
  in the others. Engram hashes match independent documents exactly.
- Two real Gloo ranks compare CP1 and CP2 logits and parameter gradients with
  activation checkpointing enabled and disabled. Cases include a document
  crossing a rank boundary and a rank containing only padding.
- Production packer/collator token order, reset positions, unchanged shifted
  targets, bounded physical pack size, and CP1 preparation without a mesh.

The full pretrained training recipe is deepseek_v41_flash_tulu3_packed_cp.yaml.
Its CP1 and CP2 comparison requires fresh identical packed Tulu3 data, the same
checkpoint, seed, GBS64, LBS1, EP64, learning-rate schedule and FusedAdam settings.
Set defer_fsdp_grad_sync=false for each-microbatch gradient synchronization.
Both runs perform 100 optimizer steps with sequence length 4096 and native
training/validation metrics. Per-step pack hashes, supervised-token counts,
initial parameter samples and completion receipts accompany the comparison.

GPU BF16/FP8/NVFP4 arithmetic is assessed separately from strict FP32 semantic
tests. A 6-layer small GPU CP1/CP2 test passed loss, gradient and update gates
but exceeded the predeclared 1% logit relative-RMSE gate (1.63%). This is retained
as a failed strict numerical check, not silently reclassified. Layer tracing
found identical layer-0 attention inputs/outputs and differences before the
next attention. The full-width 100-step results below provide separate training
evidence; the failed strict small-model numerical check remains recorded.

## Cluster validation recorded before the full-width runs

On GB200, the production recipe with a reduced-width 40-layer model completed
four CP1 and four CP2 optimizer updates, including validations after updates
two and four. It used the complete source/reuse/reindex schedule, both Engrams,
EP4, FSDP2, TileLang, TE and FusedAdam. All pack hashes, initial parameter
samples, supervised-token denominators, labels and learning rates matched.

Maximum training-loss difference was 0.000174523; maximum validation-loss
difference was 0.000252962; maximum relative gradient-norm difference was
0.000329986. The 232 model/data unit tests and 45 gradient-clipping tests passed.

A separate six-layer diagnostic replayed captured CP1 attention inputs on CP2.
All valid attention outputs, compressed KV and selected key sets were exactly
equal at every layer, with identical initial parameters on all four ranks.
This isolates attention from the upstream numerical differences in that case;
it does not change the failed strict end-to-end logit threshold above.

Real-width synthetic operator prewarm passed for CP1 and CP2, covering hidden
size 5120, representative attention layers 0/2/20 (layer 0 is window-only),
mHC, TE norms and six local experts. It does
not load the pretrained checkpoint or replace the required full-width
100-step training comparison.

## Full pretrained packed CP1 versus CP2: 100 steps

Both runs completed on 2026-09-12 from implementation commit
ac4ad42058611da92cc78cc648d33bec0b9a2a76 and the same pretrained checkpoint
df42c109f1defefcbfcedbe7d905718a12266e40. This is the full 40-layer text model:
hidden size 5120, vocabulary 129280, all encoder/decoder source and reuse paths,
and Engram layers 1/14. Each run used 64 GB200 GPUs on 16 nodes, EP64, Engram
owner64, GBS64, LBS1, per-block activation checkpointing, and
defer_fsdp_grad_sync=false. Global sequence length was 4096; local sequence
length was 4096 for CP1 and 2048 for CP2.

| Metric | CP1 | CP2 |
| --- | ---: | ---: |
| Slurm job | 7101485 | 7101490 |
| Completed optimizer updates | 100 | 100 |
| Job elapsed time | 15m 38s | 20m 13s |
| First training loss (step 0) | 0.843975544 | 0.843463540 |
| Final training loss (step 99) | 0.570848286 | 0.571441889 |
| Validation loss (step 49) | 0.552888751 | 0.553111196 |
| Validation loss (step 99) | 0.546019495 | 0.546463490 |
| Peak logged allocated memory (GiB) | 119.520710 | 132.086376 |

Memory is the maximum of the native reference-GPU metric over logged updates,
not the maximum across all 64 GPUs. Job elapsed time includes setup and reporting.

All predeclared training-parity gates passed. Maximum absolute training-loss
difference was 0.001736820 (limit 0.01), mean absolute difference 0.000554583
(limit 0.002), and maximum validation-loss difference 0.000443995 (limit 0.01).
The 100 per-step input audits, labels, learning rates and supervised-token
denominators matched exactly. Initial parameter samples matched on all 64
ranks, and both jobs produced all 16 node-completion receipts. These are sampled
initial-state checks, not a bitwise comparison of every parameter.

The separate 5% maximum relative gradient-norm diagnostic **failed**:
maximum 27.1212%, mean 5.9180%, median 3.7390%, and 95th percentile 18.6313%.
CP2/CP1 norm ratios ranged from 0.739402 to 1.271212, with mean 1.006169.
The bidirectional differences do not show a fixed CP-count scaling factor.
They remain a numerical limitation; passing the loss/validation gates does not
establish exact BF16 gradient parity. The diagnostic was separate from the
training-parity gates before these runs, and its threshold was not changed.

Native W&B history was read back for both runs: 100 rows, validation at steps
49/99, every original payload value exact, and both runs finished. The runs
can be overlaid by their native training step:

- [Packed CP1](https://wandb.ai/Nemo-automodel/huiyingl_workspace/runs/ds41packedcp17101485)
- [Packed CP2](https://wandb.ai/Nemo-automodel/huiyingl_workspace/runs/ds41packedcp27101490)

Cluster artifacts are under
/lustre/fsw/portfolios/coreai/users/huiyingl/ds41/logs/packed_single_7101490/:
comparison.json, comparison.csv, comparison.png, comparison.pdf, and
gradient_norm_review.json. Each run directory contains
wandb_native_verification.json alongside its original training/validation logs.

The strict FP32 semantic tests, full pretrained training gates, and reviewed
numerical diagnostics support proceeding to the requested packed CP8 32K
experiment without changing the model implementation. That separate run has
no CP1 32K baseline and must not be described as a 32K parity result.
