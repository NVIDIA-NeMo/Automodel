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
next attention. End-to-end full-width 100-step results must be recorded
separately; small models and synthetic operator prewarm are not that evidence.

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
size 5120, source layers 0/2/20, mHC, TE norms and six local experts. It does
not load the pretrained checkpoint or replace the required full-width
100-step training comparison.
