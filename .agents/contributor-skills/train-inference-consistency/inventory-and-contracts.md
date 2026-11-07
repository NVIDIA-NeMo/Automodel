<!-- Adapted from NVIDIA-NeMo/labs-molt, .agents/skills/train-inference-consistency/ (contracts.md,
training-side.md) at commit 745f0966c2c163e3a13ac1b72fc5f12deceb70b3 (Apache-2.0); see SKILL.md. -->

# Module inventory and numerical contracts

What must be equal, operator by operator, for the trainer's recomputed logprob to be the bytes the
engine returned. The inventory is the first deliverable of an aligned forward and the table every
later step reads; the contracts say what "equal" means per row.

## Module inventory

Day one, before any code: one row per module, every column filled, `unknown` in blanks, never
"should exist".

| Column | Question to answer |
|---|---|
| Module | One row per fusion boundary as the engine executes it (residual-add + norm is one op), not per HuggingFace class |
| Engine kernel, prefill / decode | The function the engine actually calls, not the class name: attention backend, custom op vs compiled path, the branch under the engine's batch-invariant switch; two entries when prefill and decode differ |
| Same-math training implementation | Code with autograd that computes the same function: the model's native module, torch, a training kernel library. Only-rounding differences are same math; a different clamp, norm variant (`1 + w`) or router formula is not |
| Quantization rounding | Where the forward rounds (weights, activations, KV cache) and with which scale rule; bf16 route: none |
| Discrete choice | Top-k expert ids, kept token set, tie-break, ordering. Ids are fixed and not differentiated; the continuous score needs a derivative |
| Backward exists | In which library, and whether its saved intermediates (LSE, rstd, token permutation) line up with what the engine forward produces |
| Verdict | Forward: already equal / swap trainer op / needs an engine change. Backward: reuse / glue / new ([rules](aligned-forward.md#backward-reuse-glue-or-new)) |

A module is anything producing a value on the logprob path: embedding, every norm (q/k and final),
position encoding, attention, state layers, router, routed and shared experts, activation, lm_head,
every encoder, the trainer's scorer and the engine's sampler logprob kernel. Parts that do not take
part in training (an MTP head, a draft model, a frozen vision tower) get a row with mapping and
fact: load and refit skip them by explicit list and the log says "skipped", not "no error". A
frozen part both sides run is on the value path: identical kernels on both sides or embeddings
shipped from the engine; freezing does not make bytes equal.

Read the engine's model file, not the HuggingFace modeling file, in this order:

1. Layer types and order; residual and norm positions; position encoding; attention visibility
   (sliding window, sinks, grouped heads); router (softmax or sigmoid, bias, renormalisation,
   grouped top-k, tie-break); activation; shared experts; state layers; special embeddings and
   encoders.
2. The kernel each layer calls and its quantization variant; attention backend and KV dtype.
3. The weight loader: name mapping, fused order, derived scales, post-load layout changes.
4. Branches under the engine's batch-invariant switch; parameters and buffers confirmed from the
   checkpoint and config.

Model-specific math is written from the inference source, not from the paper or the HuggingFace
file. Sizes sharing one engine model file share one port; a multimodal variant is validated per
task.

## Object of comparison

- What: for every valid response token `y[t]`, `log p(y[t] | prompt, y[:t])`, the engine's
  selected-token logprob vs the trainer's recompute. The first response token comes from the logits
  at the last prompt position.
- Under which weights: the same weight version, token history and mask, before that version's
  update.
- As what bytes: the raw fp32 logprob compared in the original dtype (`view(torch.int32)`; `int16`
  for a 16-bit dtype). A dtype difference is a contract violation, never cast away.
- What fails: NaN or Inf at a valid position fails even with identical bytes; zero valid tokens is
  `inconclusive`; a shift by one position is a comparison bug, not a tolerance.
- Processing: the engine applies temperature in its logprob mode; the trainer divides by the same
  temperature in fp32 after `.float()`. Identical processing requires `top_p = 1` and no top-k; any
  truncation makes the kept token set a discrete decision compared before the renormalised logprob
  ([discrete decisions](#discrete-decisions)). Equal sampled-token logprobs do not prove equal
  logits.

## Decomposition

Each term guards one root cause, a different order of operations or a different computation path,
and one check covers it. `mismatch = 0` means both sides compute the same thing, not that training
is good.

| Term | Meaning | Checked by |
|---|---|---|
| Kernel determinism | same kernel, same input, same bytes in every run and process | one-operator test: repeat, cross-process, graph vs eager; engine A/A |
| Batch independence | a request's bytes do not depend on batch-mates, slot, or the engine's merge and split; trainer: micro-batch, packing, per-expert token counts | engine alone vs with neighbours; trainer packing A/A then A/C |
| Phase consistency | one token gets the same bytes in full prefill, chunked prefill and decode | engine cells (owned by the engine and the loop) |
| Cross-GPU consistency | reductions (all-reduce, reduce-scatter, expert combine) fixed in order and dtype; a degenerate layout (TP = 1) matches a multi-rank side only by simulating its tree | [layout rules](aligned-forward.md#layout-length-and-world-size) |
| Same numerical contract | same accumulation dtype, rounding and kernel arithmetic | inventory, [per-operator contract](#per-operator-contract), forward parity |
| Same discrete choices | top-k ties and order, routing, kept token set | forward parity with the set-flip rate reported |
| Correct backward | the forward terms say nothing about gradients | [gradient contract](aligned-forward.md#gradient-contract-and-validation) |

## Per-operator contract

"Same math" is not "same bytes": the same source executes differently through autotune, compile
specialisation, a different caller or GPU. Verify the executed kernel and its arguments, not the
Python function. Semantics first: a feature inside the kernel on one side and outside or missing
on the other (sink, sliding window, `1 + w` norm, q/k-norm position, RoPE variant) never goes away
with precision settings; enable one feature at a time on the proxy.

| Operator class | Must match |
|---|---|
| Dense GEMM (q/k/v/o, gate/up/down, lm_head) | one kernel and one split-K / tile plan for every M; fp32 accumulation; the epilogue |
| Attention | split constants, reduction order, scale and LSE, in-kernel features, KV page layout ([attention](#attention-contract)) |
| Norm (RMSNorm, fused add + norm) | same kernel and block geometry at every M; fp32 variance; eps position; `1 + w` variant; the `(hidden, residual)` pair |
| Activation (+ quant) | `silu(gate) * up` in fp32, rounded once; `cat(gate, up)` order; same approximation |
| Router / top-k | fp32 gate GEMM at every M; fp32 softmax or sigmoid, bias, renormalisation, grouped top-k; stable tie-break |
| Expert GEMM | per-row result independent of layout and position; one tile config; no split-K along K |
| Combine | fp32 accumulation on the owning rank in top-k slot order, independent of EP degree |
| KV read path | same bytes through the same dequantisation in prefill, chunked prefill, decode and trainer |
| State layers (scan, conv) | chunk grid, boundary-state dtype, gate position, conv window, `cu_seqlens` ([state layers](#state-layer-contract)) |
| lm_head + scorer | head GEMM as a dense GEMM; logits dtype at hand-over; fp32 divide by temperature; the trainer's scorer and the engine's sampler kernel inventoried as two kernels |
| Embedding | table bytes as in the checkpoint; exact gather; any scale in the same dtype and position |
| RoPE | cos/sin table device and dtype; neox vs interleaved; partial dims; position relative to q/k norm |

Details per row:

- Dense GEMM: no shape-dependent kernel switch (small-M branch, library heuristic); tile M
  independent of the K reduction; fixed cuBLAS workspace; bias, fused activation and the rounding
  point of the output. A library that picks its tactic from M (autotune, a managed cache) must be
  pinned to one tactic or padded to one shape class on both sides.
- Norm: a kernel that switches block size with the token count must be pinned, or tested at the
  switch (255 / 256 / 257 rows alone vs batched); weight dtype; the fused op returns the new
  residual; norm + activation + quant fusion only if identical on both sides.
- Activation: rounding `silu` first can differ by one ulp, which a later low-precision quant turns
  into a code flip; `erf` vs `tanh` GELU; if quantized: scale grouping, amax, rounding, clamp,
  aggregation order.
- Router: autocast off; tie-break by request-local index; output order fixed; invalid or padding
  index sentinel; routing weights re-gathered from the fp32 scores; a fused grouped top-k is called,
  not re-derived in torch. Token ids that route differently by construction (special or sentinel
  ids the engine treats separately) are part of the contract: both sides must apply the same rule
  to the same id range.
- Expert GEMM and combine: masked vs contiguous layout, capacity padding; same input rounding
  across dispatch modes; padding rows never alter valid rows; routing-weight dtype and
  multiplication order; shared-expert add position.
- KV read path: a dtype label is not alignment; page size and block-table semantics; prefix-cache
  hit equals miss; cache reset after every refit.
- lm_head + scorer: no unconditional `.float()` on one side only; `.float()` then divide by
  temperature even at temperature 1; fp32 log-softmax and its reduction order; vocab-shard
  reduction order under TP; the engine's sampler logprob kernel and a `gather - logsumexp` in torch
  are different kernels. Call the engine's, or prove equal bytes.
- Embedding and RoPE: a tied head shares the table bytes; vision or audio embeddings come from the
  engine, not a recompute; CPU and GPU round cos/sin differently, so the table is built where the
  engine builds it; positions identical under packing.

### Attention contract

- Fixed split and segment constants over keys, independent of batch and length; a length ceiling
  raises, never re-splits silently.
- QK, online softmax, PV and the merge order of partials; softmax scale and where it is applied;
  LSE base (ln vs log2) and dtype; fp32 partial accumulators.
- Mask, sink, sliding window and grouped-head mapping inside the kernel on both sides.
- KV page layout as the engine resolves it (read the resolved value in the engine log).
- Algebraic rewrites (weight absorption, a folded softmax scale) are different rounding paths, not
  the same kernel. Sparse or indexed attention adds the index selection as a discrete decision:
  compare the selected key set per query before the attention output.

### State-layer contract

- Chunk grid: chunk size and chunk boundaries at sequence boundaries; a chunked scan is
  split-invariant only at its aligned splits, so a shared chunk grid is a multiple of that
  alignment.
- Boundary-state dtype and where it is rounded; gate applied inside the fp32 scan vs after
  rounding; conv window alignment; both sides share the tail offset of the conv state.
- Every packed sequence consumes `cu_seqlens`; the scan restarts at every sequence start, not at
  the row offset.
- Chunked prefill and per-token decode are two rounding paths; the model is supportable only when
  the engine proves both give the same bytes. That proof belongs to the engine; the trainer walks
  the prefill grid.

## Discrete decisions

Discrete decisions are compared before any last-bit error, because a different selection changes
every downstream gather and reduction order.

- Router: after the logits, compare the selected set, its order and the tie-break. The tie-break
  defines the index order for equal scores (request-local index, never the global batch offset) and
  the invalid or padding index.
- Top-k / top-p sampling: a last-bit logit difference moves tokens across the cutoff and the
  processed-logprob denominator jumps with them; compare the kept token set before the processed
  logprob. Forcing the token set is legal only if the report says it masks the set difference and
  counts the set-flip rate separately.
- MoE: dispatch, expert capacity, grouped-GEMM geometry, quantization and combine are fixed or
  verified, not assumed. Replaying the engine's expert ids in the trainer is a declared contract
  that takes natural routing off the compared path; report the rate it masks. Forcing ids ad hoc
  to pass a comparison isolates the downstream (a labelled diagnostic); it never fixes a router
  difference.
- Sparse attention indexers: the selected key set per query is a discrete decision with the same
  rules (tie-break, invalid sentinel, K larger than the valid count, empty rows, tail blocks).

## Shape, process and state boundaries

For one victim sequence build the cells of the measurement matrix: batch 1 vs with neighbours; a
different physical slot; lengths just before and after every boundary (chunk, page, segment,
capture bucket, pad multiple); a fresh process (autotune and compile caches); graph vs eager.
Launches differing when only M changes is not a numerical finding; measure the output. No
difference covers only the shapes that ran. An oracle whose autotune picks its blocking by timing
flakes an A/A: pin its config.

History-dependent operators (conv, scan, attention over a cache): localise over the full prompt
batch, first differing token per layer, then the operators inside that layer in order
([bisect](debugging.md#bisect-and-first-divergence)). A single-position probe narrows candidates,
but an upstream history difference looks like an error in the probed operator.

## Weights and precision chain

bf16 route (bf16 checkpoint, bf16 engine): both sides read the same checkpoint bytes and the weight
contract is three facts.

| Fact | Rule |
|---|---|
| fp32-kept parameters | Parameters the trainer computes in fp32 (router gate, norms the class keeps fp32) are fp32 on the engine too or rounded identically before use. FSDP2's `MixedPrecisionPolicy(param_dtype=bf16)` downcasts them unless excluded, and a later `.float()` cannot restore them: measure the dtype inside the forward. The model-wide cast goes through `cast_model_to_dtype`, which honours `_keep_in_fp32_modules`; a raw `self.to(dtype)` does not |
| Refit dtype | The refit sends the fp32 master and the engine rounds it to bf16 on load, the same cast FSDP2 applies for the training forward: equal bytes for plainly copied parameters, not for derived ones (a `-exp(A_log)`, a calibration scale). Prove the arriving bytes by hashing the same tensor on both sides after load |
| Mapping | Checked by value and semantics, not shape: fused QKV and gate/up order, expert numbering, tied weights, buffers, derived state; a written reason for every missing or extra item |

Four-stage chain, one conversion owner per tensor family, converted exactly once:

```
A master (fp32 + declared exceptions)
  -> B weights the training forward sees
  -> C exported bytes
  -> D engine bytes after load and post-load processing
```

- A: dtype recorded per parameter and buffer.
- B: bf16, or under a quantized route the value the engine kernel multiplies; equal to
  `dequant(C)` bitwise.
- C: computed on the full logical tensor (`full_tensor()` then convert, never per shard: block
  boundaries and fusion-group scales cross shards), idempotent on the same master.
- D: post-load only reorders; calibration scales pass through unchanged.

Dtype boundaries to record per model:

| Boundary | In this repository | Must check |
|---|---|---|
| Source weights and conversion | HuggingFace loader, initialisation dtype, the model's state dict adapter | Same revision; logical value, scale and layout match, not only the target dtype |
| Parameter storage | fp32 masters; class-level fp32 exceptions | An fp32 master does not mean an fp32 forward |
| Forward parameters | `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=fp32)`, `cast_forward_inputs` | Measure inside the forward, under the real autocast state (the aligned path runs none) |
| Operator computation and output | `BackendConfig` kernels on the default path; the engine kernels on the aligned path | An fp32 output does not prove an fp32 product; a bf16 output does not prove bf16 accumulation |
| Gradient and optimizer | `reduce_dtype=fp32`, gradient accumulation, GPU or offloaded optimizer | fp32 reduction does not restore values rounded in the forward; CPU and GPU optimizer steps are not bit-identical |
| Export and serving | DTensor `full_tensor()`, HuggingFace naming, dtype-faithful refit, engine cast on load | Compare values on the same logical layout; the first refit after a restore sends the pre-save bytes |

### Quantized checkpoints

A checkpoint released in a quantized format (FP8 block scales, MXFP8, NVFP4) makes the quantized
bytes the deployed value, and the engine kernel that consumes them defines the numbers.

- The aligned forward runs the engine's quantized kernel on the checkpoint's native bytes. It never
  dequantises to bf16 and imitates the kernel with a dense GEMM: that is a different rounding path
  and the first thing a byte comparison catches.
- Trainable groups keep fp32 masters alongside the native bytes. After every optimizer step the
  deployed bytes are `requantize(master)` with the exporter's own function (range, axis, block
  size, scale dtype and rounding, fusion-group scale), and `dequant(requantize(master))` is what
  the forward multiplied. The STE is written explicitly (`q.detach() + (x - x.detach())`); a
  low-precision cast on the autograd path underflows small gradients to 0.
- Groups that are not trained (for example routed experts whose bytes stay as released) are a
  declared contract: their bytes never change, the refit never sends them, and the backward still
  propagates `dX` through them with their dequantised value. The record names the trainable set.
- A format name is not a rule: block vs per-tensor scale, scale dtype, rounding, static vs dynamic
  activation scale come from the checkpoint's quantization config layer by layer. Activation and KV
  scales from calibration pass through unchanged.
- bf16 training with a quantized rollout is not quantization-aware training, and a difference that
  survives a precision change was never quantization: rule out routing, mask, tokens and layout
  first.

## Shared forward and backward boundary

The visible forward calls the engine kernel for the value (bitwise with inference) and saves inputs
without building a graph; the native backward is the training stack's same-math implementation and
gives the VJP ([reuse, glue or new](aligned-forward.md#backward-reuse-glue-or-new)). It is not
bitwise; it satisfies a written gradient contract (STE form, which weight `dX` uses, discrete ids
fixed and not differentiated, VJP collectives, saved state) and checks that the master has not
changed since the forward, so a recompute cannot mix weight versions.

- Never detach the inference output and train on it. The two legal forms are a
  `torch.autograd.Function` that recomputes the same math in `backward`, or
  `visible + (graph - graph.detach())` in eager with `graph` the same math in PyTorch.
- Forward parity, a finite gradient for every trainable parameter, and the gradient contract are
  verified separately; operator-level VJP passes do not prove the whole-model gradient, and a
  finite gradient is not learning.

## Changing the arithmetic baseline

When both sides' arithmetic changes on purpose (a new kernel, new segment or tile constants, another
reduction tree): keep the old baseline, build a new frozen rollout under the new arithmetic, state
that old outputs are no longer guaranteed bitwise unchanged, and re-run quality and performance. Old
numbers do not transfer. Shared constants change together, are re-proved bitwise, and are accepted
on the end-to-end comparison only.
