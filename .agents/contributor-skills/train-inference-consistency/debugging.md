<!-- Adapted from NVIDIA-NeMo/labs-molt, .agents/skills/train-inference-consistency/ (debugging.md,
contracts.md) at commit 745f0966c2c163e3a13ac1b72fc5f12deceb70b3 (Apache-2.0); see SKILL.md. -->

# Debugging a mismatch

Find the earliest divergence, not the largest; isolate the side before bisecting; classify before
touching precision. One variable per experiment; a path that did not execute is `inconclusive`
([evidence labels](verification.md#evidence-labels)).

## Triage

1. Was the aligned path built at all? Check the resolved config field and the instantiated decoder
   class in the log; a default decoder compared with the engine reads the default path's
   mismatch, which is the expected baseline, not a fault.
2. Did the process-setup function run before the first engine import and CUDA GEMM in this process?
   The engine reads some settings once; a late call measures another configuration.
3. Is every operator of the model supported by the aligned path? Modules whose inventory verdict is
   "needs an engine change" or "unknown" are not aligned; their mismatch is known, not news.
4. Was the comparison made within one weight version (before that version's update) and on the
   tokens the engine sampled, not re-tokenised text?

Only then the controls below.

## Symptom to first step

| Symptom | Do first |
|---|---|
| Rollout and recompute differ at the initial weights | [Isolate the side](#isolate-the-side-first): a side whose A/A and alone-vs-batched both pass is stable; the other side moved |
| One stable failing sample | Same-prefix sibling (the engine's logprobs differing over the shared prefix puts it on the engine side); save that rollout, replay it alone and with its original neighbours, then bisect by layer |
| The difference starts at one position and persists to the end of the sequence | Look at the token id sampled just before it: an id range one side treats specially (a sentinel, a placeholder, a vision id) applied by the engine's router or embedding and not by the trainer, or with another base id. Semantic class; fix the id rule, not the precision |
| Gap grows with training, initial weights fine | Score one saved batch repeatedly per side for its noise floor, then early vs late weights; then the refit (deployed bytes vs `requantize(master)`) |
| Identical operator inputs still differ | Kernel or arguments differ: executed kernel name, accumulation dtype, splits, tile, epilogue, fusion boundary, quantizer dispatch (compiled native vs CUDA op); an autotuned tactic chosen per shape class |
| Constant logprob offset on every token | An fp32-kept parameter downcast by the `param_dtype` policy, or temperature divided in bf16 ([precision chain](inventory-and-contracts.md#weights-and-precision-chain)) |
| Difference survives a precision change | It was never quantization: back to the semantic, discrete and weights classes ([order](#classification-and-handling-order)) |
| Only under graph, compile or cache | [Execution state](#execution-state) |
| Fails after changing packing, CP, TP or EP | Trainer packing A/C and the [layout rules](aligned-forward.md#layout-length-and-world-size); a failure may be a page, tile, segment or expert boundary rather than the packing code |
| Backward differs between two identical runs | A reduction with repeated destination rows (`index_select` / `gather` with repeated indices, `repeat_interleave` with tensor repeats) accumulates atomically; a thread-local deterministic config that autograd's thread does not see ([execution state](#execution-state)) |
| Gradient fine on one micro-batch, differs on another split | Accumulation order and loss normalisation: expected within the noise floor; compare against the floor, not the fp32 ideal |

## Isolate the side first

A rollout logprob that differs from the recompute does not say which side moved. Same weights, same
tokens, four controls:

| Control | How | Tells you |
|---|---|---|
| Engine A/A | Same request twice, separate processes when in doubt | Run-to-run nondeterminism |
| Engine alone vs neighbours and position | One request vs inside a full batch, and moved to another slot | Batch invariance |
| Engine prefill vs chunked vs decode | Score one history by full prefill (a prompt-logprobs request), chunked prefill and token-by-token decode | Phase invariance (attention splits, state-layer chunk grid) |
| Trainer A/A and packing A/C | Recompute twice; alone vs packed with others, permuted | Trainer determinism, packing invariance |

- Prompt-logprob scoring: the first response token is predicted at the last prompt position; never
  shift. Check that every response position got a score and reject missing ones (a cache hit can
  omit cached positions).
- Verdict: trainer A/C fails, engine stable: trainer layout first. Both stable alone: weights,
  operator semantics, scorer (run both scorers on one identical logits tensor). Same-prefix
  siblings agree over the shared prefix only; engine equal, trainer differs: trainer layout or
  state.
- Per-sequence pattern (valid and mismatching tokens, first failing position, longest run,
  lengths, group, packing slot, rank). Check shift, mask and EOS first; an offset of one or two
  positions is a comparison bug.
  - from token 0: prefill, weights, scorer or a shift (fix that token first with a prompt-only
    recompute);
  - after a boundary (KV segment, chunk, page, packing edge): state layer or split policy;
  - from a specific sampled token onward: a special id rule applied on one side only;
  - growing: propagation, only the first position is independent;
  - sparse spikes: discrete flip or a last bit across a cast;
  - all drift: weights (a stale version) or scorer dtype; only after an update: refit.
- Cheapest control: save a small batch of real prompt and response ids, prompt lengths and mask;
  only score from then on; score one version repeatedly per side for its noise floor.
- Measurement basis: per valid token, the sampled token's fp32 logprob as bytes; never a batch
  mean, a signed KL (positive and negative differences cancel) or a 6-decimal dump. Relative L2 and
  cosine are for hidden tensors only.
- Default-kernel A/A fails by design: request arrival changes each step's token count and
  composition, and default kernels are not step-invariant (norm block size, cuBLAS and MoE tiles
  follow M; an attention backend may take another split plan in decode-only steps); autotune adds a
  per-process factor. A failing default-kernel A/A is the baseline, not a lead; isolate on fixed
  tokens under the aligned settings.
- Validity of controls: cache off may run a prefill that never ran; eager changes scheduling; count
  executions of the target branch; one round proves nothing; a switch that also changes scheduling
  is not attributable.

## Compare per-token logprobs

Each side exports `{request_id: (logprobs, response_mask)}` as original-dtype tensors via
`torch.save`. Align by request id, not row; request sets must match (error, not intersection);
masks equal; at least one valid token per sequence; hash the input tokens first. Positions after
the first mismatch are not independent for history-dependent layers: count them, localise on the
first. Pass is `bitwise_count` 0 with `valid_tokens` > 0 and `nonfinite` 0.

```python
def compare_logprobs(a, b):
    """Byte-compare per-token logprobs of two sides.

    Args:
        a: Mapping request_id -> (logprobs, mask); logprobs is a Tensor of shape [tokens] in the
            original dtype, mask a bool Tensor of shape [tokens] marking valid response positions.
        b: Same mapping for the other side; the same request ids, dtypes and masks are required.

    Returns:
        Dict with dtype, valid_tokens, bitwise_count, nonfinite, abs_diff_max, abs_diff_mean and
        first_fail = (request_id, position) or None.
    """
    assert a.keys() == b.keys(), "request sets differ"
    n = bad = nonfinite = 0
    first = None
    max_abs = sum_abs = 0.0
    for uid in sorted(a):
        (x, mx), (y, my) = a[uid], b[uid]
        assert x.dtype == y.dtype and torch.equal(mx, my) and mx.any(), f"{uid}: dtype, mask or no valid token"
        x, y = x[mx], y[my]
        itype = torch.int32 if x.dtype == torch.float32 else torch.int16
        diff = (x.view(itype) != y.view(itype)).nonzero()
        if len(diff) and first is None:
            first = (uid, int(diff[0]))
        d = (x.double() - y.double()).abs()
        max_abs = max(max_abs, d.max().item())
        sum_abs += d.sum().item()
        n += len(x)
        bad += len(diff)
        nonfinite += int((~torch.isfinite(x) | ~torch.isfinite(y)).sum())
    return {"dtype": str(x.dtype), "valid_tokens": n, "bitwise_count": bad, "nonfinite": nonfinite,
            "abs_diff_max": max_abs, "abs_diff_mean": sum_abs / n, "first_fail": first}
```

Per-layer trace: same semantic boundaries on both sides under the same packing (embedding, each
layer's attention or scan output, router ids, residual, final norm, logits), keyed by request id,
boundary and logical slice after stitching TP/EP shards by global coordinates and stripping padding.
Hash first (`sha256` of the contiguous bytes plus shape, dtype and execution index) after `clone()`
(engine workspaces and graph output buffers are overwritten by the next call); raw tensors only at
the first differing boundary, which is not the cause until its input hash is shown equal. For the
final logits add full-vocab log-softmax, probability total variation and argmax as separate rows;
keep one clean sibling as the negative control.

Replay controls: one process per variant through a synchronous entry, a new directory per variant
with command, environment, exit code and time; prove frozen (weight fingerprint before and after,
no sampling, no optimizer step, the packing actually in effect). Variants touching compile,
libraries, autotune or init environment start a fresh process.

## Bisect and first divergence

- Search axis, one at a time: version (good and bad on the same reproducer; save the full dirty
  patch, a SHA is not a program); config (paired single factor, 2x2 when coupled); data trigger
  (subset, length, neighbours, boundary remainder, a special token id); compute path (layer,
  attention/MoE/norm, operator, metadata, launch order).
- Single-request prefill first: fixed-input scoring, same weights both ends, repeated for the noise
  floor; then a minimal forward (one request, one prefill, one output token, no cache hit, warm-up
  and decode calls told apart); earliest divergence at embedding, block outputs, logits at the last
  valid prompt position; same-input operator replay; then back to the original failing path at
  full depth, decode and graph.
- Dump order inside the suspect layer: norm in, norm out, q/k/v projections (combine a fused QKV
  before comparing with separate ones), position encoding (q/k after RoPE, q/k norm in the engine's
  order), attention core or scan, router logits, ids and weights, expert outputs, combine, output
  projection, residual add (a fused add+norm returns the new residual). Core and output projection
  get separate boundaries.
- Align probes by semantics: tensor meaning, dtype, global token coordinate, valid mask, not module
  names or tuple indices (FSDP wrapper names, `(hidden, residual)` pairs, THD vs padded, LSE ln vs
  log2, before or after norm or residual). EP dispatch reorders, duplicates and pads tokens: align
  expert tensors by (token, expert, slot), then compare reduction, shared+routed combine and
  post-compile rounding separately.
- Credible observations: a hook never replaces an output; under FSDP sample boundaries already
  materialised in this forward; count forward vs activation-recompute vs graph-replay executions so
  the recompute does not overwrite the forward; refuse comparisons with missing boundaries or
  incompatible shapes; non-finite fails explicitly.
- Limits of bisect: a difference can vanish downstream in rounding, masking or routing, so "layer k
  differs" is not monotonic in depth: coarse scan, suspect interval, fine scan, re-verify the
  boundaries before it. Stop at a discrete decision (top-k, expert id, tie-break, selected key set):
  fix the selection, then continue. A cut point carries full state (residual pair, KV and block
  table, state-layer state, positions, routing ids, `cu_seqlens`). Only a same-input A/B separates
  propagation from a local error; injected oracle activations, forced routing or a swapped kernel
  prove causality but are diagnostics; accept the fix on the restored path.
- Reviewable root cause: minimal input, last-equal and first-unequal boundary, same-input control,
  mechanism, trigger, minimal change; a config difference is a lead, not a cause.
- Stop rules: three rounds of only "final score differs" means add probes or re-check the sample
  mapping, do not rerun; stop when this round's goal is met, no config changes along the way;
  output replacement is diagnosis, not a fix.

## Classification and handling order

| Class | Signature | Action |
|---|---|---|
| Semantic | One side lacks or applies a feature (sinks, sliding window, q/k-norm position, `1 + w` norm, RoPE variant, a special id range) | Port the feature; no numerical work until both sides agree |
| Discrete | Same scores, different selection, order or tie-break | Fix the ordering contract; replay the engine's choice if it must be reproduced |
| Reduction order | Same inputs, last bits differ, grows with batch or length | Make the kernel invariant (fixed splits and tiles) or call the same kernel from the trainer |
| Dtype / cast position | Appears at a cast or an fp32 island | Align accumulation dtype and where the cast happens |
| State | Only after a chunk boundary, cache hit or decode step | Chunk grid, boundary-state dtype, cache reset, prefill/decode path (engine side) |
| Execution state | Only under graph, compile or a specific shape | Compile boundary, capture shape, autotune: pin the kernel on both sides |
| Weights | Bytes differ between trainer and engine | Refit dtype, fp32-kept params, requantisation; same-tensor hash after load |

Order: semantic, discrete, weights, reduction and dtype, execution state; do not move on until the
previous class is ruled out; `unknown` is a valid label, "just rounding" needs same-input evidence.
Delivery: re-verify on the original reproducer at full depth, decode and graph, counts before and
after, production path restored; only then shrink. Write "side, first operator, cause, fix,
re-verified on the original reproducer" with counts before and after; keep failing samples and
ruled-out hypotheses for the next round.

## Execution state

Each state is its own closure row ([closure and ledger](verification.md#closure-and-ledger)).

- Graph vs eager: a successful capture does not make graph bytes equal eager bytes; piecewise graphs
  fall back to eager above the capture size, each path invariant on its own and the two not equal.
  Python hooks run at capture, not at replay: observe replay through device-side static buffers and
  state whether the instrumentation changed the capture; a hook not firing is not a layer not
  executing. Localise in eager, re-check in the production graph mode, keep eager, compiled-without-
  graph and graph artifacts separately.
- Compile boundaries: Inductor fusions round differently; a same-named module may run a compiled
  native forward on one side and a CUDA op on the other. Check the executed kernel (profiler), not
  the Python name.
- Autotune: per process and keyed by the first shapes seen (Triton matmul, fused MoE, a managed
  tactic cache): pin configs on both sides, rerun a possibly re-tuned variant in a fresh process.
- Determinism only on the main thread: `torch.use_deterministic_algorithms(True)` and the Inductor
  deterministic config are thread-local in recent PyTorch; autograd's CUDA thread still sees the
  default, a compiled VJP first compiled during backward autotunes its reduction block, and backward
  step 0 is not reproducible across runs (forward logprobs unaffected). Do both:
  `torch.compile(..., options={"deterministic": True})` at the compile site and
  `TORCHINDUCTOR_DETERMINISTIC=1` in the environment; a backward with atomics needs its own
  deterministic mode. Main-thread determinism is not batch invariance.
- Cache state: prefix hit, miss and off must be bitwise; every cache binds weights, input and
  config; a new process passing does not fix the old process's cache or graph.
- Parallel layout: bitwise holds for the verified layout only; record it per row.
