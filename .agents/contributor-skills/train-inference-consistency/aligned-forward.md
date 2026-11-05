<!-- Adapted from NVIDIA-NeMo/labs-molt, .agents/skills/train-inference-consistency/ (training-side.md,
molt-integration.md) at commit 745f0966c2c163e3a13ac1b72fc5f12deceb70b3 (Apache-2.0); see SKILL.md. -->

# The aligned forward and its backward

How to build a training forward that computes the engine's math for a model defined in this
repository, and the backward that trains it. One route: the model's own decoder, selected by the
config field ([placement](SKILL.md#placement-in-this-repository)), whose value-producing operators
are the engine's kernels wrapped in `torch.autograd.Function`s whose backward is the native
training math.

## Call kernels, not the scheduler

- Teacher-forced full-sequence forward, boundaries in `cu_seqlens`; borrow the kernel only. No KV
  paging logic, decode scheduling or CUDA-graph management from the engine.
- Same branch as inference: the engine's compile and custom-op boundaries, attention backend and
  batch-invariant kernel selection reproduced in the trainer process by the model package's
  process-setup function (environment before the first cuBLAS handle, the engine's own
  initialisation call, the same knob values) and in every test wrapper. The engine reads some of
  these values at import or at the first call: a test that imports the engine before calling the
  setup function measures another configuration.
- Op-level calls (the engine's custom ops, fused top-k, fused experts, the attention op), not the
  engine's module objects, which need the engine's TP/EP process groups the trainer would have to
  create.
- Keep the training parameter container: parameters stay the FSDP2 DTensors behind the fp32
  master; inside the forward use the gathered full parameter (`full_tensor()` for an explicitly
  sharded one), and no parameters or copies the optimizer does not know. Keep the model's entry
  point (`model(**batch)`): the aligned decoder is selected at construction, not by re-routing the
  call chain.
- Reuse order: autograd-capable native module, then a short torch re-implementation of the engine's
  semantics, then a ready-made op with a backward, then a custom autograd function or kernel only
  when necessary. Throughput is not a goal at this stage.
- What is trained, frozen or carries an auxiliary loss follows the model definition: no detaching
  to get it running, no auxiliary coefficient the recipe does not have.
- fp32 where the engine is fp32 (RoPE angles, router logits and bias, norm statistics) is set per
  operator, never by a forward-level `torch.autocast`: one would downcast the fp32-kept gate, and
  casting matmul inputs alone does not stop a downgrade. Measure dtypes inside the forward.

Three rules for "the same kernel" that still differs:

1. Direct and compiled paths of one kernel are two numerics: call it through the same opaque
   custom-op boundary on both sides, or with compile off on both.
2. Attribute only after a same-input op test: a parameter that differs between the sides (a
   reduction block size) is a lead, not a cause, until the identical captured kernel on the same
   input differs.
3. Reproduce the executed cast position: keep the real bf16 head output and share the scoring;
   never truncate an fp32 projection to manufacture equality.

Reproduce the math from the same checkpoint, config and inference code: layer order, parameters and
buffers, residual and norm positions, position encoding, visibility range, router, special
embeddings and the id ranges the engine treats specially.

- A frozen base still propagates input gradients: `W.requires_grad_(False)`, never `no_grad()`
  around the block (it cuts adapters and encoders upstream; a frozen table lookup is the one
  exception). A zero-initialised adapter shows zero gradient legitimately: judge with a non-zero
  adapter and a real step, not a finite loss.
- Metric-only quantities (mismatch counts, KL estimates, entropy) are computed under `no_grad` on
  detached copies, never in place on a tensor the loss needs.
- Mapping checks cover values and semantics, not only shape: per weight class, source to training to
  engine name, axis transforms, sharing (tied embeddings); fused QKV, gate/up order, expert
  numbering, extra norm or bias, non-persistent buffers, derived state; fused parameters overwritten
  whole; every missing or extra item with a written reason.
- Stateful modules (conv, scan, recurrence) consume `cu_seqlens`; a correct attention mask proves
  nothing about them. Test unequal lengths alone and merged; sentinel-fill outputs with a different
  value per side to catch missed writes (zero-filling both sides reads as a false green).
- Recompute under activation checkpointing reads the same saved state (routing ids, RNG) and is
  bitwise, because the forward is deterministic; a near-tie token must not re-route in the
  recompute.
- Proxy: the smallest layer set covering every non-equivalent path (usually 2 to 8 real layers):
  keep mixed attention, cross-layer sharing and lookups that live in specific layers; record the
  kept layer indices and the config transform; it proves the covered paths only
  ([proxy model](verification.md#proxy-model)).

Order of work: inventory, then module comparison (same weights and inputs; forward bytes, gradients
vs an oracle), then proxy training (non-zero loss, finite gradients, the intended parameters
change), then save and restore in a fresh process. Every step ends with a short conclusion, the
source version, a reproducible command and an artifact; distribute only after the first two pass.
Cost is judged per blocker (an operator without a backward or clear semantics is its own item),
never as a uniform figure.

## Layout, length and world size

Record the trainer and engine topologies separately: the trainer mesh the recipe resolved (data,
tensor, context, pipeline and expert axes) next to the engine's tensor-parallel size, expert
parallelism and engine count.

Budget the aligned forward's memory before asking for GPUs: fp32 attention partials sized by the
context ceiling, one paged KV copy per packed sequence, the fp32 recompute in backward, chunked fp32
logits in the scorer, the refit's per-parameter `full_tensor()` gather. On a quantized route add
the dequantised copies the backward needs for the active experts of a micro-batch; chunk them when
they do not fit, with the accumulation dtype of the partial `dX` recorded (chunked and one-shot
`dW` are bitwise equal, chunked `dX` differs by its accumulation order and is deterministic).

Packing: the micro-batch is one row with `cu_seqlens` and per-sequence positions; the aligned
attention builds one block table per packed sequence so bytes do not depend on neighbours. Prove it
(alone vs packed with 1 / 3 / 7 / 15 others, bitwise). Padding rows the kernels require (a row
multiple) are inert: they never alter valid rows and are excluded from every loss and metric.

Context parallelism is out of the aligned path's contract until it has its own forward-parity and
gradient proof: attention is exact only with an all-gather of K/V and the local query rows at their
global positions; ring exchange is another softmax merge order. The path raises on `cp_size > 1`
rather than running a layout it has not proven.

EP and collectives: the aligned expert forward runs on one rank's local tokens with the engine's
fused kernel and tile config; the combine (fp32, top-k slot order) is part of the contract. Record
expert mapping, tokens per rank, padding capacity, route order; an empty rank still joins the
collective; reused buffers get sentinel fills.

Bitwise identity after changing the parallel layout: addition is not associative, so the layout
decides what is summed in which order and where it rounds, and "bitwise" holds for the verified
layout only.

1. The reduction tree is part of the contract: per level the dtype, order and rounding points;
   every legal engine layout and the trainer walk the same tree (engine TP all-reduce, EP combine,
   trainer vocab-shard logprob reduction).
2. A degenerate layout (TP = 1, EP = 1) is not automatically equal to a multi-rank one; it simulates
   the tree: group by rank, partial sums, round at the same places, merge in the same order. A
   simulated engine layout is a config field of the aligned path, recorded with every result.
3. After a layout change compare raw bytes on fixed inputs first, then train; never revert to the
   old layout to get around a non-zero mismatch.
4. Locate by layer-wise dump: the first differing layer whose inputs (router logits included) are
   bitwise has the fault in its reduction.
5. Simulating an order costs time; measure it and report it apart from the layout change's gain.

| Long-sequence boundary | Fact | Test |
|---|---|---|
| Segment ceiling | The aligned attention's context ceiling is a correctness bound shared by both sides, not a performance knob | Exceeding it raises, never truncates; two real lengths below it; the engine's decode equals prefill at segment edges |
| Chunk boundaries | Engine prefill chunking and a state layer's chunk grid are independent numerical boundaries | Chunked equals full prefill at the chunk edges (engine cell); one sequence crossing a state chunk edge |
| KV block | The trainer's paged KV copy has block edges | Sparse `dY` across a block edge |
| Packing prepend | Prepending a short sequence shifts every row offset of the target | Two prepend controls vs none, bitwise |
| Micro-batch fits the longest row | The micro-batch must hold the longest packed row at the formal length | One OOM probe at the formal length; no extrapolation from a shorter one |
| Embedding backward | CUDA `embedding_dense_backward` takes another accumulation path for small index counts | Tolerance tiered by length |

Micro-batch split and gradient difference: the same batch under another split gives a measurably
different gradient (loss normalisation, accumulation order, the embedding backward above). Not a
bug, but the criterion denominator is `max(bf16 rounding floor, implementation noise floor)`, never
the fp32 ideal; both arms use the same split; a split change means A/A first; the gradient norm is
not comparable across micro-batch counts.

## Backward: reuse, glue or new

Backward is determined by the math, not the forward kernel: `y = f(x, W)` needs the formula and the
saved inputs (`dX = G @ W`, `dW = G^T @ x`). An engine forward kernel does not call for a new
backward, and "no backward at the same precision" is no reason to write one; the backward runs in
bf16 or fp32 anyway.

Wiring: one `torch.autograd.Function` per value-producing op. Its forward calls the engine op
without a graph and saves inputs plus the intermediates the native backward consumes (LSE, routing
ids); its backward recomputes the native math from them and differentiates it. Never return
`visible.detach()` as the trainable value. Where a fused kernel has no backward at all,
`visible + (graph - graph.detach())` with `graph` the same math in PyTorch keeps the kernel's bytes:
eager only (a compiler may rewrite the add and subtract), never modified in place downstream; it
holds the surrogate graph in memory, which the `Function` form avoids.

| Case | Signature | Examples | Do |
|---|---|---|---|
| A: only rounding differs | Same math; summation order or chunking differs; last-bit difference | Linear, RMSNorm, attention, `silu * up`, RoPE, scan, conv | Reuse the native backward; write no math |
| B: forward has rounding | A quantization rounds; the true derivative is 0 almost everywhere | FP8 / MXFP8 / NVFP4 GEMM, quantized Q/K/V | Fix the STE contract (rounding passes as 1; `dX` uses the dequantised value by default), then reuse the plain GEMM backward; quantify in [layer 2](#layer-2-gradient-contract-comparison) |
| C: no same-math implementation for the segment | Composite of several ops, no library backward for the whole | Routed experts: sort, grouped GEMM, activation, restore, top-k weighted sum; model-specific scores | Glue over ready-made ops (`grouped_mm`, analytic activation derivative, differentiable combine), compared against an oracle |

Decision flow: rounding in the forward (top-k id selection does not count)? no: A. yes: fix the
STE, then packaged backward for the whole segment? yes: B, no: C. Look in the model's native module
and torch autograd first, then the training kernel libraries. What forces glue (none of it "missing
same-precision backward"): a library backward bound to intermediates of its own forward (rstd, token
permutation) that the engine forward lacks, so recompute them; model-specific math in no library;
discrete selection that must copy the forward (a top-k recomputed in backward can pick another
expert on a tie: store ids, differentiate only the continuous score); a generic autograd recompute
too costly in memory or time.

Three rules when reusing:

1. Intermediate definitions match: an attention backward takes the forward's O and LSE (ln vs
   log2, softmax scale folded or not, per-head vs per-token layout); grouped-head `dK`/`dV` summed
   within the group in a fixed order. If they do not match, recompute the forward with the training
   kernel inside the backward; O, LSE and P come from one forward.
2. The backward is not required to be bitwise: recomputed intermediates differ in the last bf16
   bit, gradients by the same amount; the contract must hold and the error sit near the noise
   floor. Report bitwise and tolerance apart.
3. Reused code is verified too: an fp32 oracle that does not call the backward under test, the
   same `dY`, the `dX` / `dW` error in multiples of the noise floor.

False differences (test errors, never a train/inference difference):

- FSDP2 unsharded copy after a forward-only pass: the root module is not resharded, so a parameter
  fetched by name is the bf16 compute copy, not the fp32 master; perturbing it or reading its
  `.grad` misleads. Reshard or finish a backward first.
- The aligned path was never hit: an op instantiated directly misses the production context (the
  process-setup function, the config field), or an import without GPU visibility makes the engine
  pick a non-CUDA platform; the native path is compared with itself. Assert the target call site
  was hit (kernel name in the profiler, a counter in the module); use the production config.
- `dY` fp32 on one side and bf16-rounded on the other; dumps taken at different points (before vs
  after the FSDP reduce, EP scaling, loss normalisation) read as a multiplicative gap; "frozen" by a
  zeroed grad while weight decay still moves it (a real freeze is `grad = None` or removal from the
  optimizer).

## Gradient contract and validation

Forward 0 diff proves the same logprob, not correct gradients. Three layers, run 1, then 2, then 3:
a layer-1 failure is fixed before anything else; a layer-3 anomaly sends you back to 1 and 2; none
proves RL stability.

| Layer | Question | Method | Cannot replace |
|---|---|---|---|
| [1. Module-level VJP](#layer-1-module-level-vjp-comparison) | Does the implementation equal the contract (missing scale, wrong transpose, broken chain) | `dX`, `dW`, other params and one optimizer step per module vs an oracle | 2, 3 |
| [2. Gradient contract](#layer-2-gradient-contract-comparison) | Is the contract itself sound; does an approximation bias training | Same batch, gradients under both conventions per class and per layer | 1 |
| [3. Proxy SFT three-arm](#layer-3-proxy-sft-three-arm-control) | Does the whole chain carry a systematic bias | Three SFT arms, same data and hyperparameters | 1, 2 |

Write the contract first. A weight has three tiers and the gradient must return to the first:

| Tier | Here | What it is |
|---|---|---|
| master | fp32 FSDP2 shard | Optimizer state; updates accumulate here |
| training-forward | bf16 via `MixedPrecisionPolicy(param_dtype)`, or the dequantised native bytes on a quantized route; fp32-kept params (gate, norms per class) | The value the aligned kernel multiplies |
| deployed | The bytes the engine holds after a refit | bf16 route: the training-forward bytes; quantized route: `requantize(master)` or the untouched checkpoint bytes |

Keep the optimizer's parameter identity; the deployed value is not trainable. Per module class
write down: the visible forward (op, I/O dtype and layout, cast positions, accumulation precision;
quantized: quant positions, scale granularity, rounding); the trainable variables (input, master
weight, bias, norm weight, router; what is frozen); the surrogate (`dX` from master or dequantised
value; identity or clipped STE; bf16 route: none); the discrete state (ids, kept token sets,
ordering; continuous combine weights still get gradients); the backward (reused or glue; cast,
accumulation and reduction precision, collectives, normalisation); save and recompute (inputs, LSE,
ids, RNG; the surrogate never switches on recompute); the oracle, shapes and paths, metrics and
tolerances fixed.

Which weight `dX` uses: `Y = X @ W^T + b`, upstream `G`: `dW = G^T @ X_ref`, `db = sum_tokens G`,
`dX = G @ W_ref`. bf16 route: `W_ref` is the bf16 training-forward weight the kernel used, the
deployed byte, no choice. Quantized route: the dequantised value (default; the exact derivative of
the executed forward) or the master (reuses the plain GEMM backward, ignores the quantization error
in `dX`); the master choice makes layer 2 mandatory. State `X_ref` (original or quantized
activation) likewise. Before reusing a linear or grouped backward check flatten and transpose,
padding and valid tokens, bias reduction, saved inputs, scaling, accumulation dtype; an equal output
shape proves nothing.

### Layer 1: module-level VJP comparison

Oracle: torch ops and autograd, never the backward under test; where a discrete op is involved
implement the agreed surrogate explicitly (do not differentiate bare rounding and call the zero an
oracle). Oracle path: visible value from the engine op under `no_grad`; surrogate graph from the
same-math torch code in normal grad mode; a bridge `Function` whose forward asserts shape, device
and dtype and returns the visible value and whose backward passes the gradient to the surrogate.

1. Fix the state (inputs, weights, mask, routing ids), independent copies, cleared grads; check
   the visible forward equals the engine bytes, then the saved state; at integration confirm the
   real forward produces it.
2. Same upstream gradient, same dtype, representable values: fixed random, zero, sparse (non-zero
   at chosen positions), then gradients captured from real training (synthetic inputs are labelled
   as such). `y.sum().backward()` alone misses transposes, permutations and broadcasts.
3. Compare per item: `dX`, `dW`, every bias, norm, router and scale: missing, shape, finiteness,
   non-zero where expected; unused differentiable inputs listed, not ignored. Report max abs error,
   relative L2 `||g - g_ref|| / max(||g_ref||, eps)`, norm ratio and cosine (undefined at zero norm:
   absolute error).
4. Tolerances fixed in advance from dtype, reduction length, oracle precision and this model's own
   noise floor and baseline (never another model's number), per module, reasoning recorded; never
   relaxed after a failure, never with "STE is approximate anyway" for a same-semantics kernel.
5. Coverage: small M (decode-like) and large M, several lengths, unaligned tails, padding, packing
   and mask, zeros and extremes; MoE: empty experts, repeated routing, shared experts, top-k ties;
   shared parameters: gradient accumulation across use sites; under the target dtype and layout,
   activation checkpointing, gradient accumulation and the real parallel reduction (fp32 reduce,
   EP): local first, then integrated.
6. One-step update from the same parameters and optimizer state; compare the master change.
   Gradients close but updates apart: scaling, clipping, optimizer state, parameter binding.

Smooth modules can be cross-checked with small high-precision finite differences (discrete branch
fixed, away from non-smooth points); never accept an STE with finite differences of a quantized
forward.

| Module | Reuse first | Semantics to check |
|---|---|---|
| Linear / QKV / lm_head | Native linear VJP; gradient returns to master | `dX` weight convention, cast, bias, tied-weight accumulation |
| Norm / RoPE | Native VJP, fp32 statistics | eps, `1 + w` variant, position offset, neox layout, where cos/sin were built |
| Activation | Analytic derivative composed as the forward fused it | Saturation, fusion order (`silu_and_mul` on `cat(gate, up)`) |
| Attention | Training backward with the same mask and index semantics | Causal boundary, softmax scale, LSE definition, group sum, O/LSE/P from one forward, per-sequence block table / `cu_seqlens` |
| MoE experts | Grouped VJP plus a differentiable combine | Inverse dispatch permutation, repeated tokens, empty experts, shared experts, top-k weight normalisation, weight before or after the down GEMM |
| Router | Ids fixed, derivative of the continuous score | No gradient for ids; the router output is never detached whole; the fp32 gate stays fp32 |
| State layers | Training backward with the same chunk arithmetic | Chunk boundary, initial state, state dtype, conv state |
| Selected logprob | log-softmax / gather VJP matching the definition | Vocab-shard reduction, temperature divided in fp32, token mask, loss normalisation |

### Layer 2: gradient-contract comparison

When the contract holds an approximation (`dX` from master, any STE), fix checkpoint and batch,
compute `dX` under both conventions for every affected layer, and report cosine and norm ratio per
class and per layer with outliers named, per precision combination. On the bf16 route there is no
approximation and layer 2 is the written statement of that fact; it becomes mandatory the moment a
quantized route enters. A result holds for the precision, learning rate and run length measured;
re-measure when any of them grows.

### Layer 3: proxy SFT three-arm control

The SFT recipes in this repository run the three arms on the proxy with one shared packed dataset
and identical hyperparameters:

| Arm | Forward | Backward | Purpose |
|---|---|---|---|
| Default arm | Default kernels (`BackendConfig`) | Native autograd | The recipe's baseline |
| Aligned + native backward | Aligned forward | Native backward (production) | The deliverable |
| Reference arm | Aligned forward, same snapshot | Oracle surrogate (the layer-1 bridge) | Isolates backward implementation error |

Do not start SFT until layer 1 passes.

- Order: first compare gradients and the one-step update of deliverable vs reference arm on a fixed
  checkpoint and batch, then run the curves; after the trajectories diverge later agreement is not
  assumed; the deliverable-vs-default gap includes forward-path differences and is not all backward;
  when the budget forbids the full reference-arm curve, compare representative batches and state
  the missing range.
- Controlled: initial weights and optimizer state, data and order, labels and mask, packing, loss
  normalisation, global batch and accumulation, learning rate and schedule, clipping, precision,
  seed, recompute policy. Recorded: train loss, NLL on a fixed validation set, gradient norm, update
  RMS, clip ratio, non-finite count, throughput, on an x-axis of optimizer steps or cumulative
  tokens; window and acceptable deviation agreed before the run.
- Localisation: loss differs at the initial weights: forward, data, mask. Initial loss equal,
  one-step update differs: backward, reduction, scale, clipping, optimizer. Divergence after
  several steps: save the first abnormal checkpoint, check recompute, learning rate, RNG, then a
  same-weight replay. Unexplained: `inconclusive`; forward 0 diff does not cover it.

Freeze the criterion; prove the update lands on the master:

- The criterion (tolerance, denominator, thresholds) is frozen before the run, its hash recorded,
  never edited. Planted error first ([planted error](verification.md#planted-error)). Several
  snapshots (steps 0 / 50 / 150 / 250): a step-0 pass does not carry. On failure, bisect by
  injecting oracle gradients block by block.
- The gradient export path gets its own A/A (gradient norm bitwise, local shards equal to the dump,
  replica diff 0). Cross-process bitwise backward is not required, but a sporadically non-bitwise
  first step has a cause (compile autotune by timing, a thread-local deterministic config) and is
  found, not tolerated.
- The optimizer consumes the clipped gradient, the update lands on the fp32 master, and the
  deployed bytes after the next refit equal the cast of the master (quantized route:
  `requantize(master)`); every parameter-group type is covered (weight-decay-0 groups, fp32-kept
  params), not one step of one group. A failure splits into a criterion modelling gap (written
  down; the first Adam step moves every coordinate by the same magnitude, which voids ratio-type
  checks) and an implementation error (fixed).

Report per module class: the contract (`dX` choice, STE type, scale handling), oracle, `dY`
construction, shapes and paths covered, metrics and tolerances with reasoning; layer 2 cosine and
norm ratio per class and layer with outliers (or "none: bf16 route"); layer 3 config diff of the
three arms, curves, window conclusion; the planted-error result and snapshots; missing coverage
listed separately, never "the rest is analogous".
