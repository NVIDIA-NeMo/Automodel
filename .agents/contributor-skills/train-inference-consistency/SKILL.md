---
name: train-inference-consistency
description: Make a model's training forward match the rollout engine's logprobs bitwise (batch-invariant, aligned kernels) for on-policy RL, e.g. when vllm_kl or bitwise_frac is nonzero. Not HF parity.
---

<!-- Adapted from NVIDIA-NeMo/labs-molt, .agents/skills/train-inference-consistency/ at commit
745f0966c2c163e3a13ac1b72fc5f12deceb70b3 (Apache-2.0), rescoped to the model-definition work this
repository owns. -->

# Train-inference consistency

Goal: for every valid response token, the logprob the training forward recomputes under the same
weights equals the logprob the rollout engine returned when it sampled the token: the same bytes in
the original dtype, at every RL step. The method is the one described in the vLLM post
[Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven][post]: the training
forward calls the engine's own kernels for every operator that produces a value, so inference defines
the numbers; the backward reuses the native math of the training stack under a written gradient
contract and is not bitwise.

[post]: https://vllm-project.github.io/2026/10/14/train-inference-consistency.html

The method and the gate workflow this skill is cut from were developed for *REINFORCE Zero: Kernel
Alignment Unlocks Stable Asynchronous RL with Importance Sampling* (NVIDIA, 2026). That work shows
that once rollout and learner kernels are aligned, asynchronous RL stays stable with ordinary,
unclipped token-level importance sampling correcting the weight lag, without trust-region clipping
or sequence-rejection gates. Weight lag and kernel mismatch need different fixes: refreshing stale
weights addresses lag, while kernels that disagree at identical weights are fixed only by alignment,
which is the part this repository owns. The labs-molt skill is that work's agent procedure; this
skill is its model-definition slice.

## What this repository owns

Consistency spans three code bases. Only the middle row is built here.

| Layer | Owner | Decides |
|---|---|---|
| Engine kernels and their invariance: same bytes alone vs batched, prefill vs chunked prefill vs decode, graph vs eager | the rollout engine (vLLM in current recipes) | what the value is |
| Model definition: the aligned forward that calls those kernels, its backward, the weight and precision chain, the offline proofs | this repository, `nemo_automodel/components/models/<model>/` | that the trainer computes the same value |
| RL loop: synchronous stepping, weight refit, rollout logprobs and mismatch metrics, acceptance gates, throughput budget | the RL framework ([labs-molt][molt]) | that the loop uses it correctly |

[molt]: https://github.com/NVIDIA-NeMo/labs-molt/tree/main/.agents/skills/train-inference-consistency

- The engine row cannot be fixed from here. An engine kernel that is not deterministic or not
  batch-invariant has no trainer-side workaround; raise it with the engine. A trainer-side
  monkeypatch of an engine fault is not a fix.
- The loop row is not reproducible from here either. The gates that need the live loop (frozen
  recipe and baseline, engine invariance cells, the two-step RL sequence, resume, the paired
  throughput budget) are defined in the labs-molt skill; this skill names them only to say which
  of its own results they consume.
- A fix lives with its owner: engine backend or selector faults in the engine, model definitions
  here, the rest in the RL framework. Never hide code in a recipe or launcher to keep a PR small.

## Not the parity-testing skill

`parity-testing` compares a model against its HuggingFace reference within tolerances
(`1e-5` fp32 on CPU, `1e-2` bf16 on GPU) and starts on CPU. Here the reference is the engine
kernel's bytes on the GPU, the criterion is zero differing bytes, and there is no tolerance to
widen: a CPU or fp32 pass says nothing about the kernel path that runs in training, and "close" is
a fail. Keep `parity-testing` for the correctness of the default path against HuggingFace; use
this skill for the aligned path against the engine.

## Start by task

Read the sections of your route in order; the rest of the skill is lookup material.

| Task | Read in order |
|---|---|
| Add the aligned forward for a model | 1. [module inventory](inventory-and-contracts.md#module-inventory) 2. [placement in this repository](#placement-in-this-repository) 3. [call kernels, not the scheduler](aligned-forward.md#call-kernels-not-the-scheduler) 4. [backward: reuse, glue or new](aligned-forward.md#backward-reuse-glue-or-new) 5. [what can be proven here](verification.md#what-can-be-proven-from-this-repository) |
| A mismatch is reported against a model defined here | 1. [isolate the side first](debugging.md#isolate-the-side-first) 2. [symptom to first step](debugging.md#symptom-to-first-step) 3. [bisect and first divergence](debugging.md#bisect-and-first-divergence) 4. [classification](debugging.md#classification-and-handling-order) |
| Review a PR that adds or changes an aligned forward | [review checklist](verification.md#review-checklist-for-an-aligned-forward-change), then [evidence labels](verification.md#evidence-labels) |

## Placement in this repository

The labs-molt reference implementation patched model class forwards process-wide from the trainer.
That design is not accepted here; the repository rules in `AGENTS.md` and `CODING_GUIDELINES.md`
map the method onto the model directory as follows.

- Everything model-specific lives under `nemo_automodel/components/models/<model>/`: the aligned
  decoder, one module per operator family that wraps the engine kernels, the weight loader for the
  engine's layouts, the process-setup function. Shared components and recipes carry no model
  identity and no engine-kernel knowledge.
- Selection is a typed config field with a backward-compatible default (off). When set, the model
  class builds the aligned decoder at construction; the default module tree and its forward are
  untouched, and with the field off the default path's bytes are identical to before (replay one
  saved batch before and after the change). No process-wide class patching, no import-time side
  effects, no monkeypatching of torch or HuggingFace globals.
- Engine imports are optional dependencies: `safe_import()` / `safe_import_from()` from
  `nemo_automodel.shared.import_utils` at module level, or imports inside the aligned path's
  functions. The default path imports nothing from the engine. A missing or wrong-version engine
  fails with a specific exception when the aligned path is requested, never at import and never
  silently.
- Process setup (environment the engine kernels read, the engine's batch-invariant initialisation,
  knobs that must equal the engine's) is an explicit function in the model package. The trainer
  process calls it before the first CUDA GEMM. It checks values that would disagree with the engine
  and raises; it does not override them. Nothing runs at import.
- Parameters keep the optimizer's identity: fp32 masters are the FSDP2 DTensors; the aligned
  forward consumes the gathered full tensor (or the checkpoint's native quantized bytes for groups
  that are not trained), never a copy the optimizer does not know. Parameters the engine keeps in
  fp32 stay fp32 through the model-wide cast (`cast_model_to_dtype` honours
  `_keep_in_fp32_modules`; a raw `self.to(dtype)` does not).
- Unsupported combinations (context parallelism, images, pipeline stages, packing variants the
  aligned path does not implement) raise a specific exception at construction or at the first
  forward, with the field that caused it. Nothing falls back to the default path silently.
- Tensor contracts are documented on every function that takes a tensor: layout, dtype, packing
  (THD with `cu_seqlens`), per-rank vs global shape for DTensors.
- Tests follow [tests in this repository](verification.md#tests-in-this-repository): CPU unit
  tests for construction, validation and mapping; GPU functional tests, skipped without the engine,
  for bytes.

## Rules that hold everywhere

- Compare logprobs as bytes in the original dtype (`view(torch.int32)` for fp32). No KL,
  probability difference, cosine or printed decimal proves bitwise
  ([object of comparison](inventory-and-contracts.md#object-of-comparison)).
- Inference defines the value. Never detach an inference output and train on it, and never rewrite
  arithmetic to imitate a kernel: call the kernel
  ([forward and backward boundary](inventory-and-contracts.md#shared-forward-and-backward-boundary)).
- Never change tokens, weights, geometry, precision or tolerances to make a comparison pass.
- "Same math" is not "same bytes": verify the executed kernel and its arguments, not the Python
  function name ([per-operator contract](inventory-and-contracts.md#per-operator-contract)).
- A conclusion's scope is the executed closure: GPU architecture, model geometry, weights, source
  and image, precision, layout, graph or eager, cache state. Changing any of them invalidates the
  result ([closure and ledger](verification.md#closure-and-ledger)).
- A metric that is absent, or 0 by construction, is not evidence. Check the producer before
  trusting a 0 ([false greens](verification.md#false-greens)).
- Forward parity, a finite gradient for every trainable parameter, and the gradient contract are
  three separate results; none stands in for another
  ([gradient contract and validation](aligned-forward.md#gradient-contract-and-validation)).
- Code changes keep the default path byte-identical with the switch off, follow the repository's
  review rubric, and ship with the tests that prove the bytes.

## Files

| File | Open when | Jump to |
|---|---|---|
| [inventory-and-contracts.md](inventory-and-contracts.md) | Listing the modules on the logprob path and deciding what "equal" means per operator, weight or backward | [module inventory](inventory-and-contracts.md#module-inventory), [per-operator contract](inventory-and-contracts.md#per-operator-contract), [discrete decisions](inventory-and-contracts.md#discrete-decisions), [weights and precision chain](inventory-and-contracts.md#weights-and-precision-chain) |
| [aligned-forward.md](aligned-forward.md) | Building the aligned forward, its backward and the gradient validation | [call kernels](aligned-forward.md#call-kernels-not-the-scheduler), [layout and length](aligned-forward.md#layout-length-and-world-size), [backward](aligned-forward.md#backward-reuse-glue-or-new), [gradient contract](aligned-forward.md#gradient-contract-and-validation) |
| [verification.md](verification.md) | Declaring a pass, planning tests, reviewing an aligned-forward PR | [what can be proven here](verification.md#what-can-be-proven-from-this-repository), [forward parity protocol](verification.md#forward-parity-protocol), [evidence labels](verification.md#evidence-labels), [tests](verification.md#tests-in-this-repository), [review checklist](verification.md#review-checklist-for-an-aligned-forward-change) |
| [debugging.md](debugging.md) | A mismatch, a graph-only or cache-only fault, a gradient that differs between runs | [isolate the side](debugging.md#isolate-the-side-first), [compare code](debugging.md#compare-per-token-logprobs), [bisect](debugging.md#bisect-and-first-divergence), [classification](debugging.md#classification-and-handling-order), [execution state](debugging.md#execution-state) |

## Terms

One sense per term, used in every file.

| Term | Meaning |
|---|---|
| A/A | the same input scored twice on one side: run-to-run determinism |
| A/C | the same sequences scored with other neighbours, order or packing: batch and layout invariance |
| aligned path | the model's forward that calls the engine kernels, selected by the config field; the default path is the module tree built without it |
| BI | batch invariance: a request's bytes do not depend on its batch-mates |
| closure | the fields that bound a result: GPU architecture, geometry, weights, source and image, precision, layout, graph or eager, cache state |
| deployed bytes | the weights the engine holds after a refit; on the bf16 route the training-forward bytes, on a quantized route `requantize(master)` or the untouched checkpoint bytes |
| engine | the rollout engine whose kernels define the value; vLLM in current recipes |
| false green | a `pass` that is not evidence: nothing executed, 0 by construction, the wrong path ran |
| LSE | log-sum-exp, the softmax denominator attention saves for its backward |
| master | the fp32 parameter the optimizer updates; an FSDP2 DTensor shard |
| oracle | an independent forward or gradient reference that does not call the code under test |
| planted error | a known error injected to prove that a check reads `fail` |
| proxy | 2 to 8 real layers of the target with every layer type kept: the daily development model |
| STE | straight-through estimator: a rounding passes its gradient through as identity |
| THD | packed token layout (total tokens, heads, dim) with `cu_seqlens` boundaries |
| VJP | vector-Jacobian product: what a backward computes |
