<!-- Adapted from NVIDIA-NeMo/labs-molt, .agents/skills/train-inference-consistency/ (workflow.md,
evidence.md, acceptance.md) at commit 745f0966c2c163e3a13ac1b72fc5f12deceb70b3 (Apache-2.0); see SKILL.md. -->

# Verification and evidence

What this repository can prove about an aligned forward, how to record it, and what it cannot
claim. Labels, false greens and the ledger are shared with the RL framework's skill so a result
produced here can be consumed there without translation.

## What can be proven from this repository

| Result | Produced here | Needs the RL framework |
|---|---|---|
| Module inventory with a verdict per row | yes | |
| Proxy model and its coverage table | yes | |
| Engine invariance: A/A, alone vs batched, prefill vs chunked vs decode, graph vs eager, cache hit vs miss | | yes (engine cells; the engine's own property) |
| Forward parity: the aligned forward recomputes a saved engine scoring bitwise, packing A/A and A/C | yes, offline against saved engine logprobs | the live mismatch metric at every step |
| Default path unchanged with the field off | yes (replay before and after) | |
| Backward verdicts, gradient contract, layer-1 VJP, planted error | yes | |
| Layer-3 proxy SFT three-arm | yes (SFT recipes) | |
| Two-step RL sequence (rollout, compare, update, rollout, compare), refit bytes, resume | | yes |
| Paired throughput and the rollout budget | | yes |

A result in the middle column is `not run` in the RL framework's acceptance until it is repeated
there; a result in the right column is never claimed from here. A forward-parity pass on a proxy
verifies the execution path, not full-depth quality, and says nothing about the live loop: a
proxy's mismatch magnitude does not extrapolate to the full model (a defect that reads `1e-12` on a
proxy reads `1e-2` at full depth).

## Proxy model

Cut 2 to 8 real layers of the released checkpoint keeping every layer type, the real hidden, head
and expert geometry, the divisibility the parallel layout needs, the state paths (KV sources,
cross-layer caches, lookups that live in specific layers) and the layer adjacency that matters;
export it as HuggingFace weights with the same quantization format. It is the daily development
environment.

- Pass when a coverage table (feature, proxy layer, test) lists what the proxy does not exercise.
- Keep the kept layer indices and the config transform next to the checkpoint.
- A proxy at chance-level language modelling still proves bytes and gradients; it proves nothing
  about learning, so a training curve on it is plumbing evidence only, and a reward that any
  distribution can satisfy (a format statistic) shows mode collapse, not capability. Say so when
  the curve is shown.

## Forward parity protocol

Input: the proxy, fixed weights, and a scoring the engine produced under the aligned settings for
fixed token ids: prompt ids, response ids, prompt lengths, response mask, and the per-token
logprob of each sampled token in its original dtype, saved with `torch.save` (no JSON, no printed
decimals), together with the engine's resolved config, its version, and the weight digest.

1. Freeze the tokens: never resample or re-tokenise; the first response token is predicted at the
   last prompt position.
2. Call the model's process-setup function, build the model with the aligned field set, load the
   same weights, and recompute the sampled-token logprobs through the entry point the trainer uses
   (`model(**batch)` with the same packing), not through a module instantiated by hand.
3. Compare as bytes per valid token ([compare code](debugging.md#compare-per-token-logprobs)):
   `bitwise_count` 0, `valid_tokens` > 0, `nonfinite` 0.
4. Same-packing A/A (the same input twice, separate processes when in doubt), then
   different-packing A/C (the same sequences with other neighbours, order, or alone).
5. Record: engine version and resolved config, the model's resolved config, the executed kernels
   (profiler names, not Python names), dtypes measured inside the forward, valid tokens, mismatch,
   the first differing position.

Both scorers the trainer may run (a no-grad scoring path and the grad-enabled forward the loss
uses) are compared; a pass of one does not prove the other. The engine's own scoring of the same
tokens (a prompt-logprobs request) is the control that separates "the engine is not
self-consistent" from "the trainer differs" ([isolate the side](debugging.md#isolate-the-side-first)).

## Evidence labels

| Label | Meaning | Reason code, always recorded |
|---|---|---|
| `pass` | every declared item of the scope completed and met its criterion | none |
| `fail` | a numerical comparison or a criterion failed | the first failing item |
| `inconclusive` | the check did not answer | *invalid*: the target path did not run (aligned kernel not dispatched, default decoder built, forward skipped); *vacuous*: zero hits (no valid tokens, no sequence crossed the boundary, no token reached the expert); *fault*: process lost, NCCL timeout, OOM |
| `not run` | out of scope or not executed; never `pass` | the scope decision |

A finished run, train/rollout consistency and a quality gain are three conclusions. Small-model
error magnitudes do not extrapolate. Completion (finished steps, finite gradients) is not
acceptance: record the instantiated class and its import source. A pass on one path or split
covers no other batch, phase, or layout; each gets its own row.

## False greens

Check before reporting a pass:

- zero valid tokens; mask all false; the comparison never executed; a rank or sample missing;
- a KL, probability or log-perplexity of 0 without a byte comparison; a metric key absent from the
  log read as 0;
- the default decoder was built (field unset, or a config object rebuilt without it) and compared
  with itself; an op instantiated directly, outside the process-setup function and the production
  config;
- a diagnostic switch (eager, substitute attention, forced routing, graph off) was on: it proves
  that path only;
- small model or short output passes, full depth or length never run;
- the trainer skipped the forward (a zero-advantage batch) and reported 0; gradient 0 or all
  rewards equal means no effective update happened;
- the job exited 0 but the script's own exit code or skipped-test count says otherwise; a success
  marker written before the process died.

## Planted error

Freeze the criterion before the run (window, tolerance, denominator, pass line, its hash recorded)
and plant a known error: a flipped bit in a byte comparison, one module's gradients scaled by 1.25
and by 1.05, a known-wrong scale rule, the engine's special id range shifted by one. The check must
read `fail`; a check that cannot resolve a 1.05 scale is useless, and no threshold moves after the
result is seen. Negative-control perturbations never enter production code: they live in the test.

## Closure and ledger

A result belongs to one execution closure: GPU architecture, model geometry (real layer count,
adjacency, hidden/head/expert), weights digest, engine and model source or image, precision,
parallel layout, graph or eager, cache state, packing. Changing any of these invalidates the row;
old passes stay in the ledger as history and are not copied into the current table. Append-only:
new results are added, nothing overwritten or deleted; a record holds the source and image hashes,
the exact commands, per-item counts (valid, mismatch, non-finite), the earliest difference, the
hypotheses excluded and how.

| Change | Minimum rerun; what no longer carries |
|---|---|
| Proxy layer count or adjacency, or hidden/head/expert geometry | Coverage table, forward parity; the old proxy's 0 is no longer promotion evidence |
| Kernel, op, tile or segment constant (always both sides together) | Same-input op test on both sides, module VJP, forward parity A/A and A/C, the boundary rows of [layout and length](aligned-forward.md#layout-length-and-world-size) |
| Quantization, dispatch, JIT or compiler | Module bytes on real inputs, proxy forward parity; a direct-call unit test does not cover the compiled module |
| Attention or MoE op, or its autograd wrapper | Class and source, same-input op, its VJP, proxy forward parity; a substitute-backend pass is diagnostic |
| Graph, cache or packing | The matching engine cells, packing A/C; eager does not cover a captured graph |
| Parallel layout (trainer TP/EP/CP, simulated engine layout, engine TP/EP) | Layout rules 1 to 5, then forward parity, then the live loop |
| Micro-batch split, recompute mode or offload | Gradient A/A, one-step update |
| Engine or model version, native build, image | Closure check inside the worker, everything in the closure; development-mount evidence does not transfer |

A proven non-dependency may be written into the ledger instead of a rerun; file names, PR text, a
tag or "looks like a cleanup" never prove it.

## Tests in this repository

The repository's test tiers and layout are in the `testing` skill; the rules below say what an
aligned-forward change adds.

- Unit tests (`tests/unit_tests/`, CPU, no engine): config validation (the field's default is off;
  unsupported combinations raise the specific exception with the field named), the inventory-driven
  weight mapping (name, layout and dtype per weight class on tiny shapes), the state dict adapter
  round trip for the engine layouts, the padding and packing arithmetic (inert rows excluded from
  loss and metrics), the STE and requantisation helpers on small tensors against a hand-written
  reference. Mark modules that never touch CUDA `cpu_only`.
- Functional tests (`tests/functional_tests/`, GPU, within the 2-GPU cap): `pytest.importorskip`
  the engine package and skip without a supporting GPU; assert bytes (`view(torch.int32)` equality
  of logprobs against a saved engine scoring committed as a small artifact, or produced in the test
  by the engine's scoring path on tiny shapes), kernel dispatch (the engine op was called: a counter
  or the profiler), and the backward (gradients bitwise across two runs with a random upstream
  gradient and imbalanced groups; `out.sum().backward()` cannot detect accumulation-order
  nondeterminism). A test that proves the default path's bytes are unchanged with the field off
  belongs here too.
- Never assert on source strings, and never make a test pass by widening a tolerance: a bitwise
  test has none.
- Tests that need the released checkpoint, a multi-node layout or the live loop are not repository
  tests; they are recorded in the ledger with their commands and run where the checkpoint lives.

## Review checklist for an aligned-forward change

Apply the repository review rubric (`skills/pr-review/SKILL.md`) first; this list adds what is
specific to an aligned path.

1. Model directory only: no engine knowledge, model identity or kernel choice in shared components,
   recipes or launchers.
2. The field defaults off and the default path is byte-identical with it off; the PR shows the
   replay.
3. Engine imports guarded (`safe_import()` or function-local) and no engine import on the default
   path; a missing engine raises a specific exception when the aligned path is requested.
4. No process-wide patching, import-time side effects, or monkeypatching of torch or HuggingFace
   globals; the process-setup function is explicit and checks rather than overrides.
5. Every value-producing operator calls the engine kernel; no torch re-implementation stands in for
   a kernel's bytes; no `detach()` of an inference output on the training path.
6. Each `torch.autograd.Function` saves what its backward needs, returns one gradient per input,
   handles `None`, declares no in-place mutation it does not perform, and has a backward test.
7. The inventory table is in the PR (or the docs it updates) with a verdict per module, including
   the trainer scorer and the engine's sampler kernel, special id ranges, frozen groups.
8. Parameters keep the optimizer's identity; fp32-kept parameters are excluded from the cast through
   the supported mechanism; quantized groups state their trainable set and requantisation function.
9. Tensor contracts documented on every new tensor-taking function (layout, dtype, packing, per-rank
   vs global shape).
10. Evidence attached as labels with closure fields: forward parity (both scorers, A/A, A/C), the
    gradient layers run, the planted error, and the explicit list of what is `not run` here and
    awaits the RL framework.
