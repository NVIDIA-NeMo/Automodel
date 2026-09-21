# Kernel Factory in NeMo Automodel — proof of concept

**Status: integration proven, campaign not yet run.** The backend, the adapter, the
tests and the A/B harness are in place and measured on H100. The kernel occupying
the generated-kernel slot is a hand-written placeholder, not a Kernel Factory
result.

What this POC set out to answer: *can a Kernel Factory kernel be dropped into
Automodel without touching model code, and can we tell whether it actually helps?*
Both yes. It also answered a question nobody asked, which turned out to matter more
than the first two — see [Findings](#findings).

---

## 1. Result

One H100 80 GB SXM, 8192 tokens (`--batch-size 4 --seq-len 2048`), bf16,
hidden 2048, forward only, steady state. **GPU kernel time from
`torch.profiler`, which excludes launch overhead:**

| backend | kernel time | launches | kernel BW | % of HBM peak |
|---|---|---|---|---|
| `kf_triton_h2048` (placeholder) | **0.0238 ms** | 1 | 2.81 TB/s | **84%** |
| `te` (TransformerEngine 2.11) | 0.0256 ms | 1 | 2.62 TB/s | 78% |
| `torch` (`nn.RMSNorm`) | 0.0271 ms | 1 | 2.48 TB/s | 74% |
| `torch_fp32` (shipped baseline) | 0.1473 ms | 4 | 0.46 TB/s | 14% |

The operator is memory-bound, so there is a hard ceiling:

```
traffic  = read x + write y = 2 x 8192 x 2048 x 2 B = 67.1 MB
roofline @ 3.35 TB/s (H100 SXM HBM3)      = 0.0200 ms  <- nothing can beat this
                              kf (mine)   = 0.0238 ms  1.19x off
                              te          = 0.0256 ms  1.28x off
                              torch       = 0.0271 ms  1.36x off
                              torch_fp32  = 0.1473 ms  7.4x  off
```

A hand-written 10-line Triton kernel is already within 19% of the memory roofline.
**There is almost nothing left to search for in the forward kernel.**

Wall-clock per call tells a different and, for the campaign, less relevant story:

| backend | wall | kernel | CPU launch overhead |
|---|---|---|---|
| `torch` | 0.0287 ms | 0.0271 | **0.0016** — direct aten dispatch |
| `te` | 0.0469 ms | 0.0256 | **0.0213** — TE module wrapper |
| `kf_triton_h2048` | 0.0834 ms | 0.0238 | **0.0595** — compile wrapper + custom op + Triton launcher |
| `torch_fp32` | 0.1532 ms | 0.1473 | 0.0059 |

The fastest kernel is the second-slowest backend, because its dispatch stack costs
2.5x what the kernel does. See [Finding 1](#finding-1-the-kernel-was-never-the-bottleneck).

Warmup was not a factor: steady-state timings are identical at 5, 50 and 200 warmup
iterations for every backend, including the `torch.compile`d ones
(`torch` 0.0287/0.0287/0.0287, `torch_fp32` 0.1531/0.1532/0.1535,
`te` 0.0408/0.0406/0.0411, `kf` 0.0806/0.0800/0.0788).

### The ranking is size-dependent — and the wall-clock ranking flips at 32K tokens

Sweeping tokens from 512 to 131072 at hidden 2048, bf16, forward only:

| tokens | kernel winner | wall winner | `torch` wall | `te` wall | `kf` wall |
|---|---|---|---|---|---|
| 512 | `kf` 0.0020 | `torch` | 0.0125 | 0.0473 | 0.0844 |
| 2048 | `kf` 0.0042 | `torch` | 0.0124 | 0.0466 | 0.0835 |
| 8192 | `kf` 0.0240 | `torch` | 0.0288 | 0.0467 | 0.0835 |
| 16384 | `kf` 0.0457 | `torch` | 0.0522 | 0.0542 | 0.0835 |
| **32768** | `kf` 0.0900 | **`kf`** 0.0920 | 0.0998 | 0.1019 | 0.0920 |
| 131072 | `kf` 0.3527 | `kf` 0.3550 | 0.3826 | 0.3886 | 0.3550 |

**`kf_triton_h2048` has the fastest kernel at every single size, 512 to 131072.**
It never wins the wall clock below 32768 tokens.

The mechanism is that wall time is `max(CPU launch cost, GPU kernel time)`, not
their sum. Below the crossover the loop is CPU-bound and every backend sits at a
flat floor equal to its dispatch cost:

```
wall (ms)
0.09 ┤ kf  ████████████████████████████▁▁▁▁▁▁            floor 0.0835 = Triton launch path
0.05 ┤ te  ██████████████████████▁▁▁▁▁▁▁▁▁▁▁▁            floor 0.0466 = TE module wrapper
0.01 ┤torch ████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁            floor 0.0124 = one aten dispatch
     └────┬─────┬─────┬─────┬─────┬─────┬─────┬──
         512  2048  8192  16K   32K   64K  131K
                         ^torch     ^kf/te become GPU-bound
                          becomes
                          GPU-bound
```

Once a backend is GPU-bound its wall time converges onto its kernel time, the
dispatch cost disappears, and the ranking becomes the kernel ranking.

**Moonlight-V4 runs at 8192 tokens per norm call** (local batch 4 x sequence 2048),
which is below every crossover except `torch`'s. At that operating point `torch` is
the fastest backend by wall clock and the model is paying dispatch, not bandwidth.

---

## 2. What was built

```
examples/scalable_ai/
  kernel_factory/
    README.md                        <- this file: the POC record
    moonlight-rmsnorm/               <- the campaign, ready to launch
      definition.json                   operator contract (Kernel Factory trace format)
      workload.jsonl                    tokens 1024 / 4096 / 8192, eps 1e-6, tolerances
      prompt.md                         numerical + integration constraints for the agents
      baseline.solution.json            the number a winner must beat
      README.md                         campaign specifics + corrected slide
  configs/
    moonlight_v4_16b_kf_rmsnorm.yaml <- e2e A/B twin; one line differs from the baseline config

nemo_automodel/components/models/common/
  kf_rms_norm_kernel.py              <- GENERATED-KERNEL SLOT (the only file a winner replaces)
  kf_triton_rms_norm.py              <- adapter: custom op, sharding, reference backward, nn.Module
  utils.py                           <- BackendConfig.rms_norm + initialize_rms_norm_module accept the backend

examples/scalable_ai/profile_layer.py        <- --backend-rms-norm kf_triton_h2048
tests/unit_tests/models/common/test_kf_triton_rms_norm.py   <- 18 tests
```

No DeepSeek-V4 model code changed. The block and final norms already route through
the common backend factory (`deepseek_v4/model.py:242`, `:245`, `:650`); the
attention q/kv, compressor and indexer norms are pinned to `torch_fp32` in
`deepseek_v4/layers.py:732`, `:877`, `:1226` and are untouched.

### The seam

```
YAML: backend.rms_norm: kf_triton_h2048
   |
   v
initialize_rms_norm_module()            utils.py — one new elif
   |
   v
KFTritonRMSNorm                         hand-written, stable, NOT regenerated
   |  owns: `weight` param (checkpoint keys), [B,S,2048] -> [B*S,2048] flatten,
   |        contiguity normalization, DTensor sharding rule, autograd registration,
   |        explicit rejection of wrong dim / dtype / eps / layout
   v
kf_rms_norm_kernel.run(x, w, eps, y)    generated, volatile, replaced by `cp`
```

That split is the whole design. The regenerated file has no contracts in it; every
contract lives in the file a human owns.

---

## 3. Design decisions

| decision | why |
|---|---|
| Two files, not one | The campaign winner overwrites `kf_rms_norm_kernel.py` wholesale. Nothing that must survive regeneration can live there. |
| Forward from the kernel, **backward reused verbatim** from `Float32RMSNorm` | The campaign definition is forward-only. Reusing `_float32_rms_norm_backward` rather than reimplementing it means the two backends cannot drift in the backward, and gradients are bit-identical by construction. |
| Backend name encodes the contract (`kf_triton_h2048`) | The kernel was searched and benchmarked at one hidden size, one dtype, one epsilon. A generic name would invite silent use outside the searched envelope. |
| Reject rather than fall back | Wrong dim/dtype/eps raises `ValueError` naming the offending value and pointing at `torch_fp32`. A silent fallback would make an A/B measure nothing. |
| `eps` is a scalar *input*, not a baked constant | Matches NVIDIA's documented RMSNorm solution ABI, `run(input, weight, eps, output)`. The adapter still pins it to 1e-6. |
| Compile a thin wrapper around the opaque forward | Not cosmetic — see [Finding 2](#finding-2-the-integration-can-dominate-the-kernel). |
| Baseline solution runs the *compiled* Automodel path | An eager-PyTorch baseline would let almost any kernel "win". |
| Placeholder kernel ships in the slot | Without it nothing is runnable, no test can execute, and the verification sequence cannot be demonstrated before the campaign returns. |

---

## 4. What was verified

`tests/unit_tests/models/common/test_kf_triton_rms_norm.py` — **18 passed** on
2x H100 80 GB (torch 2.11.0a0+eb65b36914.nv26.02, Triton 3.6.0):

| gate | how |
|---|---|
| forward parity | vs `F.rms_norm(x.float(), …).to(bf16)` at 1024 / 4096 / 8192 tokens, within one bf16 ulp |
| gradient parity | `grad_x` and `grad_weight` **bitwise identical** to `Float32RMSNorm` under `force_eager`, for input-only / weight-only / both trainable |
| custom-op contract | `torch.library.opcheck` (schema, fake tensor, autograd registration) |
| `torch.compile` | `fullgraph=True, dynamic=True` at two token counts; grad and no_grad paths produce identical output |
| DTensor / FSDP2 | 2-GPU NCCL mesh, `Replicate()` and `Shard(0)`; hidden axis stays complete, `Partial` weight gradient redistributes cleanly |
| checkpoint keys | `state_dict()` keys identical to `Float32RMSNorm` |
| rejection paths | wrong dim, wrong eps, wrong dtype, wrong weight rank — 8 of these run on CPU |
| shape handling | 3D input, zero-row input, non-contiguous input |

Numerics against the fp32 reference at 8192 tokens:

```
torch             0.000e+00   bitwise equal
torch_fp32        0.000e+00   bitwise equal (it is the reference)
te                1.562e-02   within 1 bf16 ulp, not bitwise
kf_triton_h2048   1.562e-02   within 1 bf16 ulp, not bitwise
```

TE and the placeholder are indistinguishable in accuracy. Both differ from the fp32
reference only by fp32 reduction order.

---

## 5. Findings

### Finding 1: the kernel was never the bottleneck

At 8192 tokens the RMSNorm forward kernel takes **24 µs** and the dispatch stack
around it takes **60 µs**. The ranking by wall clock and the ranking by kernel time
are almost inverted:

```
kernel time   kf 0.0238  <  te 0.0256  <  torch 0.0271  <<  torch_fp32 0.1473
wall time     torch 0.0287  <  te 0.0469  <  kf 0.0834  <  torch_fp32 0.1532
```

`torch` wins on wall clock with the *slowest* of the three single-kernel
implementations, purely because `nn.RMSNorm` is one aten dispatch (1.6 µs of
overhead) while the Kernel Factory path pays dynamo guards, then `torch.library`
custom-op dispatch, then Python-side Triton launch (59.5 µs).

Consequences:

- A Kernel Factory campaign optimizes the 24 µs and cannot touch the 60 µs. Even a
  kernel at the 20 µs roofline would move the backend from 0.0834 to 0.0796 ms —
  a 4% wall-clock win on this operator.
- The repo already has the right tool for the 60 µs: `BackendConfig.cuda_graph`.
  Capturing the region removes per-launch overhead entirely, at which point the
  kernel ranking becomes the backend ranking and this kernel is the fastest of the
  four.
- Any kernel-vs-kernel claim on a slide must come from profiler or nsys kernel time.
  Module-level wall clock at this size measures Python, not CUDA.

### Finding 2: the integration can dominate the kernel

The first measurement had the candidate at **1.020 ms** against a 0.533 ms baseline
— a 1.9x loss. The forward kernel was not the cause. `Float32RMSNorm` wraps its
opaque custom op in `@torch.compile(dynamic=True)`, which is what lets Inductor fuse
the fp32 reference *backward*. Calling the custom op directly left that backward
running eager, costing both time and 0.53 GB versus 0.22 GB of peak memory.

Adding the same compiled wrapper moved the candidate to 0.489 ms. Same kernel, same
math, 2x swing — entirely integration.

**Takeaway:** an A/B between backends is only valid if everything except the kernel
is held identical, including how the surrounding graph is compiled.

### Finding 3: the shipped baseline runs four kernels where one would do

`torch_fp32` is 5.4x slower than every other backend *on kernel time*, and the
profiler names exactly why. Its body is

```python
F.rms_norm(x.float(), (hidden,), weight.float(), eps).to(x.dtype)
```

inside a `torch.library.custom_op`. Because the op is opaque, Inductor cannot fuse
into it, so `x.float()` materializes a real fp32 tensor. The per-call kernel list:

```
0.0569 ms  x2   direct_copy_kernel                      <- x.float() and .to(bf16)
0.0548 ms  x1   vectorized_layer_norm_kernel<float>     <- the actual norm, in fp32
0.0357 ms  x1   vectorized_elementwise_kernel<bfloat16>
----------------
0.1473 ms  4 launches, 0.46 TB/s, 14% of HBM peak
```

Against one 0.0271 ms `vectorized_layer_norm_kernel` for `nn.RMSNorm`. The fp32
materialization is not a rounding detail in the cost model; it is the cost.

### Finding 4: `torch` matched the fp32 reference bitwise, at 5.4x the speed

`nn.RMSNorm` produced output bitwise identical to the fp32 reference, contradicting
the `initialize_rms_norm_module` docstring, which says `torch` "computes in input
dtype". PyTorch appears to use an fp32 accumulator type for reduced-precision input.

If that holds generally, `rms_norm: torch_fp32` -> `rms_norm: torch` is a one-line
YAML change worth more than anything this campaign is likely to return.

**This is the least-verified claim in this document.** It was one shape, one dtype,
forward only, one random weight. Gradients were not compared bitwise. Do not act on
it without the sweep in [Next steps](#next-steps).

---

## 6. Limitations of the evidence

- **Kernel times are from `torch.profiler`, not nsys.** They exclude launch
  overhead, which is the point, but they are CUPTI-instrumented and nsys should
  confirm them before they go on a slide. The wall-clock column is a Python-level
  loop and is dominated by dispatch at this size.
- **The 2048–4096 token rows read above HBM peak (up to 128%), which is an L2
  artifact of the benchmark.** At 4096 tokens `x` and `y` total 33.6 MB and fit in
  H100's 50 MB L2, so the timing loop re-reads cache, not memory. Real training
  never sees that — `x` arrives fresh from the previous layer. Treat the ≥8192-token
  rows as the representative ones; below that, the sweep measures dispatch cost
  anyway.
- **The first version of this document ranked the backends by wall clock and got
  the order wrong.** `torch` looked 2.7x faster than this kernel; on kernel time it
  is 14% slower. Any conclusion drawn from module-level timing of a 24 µs operator
  should be treated as provisional until profiled.
- **One GPU, one shape, one run per configuration.** Steady-state timings were
  stable across 5/50/200 warmup iterations, but nothing here is a multi-run study,
  and only 8192 tokens was measured at kernel level.
- **Forward-only campaign.** The adapter keeps the fp32 reference gradient, so the
  backward cost is unchanged by construction. No training-speed claim is available.
- **No end-to-end run yet.** `moonlight_v4_16b_kf_rmsnorm.yaml` exists and is a
  one-line diff from the baseline config, but the 8-GPU A/B has not been run.
- **The campaign has not been launched.** No `kf` CLI command in this repo has been
  executed; the flags were taken from NVIDIA's published CLI reference. `--prompt-file`
  in particular could not be confirmed against the documented flag list.

---

## 7. Next steps

In the order that maximizes information per GPU-hour. Note that the campaign has
dropped to fourth, because the kernel is already at 84% of the memory roofline:

1. **Attack the 60 µs of launch overhead, not the 24 µs kernel.** `BackendConfig`
   already carries `cuda_graph`; capturing the norm removes per-launch cost and
   makes the kernel ranking the backend ranking. This is worth 2.5x more than a
   perfect kernel on this operator.
2. **Settle Finding 4.** Sweep `torch` vs `torch_fp32` across hidden sizes, token
   counts and dtypes, comparing forward output *and* both gradients bitwise. If they
   agree, `rms_norm: torch` is a one-line change that removes three of the four
   kernel launches from every norm in the model.
3. **Re-point `baseline.solution.json`.** Scoring against `torch_fp32` scores
   against a backend that moves 5x the bytes. The honest bar is `te` or `torch`, and
   on kernel time that bar is 0.0256–0.0271 ms — within 8% of the placeholder.
4. **Then decide whether to launch the campaign.** The forward kernel is 1.19x off
   the memory roofline, so the entire remaining search space is worth at most 19% of
   24 µs, or 4% of the backend's wall time. Launch it to exercise the Kernel Factory
   workflow end to end, and say so; it is not a credible step-time lever here.
5. **If a training-speed claim is wanted**, write the second definition — inputs
   `x`, `weight`, `grad_output`, outputs `grad_x`, `grad_weight` — and search the
   backward, which is 0.12–0.23 ms per call versus the forward's 0.024.

---

## 8. Reproducing

```bash
# unit tests (8 run on CPU, 10 need a GPU, 1 needs 2 GPUs)
pytest tests/unit_tests/models/common/test_kf_triton_rms_norm.py -q

# isolated layer A/B
python examples/scalable_ai/profile_layer.py --layer rmsnorm --batch-size 4 --seq-len 2048 \
  --backend-rms-norm torch_fp32      --no-nsys --warmup-iters 10 --profile-iters 50
python examples/scalable_ai/profile_layer.py --layer rmsnorm --batch-size 4 --seq-len 2048 \
  --backend-rms-norm kf_triton_h2048 --no-nsys --warmup-iters 10 --profile-iters 50

# end to end, 8 GPUs
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml
torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_v4_16b_kf_rmsnorm.yaml
```

The four-way comparison and the forward/backward split were produced by the scratch
scripts under `slurm_jobs/kf/` (`compare.sh`, `fwd_only.py`, `parity.py`), run in the
`nemo-automodel:nightly_202604` container on `batch`. Those are evidence, not
deliverables; they are not part of the POC surface.

To launch the campaign itself, see
[`moonlight-rmsnorm/README.md`](./moonlight-rmsnorm/README.md).
