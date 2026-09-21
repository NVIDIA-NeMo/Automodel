# moonlight_v4_rmsnorm_h2048 — kernel constraints

You are searching for the forward RMSNorm kernel that NeMo Automodel runs for the
block and final norms of Moonlight-V4-16B-A3B (DeepSeek-V4 architecture at
Moonlight scale). The winner is dropped into a training framework, not a
microbenchmark, so the constraints below are part of correctness, not style.

## Numerics

- `hidden` is exactly 2048 and is a compile-time constant. Specialize for it.
- `x` and `y` are bf16. `weight` is bf16. Do **not** widen the storage dtype.
- The mean square, the reciprocal square root, and the multiply by `weight` are
  computed in fp32. There must be exactly **one** rounding to bf16, on the store
  to `y`. Accumulating the sum of squares in bf16, or rounding `x * rstd` to bf16
  before applying `weight`, changes training numerics and will not be accepted
  even if it passes tolerance on random inputs.
- `eps` is added to the mean square before the reciprocal square root, not after
  it and not to the sum of squares. Every workload passes `eps = 1e-6`.
- The reference is `torch.nn.functional.rms_norm(x.float(), (2048,),
  weight.float(), eps).to(torch.bfloat16)`. Tolerances are set at one bf16 ulp
  (`max_rtol = 8e-3`); that is headroom for a different fp32 reduction order, not
  a licence for a lower-precision reduction.

## Layout and ABI

- Entry point is destination-passing: `run(x, weight, eps, y)` writes into `y`
  and returns nothing.
- `x` and `y` are 2D, row-major contiguous, `[tokens, 2048]`. `tokens` varies per
  workload (1024, 4096, 8192) and is not known at compile time. Do not assume it
  is a multiple of any tile size, and do not assume `tokens >= 1024`.
- `y` never aliases `x`. Do not write in place into `x`.
- `weight` is contiguous, length 2048, and shared by every row.
- Launch on the current stream. The caller may be inside a CUDA graph capture or
  a `torch.compile` region, so no host synchronisation, no `.item()`, no
  device-to-host copies, and no allocation of a tensor whose shape depends on a
  device value.

## Workload sizes

The three token counts are the shapes this operator actually sees in the
Automodel runs this campaign is scored against, and nothing else:

| tokens | where it comes from |
|--------|---------------------|
| 1024   | the small nsys stage configuration |
| 4096   | `examples/scalable_ai/profile_layer.py` default |
| 8192   | local batch 4 x sequence 2048, the default benchmark YAML |

8192 carries double weight because it is the shape that decides the end-to-end
step-time claim. Do not tune for the Llama-3.1 workload sizes used in the public
walkthrough.

## What is out of scope

- **Backward.** This definition is forward only. The integration keeps the
  existing fp32 reference gradient, so a fused forward that saves intermediates
  for a backward this campaign does not produce buys nothing. Emitting `rstd` as
  a side output is not part of the ABI.
- **Fusing the residual add or the following linear.** The call sites are
  separate modules; a fused-epilogue kernel cannot be dropped in.
- **Changing `weight` storage.** It is an FSDP2-sharded `nn.Parameter` named
  `weight`; the checkpoint key depends on it.

## What "beating the baseline" means here

The baseline solution in this campaign is the shipped Automodel path, which is
`torch.compile`d. A win has to hold against that, on the same GPU architecture
used for the final benchmark, at all three token counts.
