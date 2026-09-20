# Profile before optimising

Moving to a shape that exercises the model properly, **12 layers, micro-batch 2, one gradient
accumulation step, grouped GEMM and DeepEP, eager attention**, gives 6.7% MFU. Rather than guess
what is wrong, capture a kernel trace of one step.

```yaml
benchmark:
  torch_profile_start: 10
  torch_profile_end: 10
  torch_profile_dir: /workspace/traces/run
```

GPU time for that step, grouped by kernel category:

| category | time | share | launches |
| --- | ---: | ---: | ---: |
| elementwise | 188 ms | 37.8% | 15,027 |
| GEMM | 125 ms | 25.1% | 1,188 |
| NCCL communication | 101 ms | 20.3% | 189 |
| attention | 24 ms | 4.8% | 99 |
| reductions | 20 ms | 3.9% | 2,254 |
| DeepEP dispatch and combine | 18 ms | 3.5% | 108 |

**Three things jump out.**

1. Elementwise work outweighs GEMMs, at 15,027 launches, for a model whose credited work is 91% GEMM.
2. The single largest kernel is a **gradient ReduceScatter in fp32**, 90 ms, 18% of the step.
3. 2,254 reduction launches are the Sinkhorn iterations: 20 per site, 2 sites per layer.

Everything that follows is this list, worked from the top. The `torch.profiler` support was added to
the benchmark recipe for this work; it previously had only nsys hooks.
