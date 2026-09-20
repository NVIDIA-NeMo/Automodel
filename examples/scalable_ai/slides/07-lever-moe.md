# Lever 2: expert-parallel communication and grouped GEMMs

Two independent choices in the mixture-of-experts layer.

**Token dispatch.** The `torch` dispatcher all-gathers activations across the expert-parallel group,
so every rank holds every token and then computes only its own 8 experts. DeepEP sends each token
only to the ranks that need it.

**Expert GEMM.** The `torch` backend loops over local experts with gather and scatter per expert.
The `gmm` backend issues one grouped GEMM per layer.

Full 16.5B model, 27 layers, sequence 2048, micro-batch 1, 524k tokens per step:

| MoE backend | step time | tokens/s | peak memory | MFU |
| --- | ---: | ---: | ---: | ---: |
| per-expert loop + all-gather dispatcher | 23.02 s | 22.8k | 60.5 GB | 4.80% |
| grouped GEMM + DeepEP | 20.09 s | 26.1k | 58.8 GB | 5.50% |

**1.15x and 1.7 GB.** Less than you might expect, and worth understanding why: at micro-batch 1 the
step is dominated by everything *except* the expert GEMMs. The same change measured on a shape where
the MoE actually dominates is worth more. A lever's value depends on what is currently the
bottleneck, which is the argument for profiling first.
