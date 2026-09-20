# Lever 6: when no library kernel fits, write one

After the previous levers the hottest single kernel was a 32x32 wmma tile burning
**60 ms, 12% of GPU time**, at about 108 GB/s. Roughly 3% of an H100's bandwidth.

The suspect was the mHC mixer projection, `[tokens, 8192] x [8192, 24]`. Twenty-four output columns
means one column tile, so the grid is only `tokens / 32` blocks: the shape, not the library, is the
problem. So: write the kernel.

`hc_kernels.py` splits the reduction over the feature axis so the grid scales, and fuses the RMS
normalisation into the same pass. Four Triton kernels: the normalisation statistic, the forward
projection with split-K accumulation, and two backward kernels.

**The detail that mattered.** The first backward version was *slower* than eager, because each block
handled one row and re-read the whole weight matrix, 8.6 GB of it. Blocking over 32 rows amortised
that read and turned no speedup into **2.1x** on a microbenchmark.

Accuracy against eager: forward 2e-4 relative, weight gradient 8e-4, input gradient 6e-3, which is
bf16 rounding noise since that gradient is stored in bf16 anyway.

| 12 layers, micro-batch 4 | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| mixer projection on cuBLAS | 0.514 s | 13.60% | 44.3 GB |
| mixer projection on the fused kernel | 0.515 s | 13.58% | 41.3 GB |

**Same speed. 3 GB less memory.** A 2.1x microbenchmark bought nothing in the step.

That result is the interesting one. Next slide.
