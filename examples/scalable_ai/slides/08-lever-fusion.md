# `compile_hc`: fuse the four-stream mixer

> Same computation, allclose numerics; fewer launches and fewer round trips through HBM.

| eager mHC | `backend.compile_hc: true` |
| --- | --- |
| `x32 = streams.float()`<br>`mix = linear(rmsnorm(x32), W)`<br>`pre, post, logits = gates(mix)`<br>`comb = sinkhorn(sigmoid(logits))`<br>`collapsed = (pre * streams).sum(hc)`<br>`expanded = post * y + matmul(comb.T, streams)` | `pre, post, logits = compile(weights)(streams)`<br>`comb = sinkhorn(...)  # unchanged`<br>`collapsed = compile(sum(pre * streams))`<br>`expanded = compile(post * y + sum_i(comb_i * stream_i))`<br><br>Inductor fuses the cast, norm, gates, and reductions; unrolling the four-stream mix turns the tiny batched GEMM into one fused pass. |
| **23 forward + 51 backward = 74 kernels/site** | **9 forward + 25 backward = 34 kernels/site**<br>**61% fewer forward, 51% fewer backward, 54% fewer total** |

The full-model A/B changes only `compile_hc`:

| 27 layers, sequence 2048, GBS 256, 8x H100 | off | on | gain |
| --- | ---: | ---: | ---: |
| step time | 14.520 s | 13.721 s | **5.5% lower** |
| throughput | 36.11k tok/s | 38.21k tok/s | **5.8% higher** |
| MFU | 7.609% | 8.052% | **+0.443 points** |
| rank-0 peak memory | 47.98 GiB | 44.65 GiB | **−3.33 GiB** |

*Kernel counts are an isolated H100 profile of one mHC site at local batch 1; the unchanged
TileLang Sinkhorn is included on both sides. There are two sites per transformer layer. Slurm job
`19039285`. The 12-layer teaching proxy shows the same result: 0.388 s to 0.355 s (8.5% lower).*
