# Scoreboard, and what is left

All rows: 12 layers, sequence 2048, micro-batch 4, one gradient accumulation step, 8x H100.

| # | change | step time | MFU | peak memory |
| ---: | --- | ---: | ---: | ---: |
| 0 | stock `transformers`, full model | does not run | - | > 79 GB |
| 1 | Automodel, eager, loop experts, micro-batch 1 | 23.02 s | 4.80% | 60.5 GB |
| 2 | grouped GEMM + DeepEP | 20.09 s | 5.50% | 58.8 GB |
| 3 | batch shape, one accumulation step | - | 7.98% | 59.3 GB |
| 4 | TileLang sparse attention | 0.599 s | 11.69% | 54.2 GB |
| 5 | `compile_hc` fusion | 0.561 s | 12.47% | 48.2 GB |
| 6 | bf16 gradient reduction | 0.554 s | 12.64% | 48.1 GB |
| 7 | bf16 vocabulary projection | 0.514 s | 13.60% | 44.3 GB |
| 8 | fused mHC projection kernel | 0.515 s | 13.58% | 41.3 GB |
| 9 | fused stream mix | 0.483 s | **14.48%** | 41.3 GB |

**3.0x the MFU, and 19 GB less memory**, on a model stock `transformers` could not train at all.

**Where the remaining time goes**, from the final trace:

| category | share |
| --- | ---: |
| elementwise | 39.2% |
| GEMM | 23.0% |
| TileLang attention kernels | 20.9% |
| communication | 10.3% |

**The honest gap to 20%.** That needs 0.350 s, another 1.4x. Elementwise is still the largest bucket
at 7,223 launches, but it is now spread across many small operations rather than one hot spot, so it
wants CUDA-graph capture or whole-block compilation rather than another point fix. The TileLang
backward kernel is 12% on its own and is not ours to tune. And 24% of credited FLOPs sit in the
vocabulary projection, whose cost is fixed by the 163,840-row vocabulary.

Diminishing returns are real: levers 1 through 4 bought 2.4x, levers 5 through 9 bought 1.2x.
