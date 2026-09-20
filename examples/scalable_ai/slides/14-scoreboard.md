# Scoreboard, and what is left

Each row adds one change to the row above. Sequence 2048, FSDP2 across 8 GPUs throughout.
`mb` is the micro-batch per GPU, `accum` the gradient accumulation steps, `EP` the expert-parallel size.

| # | change | layers | mb | accum | EP | attention | MoE and fusions | step | tok/s | MFU | memory |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | stock `transformers` | 27 | 1 | 32 | 1 | eager, dense then mask | HF module list | does not run | - | - | > 79 GB |
| 1 | Automodel native model | 27 | 1 | 32 | 8 | eager | loop experts, all-gather | 23.02 s | 22.8k | 4.80% | 60.5 GB |
| 2 | grouped GEMM + DeepEP | 27 | 1 | 32 | 8 | eager | `gmm` + DeepEP | 20.09 s | 26.1k | 5.50% | 58.8 GB |
| 3 | batch shape | 12 | 3 | 1 | 8 | eager | `gmm` + DeepEP | 0.658 s | 74.7k | 7.98% | 59.3 GB |
| 4 | TileLang sparse attention | 12 | 4 | 1 | 8 | TileLang | `gmm` + DeepEP | 0.599 s | 109.4k | 11.69% | 54.2 GB |
| 5 | `compile_hc` fusion | 12 | 4 | 1 | 8 | TileLang | + compiled mHC cores | 0.561 s | 116.8k | 12.47% | 48.2 GB |
| 6 | bf16 gradient reduction | 12 | 4 | 1 | 8 | TileLang | + bf16 ReduceScatter | 0.554 s | 118.3k | 12.64% | 48.1 GB |
| 7 | bf16 projections | 12 | 4 | 1 | 8 | TileLang | + bf16 vocabulary and mixer | 0.514 s | 127.5k | 13.60% | 44.3 GB |
| 8 | fused mHC projection kernel | 12 | 4 | 1 | 8 | TileLang | + Triton norm-projection | 0.515 s | 127.3k | 13.58% | 41.3 GB |
| 9 | fused stream mix | 12 | 4 | 1 | 8 | TileLang | + 4-way mix as multiply-adds | 0.483 s | 135.7k | **14.48%** | 41.3 GB |

Rows 0-2 are the full model at micro-batch 1, which is where the stock baseline fails; rows 3-9 use
12 layers so each later change is measured against something that runs. Tokens per second is only
comparable within a layer count, MFU across all of it. Row 7 was measured without the
gradient-reduction change and row 8 with it, hence the tie.

## And on the real model

The 12-layer model keeps the comparison honest against a baseline that could not run. The tuned
configuration on the **full 27 layers**, 16.5B parameters:

| full model, sequence 2048 | step time | tokens/s | MFU | peak memory |
| --- | ---: | ---: | ---: | ---: |
| baseline: eager, loop experts, micro-batch 1 | 23.02 s | 22.8k | 4.80% | 60.5 GB |
| tuned, micro-batch 2 | 0.742 s | 44.2k | 9.30% | 46.3 GB |
| tuned, micro-batch 3 | 0.900 s | 54.6k | **11.51%** | 62.1 GB |

Row 1 of the table above against the last row here is the end-to-end result on the model as shipped:
**22.8k to 54.6k tokens per second, 4.80% to 11.51% MFU, on the same 8 GPUs.**

**2.4x the MFU and 2.4x the tokens per second on the model people actually train.** The full model
gains less than the 12-layer one because its attention schedule is 25 compressed and heavily
compressed layers against 10, and those carry the compressor and indexer work that the levers here
do not touch.

**Where the remaining time goes**, from the final 12-layer trace:

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
