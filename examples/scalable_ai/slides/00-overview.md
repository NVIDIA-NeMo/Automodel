# From stock Transformers to a tuned training step

**Moonlight-V4-16B-A3B on one 8x H100 node**

A DeepSeek-V4-architecture model at Moonlight scale: 16.5B total parameters, 3.0B active,
27 layers, 64 experts with top-6 routing plus one shared expert.

The journey, measured end to end. Sequence 2048 and FSDP2 across 8 GPUs throughout.

| # | change | layers | mb | accum | EP | attention kernel | MoE and fusions | step | tok/s | MFU |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 0 | stock `transformers` | 27 | 1 | 32 | **1** | eager, dense then mask | HF module list | does not run | - | - |
| 1 | Automodel native model | 27 | 1 | 32 | 8 | eager | loop experts, all-gather | 23.02 s | 22.8k | 4.80% |
| 2 | grouped GEMM + DeepEP | 27 | 1 | 32 | 8 | eager | `gmm` + DeepEP | 20.09 s | 26.1k | 5.50% |
| 3 | batch shape | 12 | 3 | 1 | 8 | eager | `gmm` + DeepEP | 0.658 s | 74.7k | 7.98% |
| 4 | sparse attention | 12 | 4 | 1 | 8 | **TileLang** sparse + indexer | `gmm` + DeepEP | 0.599 s | 109.4k | 11.69% |
| 5 | fuse hyper-connections | 12 | 4 | 1 | 8 | TileLang | + `compile_hc` | 0.561 s | 116.8k | 12.47% |
| 6 | bf16 gradient reduction | 12 | 4 | 1 | 8 | TileLang | + bf16 ReduceScatter | 0.554 s | 118.3k | 12.64% |
| 7 | bf16 projections | 12 | 4 | 1 | 8 | TileLang | + bf16 vocabulary and mixer | 0.514 s | 127.5k | 13.60% |
| 8 | hand-written mixer kernel | 12 | 4 | 1 | 8 | TileLang | + fused Triton norm-projection | 0.515 s | 127.3k | 13.58% |
| 9 | fused stream mix | 12 | 4 | 1 | 8 | TileLang | + 4-way mix as multiply-adds | 0.483 s | 135.7k | **14.48%** |
| 10 | everything applied | 27 | 3 | 1 | 8 | TileLang | all of the above | 0.900 s | **54.6k** | **11.51%** |

`mb` is the micro-batch per GPU, `accum` the gradient accumulation steps, `EP` the expert-parallel
size. Rows 1-2 and 10 are the real 27-layer model; rows 3-9 use 12 layers so that every step is
measured against a baseline that runs.

**Read MFU across the whole table, not tokens per second.** Throughput jumps at row 3 mostly because
the model got shorter, and drops at row 10 because it got long again. MFU is normalised for that.
The comparison that matters end to end is row 1 against row 10, same model, same hardware:
**22.8k to 54.6k tokens per second, 4.80% to 11.51% MFU.**

**2.4x on the full model and 3.0x on the 12-layer one**, with peak memory down from 60.5 GB to
41.3 GB, on a model the stock implementation could not train at any sequence length.

## Why stage 0 cannot run, and stage 1 can

Both run the same recipe on the same 8 GPUs with FSDP2, so this is not about sharding parameters.
It is about what each implementation does per layer.

**Stock `transformers` is written for inference.**

- Its compressed-sparse layers concatenate the compressed entries onto the key axis and run **one
  dense attention over the whole extended length**, applying sparsity only as a mask. Nothing is
  skipped, so a "sparse" layer costs more than a dense one.
- Attention is eager only: there is no FlashAttention for 512-dimensional heads and no SDPA path.
  The score matrix is therefore materialised, and the softmax runs in fp32.
- **336 MB of attention probabilities per layer** at micro-batch 1 and sequence 2048, kept for the
  backward pass. Across 27 layers that is 9.1 GB for the probabilities alone, before scores,
  activations or weights. Computing only the 128-token sliding window would need 17 MB.
- It cannot use expert parallelism. The expert-parallel sharding needs Automodel's own mixture-of-
  experts module structure, so the baseline runs with `ep_size: 1`.

**The Automodel native model is written for training.**

- Sparse attention is actually sparse: the TileLang path builds per-query top-k key indices and
  gathers only those keys, so the dense score matrix never exists.
- `ep_size: 8` puts one eighth of the 64 experts on each rank, and DeepEP moves tokens between
  ranks instead of gathering expert weights.
- Its FSDP2 unit layout accounts for V4's fp32 tensors, the attention sinks, compressor biases and
  hyper-connection mixers, which are separate units that must stay resident across the backward pass.

Measured: at 4 layers, stock needs 22.0 GB where the native model needs 15.7 GB for identical work.
At 27 layers stock exceeds 79 GB at every sequence length tried, while the native model fits in
58.8 GB and later, tuned, in 46.3 GB.

Every number here is a measurement on the same hardware with the same recipe, not a projection.
Two of the eight stages did not work as expected, and both are kept in the deck: stage 7 bought
memory but no time, and slide 13 covers an optimisation that was slower than what it replaced.

Deck order: `00` overview, `01`-`03` setup and baseline, `04` the like-for-like comparison,
`05` profiling method, `06`-`12` one lever per slide, `13` a negative result, `14` scoreboard,
`15` reproduction.
