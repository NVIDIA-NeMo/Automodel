# From stock Transformers to a tuned training step

**Moonlight-V4-16B-A3B on one 8x H100 node**

A DeepSeek-V4-architecture model at Moonlight scale: 16.5B total parameters, 3.0B active,
27 layers, 64 experts with top-6 routing plus one shared expert.

The journey, measured end to end. Two tables because the shape changes: the first three stages run
the **full 27-layer model**, which is where the baseline fails, and the rest run a **12-layer model**
so that every later step is compared against something that actually runs.

## Stages 0-2: full model, 27 layers, sequence 2048, micro-batch 1, 32 accumulation steps

| # | change | parallelism | attention kernel | MoE kernels | step | MFU |
| ---: | --- | --- | --- | --- | ---: | ---: |
| 0 | stock `transformers` | FSDP2 dp8, **ep1** | eager, dense then mask | HF module list, no EP | does not run | - |
| 1 | Automodel native model | FSDP2 dp8, **ep8** | eager | per-expert loop, all-gather dispatch | 23.02 s | 4.80% |
| 2 | DeepEP + grouped GEMM | FSDP2 dp8, ep8 | eager | `grouped_gemm`, DeepEP dispatch | 20.09 s | 5.50% |

## Stages 3-9: 12 layers, sequence 2048, one accumulation step, FSDP2 dp8 + ep8

| # | change | micro-batch | attention kernel | MoE and fusions | step | MFU |
| ---: | --- | ---: | --- | --- | ---: | ---: |
| 3 | batch shape | 3 | eager | `gmm` + DeepEP | 0.658 s | 7.98% |
| 4 | sparse attention | 4 | **TileLang** sparse + indexer + Sinkhorn | `gmm` + DeepEP | 0.599 s | 11.69% |
| 5 | fuse hyper-connections | 4 | TileLang | + `compile_hc` | 0.561 s | 12.47% |
| 6 | bf16 gradient reduction | 4 | TileLang | + bf16 ReduceScatter | 0.554 s | 12.64% |
| 7 | bf16 projections | 4 | TileLang | + bf16 vocabulary and mixer projections | 0.514 s | 13.60% |
| 8 | hand-written mixer kernel | 4 | TileLang | + fused Triton norm-projection | 0.515 s | 13.58% |
| 9 | fused stream mix | 4 | TileLang | + 4-way mix as fused multiply-adds | 0.483 s | **14.48%** |

Row 7 was measured without the gradient-reduction change and row 8 with it, which is why they agree
to within run-to-run noise; row 9 has everything. Sequence length is 2048 throughout. It is not a
lever once attention is sparse: at stage 4, sequence 4096 with micro-batch 2 measured 11.67%, the
same as sequence 2048 with micro-batch 4 for the same token count.

**3.0x, and 19 GB less memory**, on a model the stock implementation could not train at all.
On the full 27-layer model the same configuration is worth **2.4x**: 4.80% to 11.51% MFU,
22.8k to 54.6k tokens per second.

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
