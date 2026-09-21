# The ladder

One model, one shape, one change per row. **12 layers, sequence 2048, micro-batch 2 per GPU, no
gradient accumulation, 8x H100.** Only the thing named in each row changes.

| # | change | step time | tokens/s | MFU | peak memory |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | baseline: training-ready implementation | 0.576 s | 56.9k | 6.08% | 43.9 GB |
| 2 | expert communication and grouped matrix multiplication | 0.505 s | 64.9k | 6.92% | 42.4 GB |
| 3 | sparse attention kernels | 0.388 s | 84.5k | 9.01% | 31.3 GB |
| 4 | fuse the hyper-connections | 0.355 s | 92.3k | 9.87% | 28.3 GB |
| 5 | bf16 gradient reduction | 0.356 s | 92.0k | 9.83% | 28.3 GB |
| 6 | bf16 vocabulary projection | 0.330 s | 99.2k | **10.59%** | 27.4 GB |

**1.74x faster and 16 GB lighter**, on a model the stock implementation could not train at all.

Two things to notice before we go through them one at a time.

**Row 5 did nothing.** The change that the profile said was the single largest kernel is worth
nothing at this shape. That is not a mistake in the profile or in the change. Slide 9 explains it,
and it is the most transferable lesson in the deck.

**Memory falls faster than time.** 43.9 GB to 27.4 GB, a 38% reduction. Memory is what decides
whether you can use a bigger batch, and slide 10 turns that saved memory back into speed.
