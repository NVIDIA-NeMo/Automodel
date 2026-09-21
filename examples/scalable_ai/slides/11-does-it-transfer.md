# Does it transfer to the real model?

The ladder used 12 layers so every stage had a baseline that runs. The model people train has 27.

**Like for like, at the baseline's exact shape** (micro-batch 1, 32 accumulation steps), so the only
difference is the code:

| full model | step time | tokens/s | MFU | peak memory |
| --- | ---: | ---: | ---: | ---: |
| baseline | 23.02 s | 22.8k | 4.80% | 60.5 GB |
| all changes | 13.77 s | 38.1k | 8.02% | 37.1 GB |

**1.67x, and 23 GB freed.** Now spend that memory, exactly as on the previous slide:

| full model, no accumulation | step time | tokens/s | MFU | peak memory |
| ---: | ---: | ---: | ---: | ---: |
| micro-batch 1 | 0.626 s | 26.2k | 5.51% | 32.4 GB |
| micro-batch 2 | 0.729 s | 44.9k | 9.48% | 49.7 GB |
| micro-batch 3 | 0.877 s | 56.0k | **11.81%** | 67.2 GB |
| micro-batch 4 | out of memory | - | - | > 79 GB |

**2.5x end to end: 4.80% to 11.81%, 22.8k to 56.0k tokens per second.**

Two observations worth drawing out.

**Even fully optimised, micro-batch 1 reaches only 5.51%.** The code is the same as the row below it.
The GPU simply has too little to do. If you take one number from this deck, take this pair.

**The full model gains less than the proxy**, 2.5x against the ladder's larger figure. It has 25
compressed-attention layers where the proxy has 10, and those carry compressor and indexer work that
none of these changes touch. The more a model does that you have not optimised, the more your
speedup is diluted.

**Transferable lesson.** Iterate on a proxy, but always close the loop on the real thing. Proxies
flatter you in ways specific to how you shrank them.
