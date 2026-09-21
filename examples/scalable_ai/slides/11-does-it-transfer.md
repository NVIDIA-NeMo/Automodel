# Does it transfer to the real model?

The ladder used 12 layers so that every stage could be compared against a baseline that runs. The
model people actually train has 27. Same changes, full model:

**Like for like, at the baseline's exact shape** (micro-batch 1, 32 accumulation steps):

| | step time | tokens/s | MFU | peak memory |
| --- | ---: | ---: | ---: | ---: |
| baseline | 23.02 s | 22.8k | 4.80% | 60.5 GB |
| all changes | 13.77 s | 38.1k | 8.02% | 37.1 GB |

**1.67x, and 23 GB freed.** No shape tricks; the only difference is the changes in this deck.

**Then spend the freed memory**, which the baseline could not afford:

| | step time | tokens/s | MFU |
| --- | ---: | ---: | ---: |
| all changes, micro-batch 3 | 0.900 s | 54.6k | **11.51%** |

**2.4x end to end on the full model.**

**Why less than the 12-layer ladder?** The full model has 25 compressed-attention layers against 10,
and those carry the compressor and indexer work that none of these changes touch. The optimisations
target what is common to all layers; the more a model does that you have not optimised, the more your
speedup is diluted.

**Transferable lesson.** Measure on a proxy to iterate quickly, but always close the loop on the real
thing. Proxies flatter you in ways that are specific to how you shrank them.
