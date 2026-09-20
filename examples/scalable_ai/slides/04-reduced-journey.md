# The same 4-layer model through four implementations

Sequence 2048, global batch 256 sequences, 524k tokens per optimizer step, 8x H100.

| stage | micro-batch | step time | tokens/s | peak memory | MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock `transformers` | 1 | 4.36 s | 120k | 22.0 GB | 6.2% |
| stock `transformers` | 4 | 2.58 s | 203k | 43.6 GB | 10.5% |
| stock `transformers` | 8 | out of memory | - | > 79 GB | - |
| Automodel, eager attention, torch dispatcher | 1 | 3.35 s | 157k | 15.7 GB | 8.1% |
| Automodel, eager attention, torch dispatcher | 4 | 2.56 s | 205k | 37.2 GB | 10.6% |
| Automodel, eager + DeepEP | 1 | 3.05 s | 172k | 15.7 GB | 8.9% |
| Automodel, eager + DeepEP | 4 | 2.26 s | 232k | 37.2 GB | 12.0% |
| Automodel, TileLang + DeepEP | 1 | 2.49 s | 210k | 14.2 GB | 10.9% |
| Automodel, TileLang + DeepEP | 4 | 1.76 s | 297k | 30.2 GB | 15.3% |
| Automodel, TileLang + DeepEP | 8 | 1.71 s | 308k | 52.4 GB | 15.9% |

**Read it as memory first, speed second.** Stock needs 22 GB where Automodel needs 14 to 16 GB for
identical work, and stock cannot fit micro-batch 8 at all. Only after the memory gap opens does the
speed gap follow: at micro-batch 4 the portable Automodel path merely matches stock, and the win
comes from the kernels layered on top.

On the full 27-layer model the same steps are the difference between "does not run" and
50k tokens per second.
