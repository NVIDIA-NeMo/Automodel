# From stock Transformers to a tuned training step

**Moonlight-V4-16B-A3B on one 8x H100 node**

A DeepSeek-V4-architecture model at Moonlight scale: 16.5B total parameters, 3.0B active,
27 layers, 64 experts with top-6 routing plus one shared expert.

The journey, measured end to end:

| stage | what changed | MFU |
| --- | --- | ---: |
| 0 | stock `transformers`, out of the box | does not run |
| 1 | NeMo Automodel native model | 4.8% |
| 2 | expert-parallel dispatch and grouped GEMMs | 5.5% |
| 3 | batch shape | 8.0% |
| 4 | sparse attention kernels | 11.7% |
| 5 | fusing the hyper-connections | 12.5% |
| 6 | precision where it is free | 13.6% |
| 7 | a hand-written kernel | 13.6% |
| 8 | fixing what the profile actually said | **14.5%** |

**3.0x, and 19 GB less memory**, on a model the stock implementation could not train at all.

Every number here is a measurement on the same hardware with the same recipe, not a projection.
Two of the eight stages did not work as expected, and both are kept in the deck: stage 7 bought
memory but no time, and slide 13 covers an optimisation that was slower than what it replaced.

Deck order: `00` overview, `01`-`03` setup and baseline, `04` the like-for-like comparison,
`05` profiling method, `06`-`12` one lever per slide, `13` a negative result, `14` scoreboard,
`15` reproduction.
