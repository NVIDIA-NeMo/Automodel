# What we measure, and why it is MFU

**Setup.** 8x H100 80GB, one node. FSDP2 with expert parallelism 8. Random weights, synthetic data,
12 steps of which 4 are warm-up. We report the mean step time and rank 0's peak memory.

**Model FLOPs Utilisation** is the arithmetic the model *needs* divided by elapsed time and by the
GPU's peak rate, 989 TFLOP/s for H100 in bf16.

> MFU = useful FLOPs / (step time x 989 TFLOP/s x 8 GPUs)

Two properties make it the right yardstick here:

- **It is shape-independent.** Tokens per second rises if you shorten the model. MFU does not.
- **It does not reward wasted work.** Attention is credited as *sparse*: a sliding-window layer earns
  only the window it is supposed to look at. An implementation that computes dense attention and
  throws most of it away pays the time and earns nothing for it. Remember this on slide 6.

**Where the useful work actually is**, for the model as configured:

| component | share of useful FLOPs |
| --- | ---: |
| expert matrix multiplications | 52% |
| vocabulary projection | 24% |
| attention projections | 16% |
| everything else, including all attention scoring | 8% |

**76% of the work is large matrix multiplications**, which an H100 runs at 40 to 60% of peak. So the
headroom is real. Anything that is not a large matrix multiplication is overhead to be minimised.
