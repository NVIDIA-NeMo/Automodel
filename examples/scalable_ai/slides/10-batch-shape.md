# The lever that is not a kernel

Every row of the ladder held the batch shape fixed, to keep the comparison honest. Now let us spend
the 16 GB it freed.

**Gradient accumulation** splits one optimizer step into several smaller forward and backward passes,
so you can train with a large effective batch on limited memory. It is not free. The original
baseline ran micro-batch 1 with **32 accumulation steps**: the parameters were gathered from across
the GPUs 32 times per step, and each expert's matrix multiplication saw roughly 190 tokens. Tensor
cores idle at that size.

Same code, same model, only the amount of work handed to the GPU changes:

| micro-batch per GPU | accumulation steps | step time | tokens/s | MFU | peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 19.75 s | 26.5k | 4.8% | 58.8 GB |
| 2 | 1 | 0.330 s | 99.2k | 10.59% | 27.4 GB |
| 4 | 1 | 0.476 s | 137.7k | 14.71% | 44.3 GB |
| 6 | 1 | 0.641 s | **153.4k** | **16.37%** | 62.7 GB |
| 8 | 1 | out of memory | - | - | > 79 GB |

**From 4.8% to 16.4% without touching a line of model code.** This is worth more than every kernel
change in the ladder put together.

**Notice where it stops: memory, not diminishing returns.** Micro-batch 6 is still improving on 4;
micro-batch 8 simply does not fit. That matters, because it means **the optimisations compound**. The
ladder cut memory by 38%, and that saved memory is what lets you run micro-batch 6 at all.

**Transferable lesson.** Before writing a kernel, check the shape you are feeding the hardware. It is
a free parameter, it changes arithmetic intensity, and it is usually the largest single lever. Then
notice that memory optimisations are throughput optimisations one step removed.
