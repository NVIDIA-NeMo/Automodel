# The lever that is not a kernel

Every row so far held the batch shape fixed, to keep the comparison honest. Now let us spend the
16 GB the ladder freed.

**Gradient accumulation** splits an optimizer step into several smaller forward and backward passes.
It is how you train with a large effective batch on limited memory, and it is not free: the original
baseline ran micro-batch 1 with **32 accumulation steps**, which means the parameters were gathered
from across the GPUs 32 times per step, and every expert's matrix multiplication saw about 190
tokens. Tensor cores idle at that size.

Same code, same model, only the shape changes:

| micro-batch per GPU | accumulation steps | step time | MFU |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 19.75 s | 4.8% |
| 2 | 1 | 0.330 s | 10.59% |
| 4 | 1 | 0.477 s | **14.66%** |
| 6 | 1 | 0.849 s | 12.37% |

**And it saturates.** Micro-batch 6 is worse than 4. The matrix multiplications are already large
enough to keep the hardware busy, and the extra activation memory buys nothing.

**Transferable lesson.** Before writing a kernel, check the shape you are feeding the hardware.
Batch size is a free parameter that changes arithmetic intensity, and here it is worth more than any
individual kernel change in this deck. But only up to a point, which you find by measuring, not by
reasoning.
