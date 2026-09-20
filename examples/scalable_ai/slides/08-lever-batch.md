# Lever 3: batch shape is a kernel-efficiency knob

The original full-model baseline ran micro-batch 1 against a global batch of 256 sequences, which
means **32 gradient accumulation steps**. Two consequences:

- Every expert GEMM sees roughly 190 tokens. Tensor cores idle.
- FSDP re-gathers all parameters 32 times per optimizer step.

Raising the micro-batch so accumulation drops to 1, holding the backend fixed at eager attention with
grouped GEMM and DeepEP:

| 12 layers, eager attention | step time | MFU |
| --- | ---: | ---: |
| micro-batch 2 | 0.506 s | 6.91% |
| micro-batch 3 | 0.658 s | 7.98% |

And it saturates. Once the other levers are in place, going past micro-batch 4 costs more than it
returns:

| 12 layers, tuned backend | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| micro-batch 4 | 0.554 s | 12.64% | 48.1 GB |
| micro-batch 6 | 0.849 s | 12.37% | 68.1 GB |

The GEMMs are already large enough by then, and the extra activation memory buys nothing.
**Micro-batch 4 is the operating point** for the rest of the deck.

Worth noting what did *not* move: the ratio between the two MoE backends stayed at 1.16x across a
32x change in accumulation count. A ratio that constant across a change that large means the
bottleneck is somewhere else entirely, which is what sent us to the profiler.
