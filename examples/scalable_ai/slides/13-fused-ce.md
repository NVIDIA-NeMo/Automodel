# An optimisation that did not pay: fused cross-entropy

Once `lm_head` is bf16, `FusedLinearCrossEntropy` becomes usable: it consumes the final hidden
states and never materialises the `[tokens, 163840]` logits tensor, which is 4 GB at micro-batch 4.

| 12 layers, micro-batch 4 | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| logits-based `MaskedCrossEntropy` | 0.514 s | 13.60% | 44.3 GB |
| `FusedLinearCrossEntropy` | 0.580 s | 12.06% | 35.4 GB |

**9 GB cheaper and 13% slower.** The trace explains it: `_cce_backward_kernel` alone costs 85 ms,
16% of the step. Recomputing the logits in the backward pass costs more than the memory traffic it
avoids at this vocabulary size and batch shape.

So it stays in the config as a documented alternative rather than a default:

> Swap it in when memory, not step time, is the binding constraint.

**The general point.** Every change in this deck was measured, and this one is why. "Avoids
materialising a large tensor" sounds unambiguously good and is not. A memory optimisation is a
time optimisation only when memory is what binds.
