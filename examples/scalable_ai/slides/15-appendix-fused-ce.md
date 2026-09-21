# Appendix C: a memory optimisation that cost time

The final loss computation produces a score for all 163,840 vocabulary entries for every token: a
4 GB tensor at this batch size. There is a well-known technique that avoids ever building it, by
computing the loss in chunks and recomputing what it needs in the backward pass.

| | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| build the full tensor | 0.514 s | 13.60% | 44.3 GB |
| chunked, never build it | 0.580 s | 12.06% | 35.4 GB |

**9 GB cheaper and 13% slower.** The trace is unambiguous: the recomputation in the backward pass
costs 85 ms, 16% of the step, which is more than the memory traffic it avoids.

It stays available, documented as:

> Swap it in when memory, not step time, is the binding constraint.

**Lesson.** "Avoids materialising a large tensor" sounds unambiguously good and is not. A memory
optimisation is a speed optimisation only when memory is what binds. Here, after the earlier work
had already freed 16 GB, it no longer was.

And note the interaction: this technique only became *possible* because of the bf16 vocabulary
projection in row 6, which put the projection and the hidden states in the same dtype. Optimisations
enable and obsolete each other, which is another reason to re-measure rather than accumulate a list.
