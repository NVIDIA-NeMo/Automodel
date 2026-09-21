# Appendix B: a kernel that was 2x and bought nothing

The diagnosis in appendix A sent me to write a kernel for the mixer projection: a very wide input
times a 24-column matrix, which no library handles well because there is only one tile of work.

**The kernel.** Split the reduction across many blocks instead of one, so the grid scales, and fuse
the normalisation into the same pass so the input is read once rather than three times. Four Triton
kernels: the statistic, the forward projection, and two for the backward pass.

**The detail worth keeping.** The first backward version was *slower than the code it replaced*.
Each block handled one row and re-read the entire weight matrix: 8.6 GB of redundant reads. Blocking
over 32 rows amortised that and turned no speedup into **2.1x** on a microbenchmark.

Correctness against the original: forward agrees to 2e-4 relative, weight gradients to 8e-4, input
gradients to 6e-3, which is the rounding noise of the bf16 they are stored in anyway.

**In the actual training step:**

| | step time | MFU | peak memory |
| --- | ---: | ---: | ---: |
| library call, bf16 | 0.477 s | 14.66% | 44.3 GB |
| the hand-written kernel | 0.482 s | 14.53% | 41.3 GB |

**2.1x in isolation. 1% slower in context, 3 GB leaner.** It is off by default and documented as the
choice when memory is the binding constraint, not speed.

**Lesson.** A microbenchmark measures a kernel in an empty machine. In a real step that kernel
overlaps with communication and other work, so removing its time removes nothing. Speed up the
critical path, and know which path that is before you start.
