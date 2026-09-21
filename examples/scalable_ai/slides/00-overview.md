# From a narrow corner to a tuned training step

**Moonlight-V4-16B-A3B on one node of 8 H100s.** 16.5B parameters, 3.0B active per token, 64 experts
per layer, sparse attention.

We start where most people start: the published implementation. The only full-model point
that completed in our sweep used micro-batch 1 with no gradient accumulation, reaching 3.67% MFU.
We finish with 3.2x its throughput, without changing the model architecture.

| full 27-layer model | batch shape per GPU | tokens/s | MFU | peak memory |
| --- | --- | ---: | ---: | ---: |
| stock `transformers`, FSDP2 only | micro-batch 1, accumulation 1 | 17.4k | 3.67% | 45.6 GB |
| training-ready implementation | micro-batch 1, accumulation 32 | 22.8k | 4.80% | 60.5 GB |
| everything in this deck | micro-batch 3, accumulation 1 | **56.0k** | **11.81%** | 67.2 GB |

Within the training-ready path, half of the gain comes from code changes and half from handing the
GPU more work at a time. The two compound: the code changes free the memory that the larger batch
needs.

The first and last rows differ in both implementation and batch shape; they are endpoints, not an
ablation. Slides 5 and 11 hold the shape fixed to isolate the code changes.

**The shape of the talk.**

- **Slides 1-4: the setup.** What the model is, what MFU measures, why the stock implementation fits
  only in a narrow corner, and how a kernel trace tells you what to fix.
- **Slides 5-9: the ladder.** One model, one fixed shape, one change per row, six rows. This is the
  spine: 6.08% to 10.59% MFU with everything else held constant.
- **Slides 10-11: beyond the ladder.** Batch shape, which is worth more than every kernel change
  combined, and whether any of it transfers to the full-size model.
- **Slide 12: what to take away.** Five ideas that outlive this model.
- **Appendices A-C: three things that did not work.** A kernel that was 2x in isolation and useless
  in context, a memory optimisation that cost time, and a profile that pointed at the wrong line of
  code. They took as long as the successes.

Every number is measured on the same hardware with the same recipe. Nothing is projected.
