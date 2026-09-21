# From "it does not run" to a tuned training step

**Moonlight-V4-16B-A3B on one node of 8 H100s.** 16.5B parameters, 3.0B active per token, 64 experts
per layer, sparse attention.

We start where most people start: a correct, published implementation that cannot train this model
at all. We finish 2.4x faster, having changed no mathematics.

| | tokens/s | MFU | peak memory |
| --- | ---: | ---: | ---: |
| stock `transformers` | does not run | - | out of memory |
| a training-ready implementation | 22.8k | 4.80% | 60.5 GB |
| everything in this deck | **54.6k** | **11.51%** | 37.1 GB |

**The shape of the talk.**

- **Slides 1-4: the setup.** What the model is, what MFU measures, why the stock implementation runs
  out of memory, and how a kernel trace tells you what to fix.
- **Slides 5-9: the ladder.** One model, one fixed shape, one change per row, six rows. This is the
  spine: 6.08% to 10.59% MFU with everything else held constant.
- **Slides 10-11: beyond the ladder.** Batch shape, which is worth more than any single kernel, and
  whether any of it transfers to the full-size model.
- **Slide 12: what to take away.** Five ideas that outlive this model.
- **Appendices A-C: three things that did not work.** A kernel that was 2x in isolation and useless
  in context, a memory optimisation that cost time, and a profile that pointed at the wrong line of
  code. They took as long as the successes.

Every number is measured on the same hardware with the same recipe. Nothing is projected.
