# From a narrow corner to a tuned training step

A guest-lecture deck on making a large mixture-of-experts model train efficiently. One markdown file
per slide; render with any markdown-to-slides tool, or read in order.

**Subject.** Moonlight-V4-16B-A3B, a DeepSeek-V4-architecture model, on one node of 8 H100s.

| | slide | subject |
| --- | --- | --- |
| | `00-overview.md` | the result and the shape of the talk |
| **Setup** | `01-the-model.md` | the model, and the three things about it that matter |
| | `02-what-we-measure.md` | MFU, and where the useful work actually is |
| | `03-the-narrow-corner.md` | where the stock implementation fits, and why the original run did not |
| | `04-how-we-find-it.md` | reading a kernel trace |
| **The ladder** | `05-the-ladder.md` | **the spine**: one shape, one change per row |
| | `06-lever-sparsity.md` | sparsity has to be real |
| | `07-lever-communication.md` | move tokens, not weights |
| | `08-lever-fusion.md` | a thousand small kernels |
| | `09-lever-precision.md` | precision, and a lever that did nothing |
| **Beyond** | `10-batch-shape.md` | the lever that is not a kernel |
| | `11-does-it-transfer.md` | the same changes on the full 27-layer model |
| | `12-lessons.md` | five transferable ideas |
| **Appendix** | `13-appendix-profile-lied.md` | the profile pointed at the wrong line |
| | `14-appendix-kernel.md` | a kernel that was 2x and bought nothing |
| | `15-appendix-fused-ce.md` | a memory optimisation that cost time |
| | `16-reproduce.md` | configs, commands, flags, gotchas |

**Teaching notes.**

- The spine is slide 5. Slides 6 to 9 each expand one of its rows and can be cut for time; the table
  still tells the story without them.
- Row 5 of the ladder is deliberately a change that measured zero. Slide 9 explains why, and it is
  the single most transferable point in the deck: a profile describes one configuration, not a
  program.
- Slide 10 is the one to keep if you only have time for two slides: the shape lever is worth more
  than every code change in the deck, and it is the one the audience can act on tomorrow.
- The appendices are self-contained. If the audience is enjoying the failures, A is the best one.
- Every measurement: 8x H100 80GB, NeMo Automodel 26.08 container, FSDP2 with expert parallelism 8,
  random weights and synthetic data, 12 steps of which 4 are warm-up.
