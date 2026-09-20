# Slides: from stock Transformers to a tuned DeepSeek-V4 training step

One markdown file per slide. Render with any markdown-to-slides tool, or read in order.

| slide | subject |
| --- | --- |
| `00-overview.md` | the journey in one table |
| `01-what-v4-adds.md` | what DeepSeek-V4 adds over V3, and why it is hard to make fast |
| `02-what-mfu-counts.md` | measurement setup, what MFU credits, the FLOPs budget |
| `03-stage0-stock.md` | stock `transformers` out of the box: does not run |
| `04-reduced-journey.md` | four implementations on the same 4-layer model |
| `05-profile-first.md` | the kernel trace that drove every decision after it |
| `06-lever-attention.md` | sparse attention instead of dense-and-mask (1.35x) |
| `07-lever-moe.md` | DeepEP dispatch and grouped expert GEMMs (1.15x) |
| `08-lever-batch.md` | batch shape as a kernel-efficiency knob, and where it saturates |
| `09-lever-fusion.md` | compiling the hyper-connection cores (1.07x, 6 GB) |
| `10-lever-precision.md` | bf16 gradient reduction and vocabulary projection |
| `11-lever-kernel.md` | writing a Triton kernel: 2.1x in isolation, 0 in the step |
| `12-misattribution.md` | the hot kernel was not the one I thought |
| `13-fused-ce.md` | an optimisation that did not pay |
| `14-scoreboard.md` | final numbers and the honest gap to 20% |
| `15-reproduce.md` | configs, runner, flags, gotchas |

Suggested narrative arc: stages 0 to 4 are about **removing waste** and are where most of the gain
is. Stages 5 to 9 are about **fusion and precision**, with smaller returns and two instructive
failures. The closing slide is deliberately not a victory lap.

All measurements: 8x H100 80GB, NeMo Automodel 26.08 container, FSDP2 with expert parallelism 8,
random init on mock data, 12 steps with 4 warm-up.
