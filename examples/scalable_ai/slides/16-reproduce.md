# Reproducing this

**Configs**, in `examples/scalable_ai/configs/`:

| file | what it is |
| --- | --- |
| `moonlight_v4_16b_hf.yaml` | the starting point, stock `transformers` |
| `moonlight_v4_16b_torch.yaml` | portable: eager attention, simple expert dispatch |
| `moonlight_v4_16b_tilelang_deepep.yaml` | sparse attention kernels and routed dispatch |
| `moonlight_v4_16b_mfu.yaml` | everything in this deck |

**Runner.** `examples/scalable_ai/run_bench.sh`, expecting `$WORK` to hold `Automodel/`,
`models/Moonlight-V4-16B-A3B/` and `logs/`.

```bash
run_bench.sh journey    # stock vs optimised on a 4-layer model, the only shape stock survives
run_bench.sh moe_ab     # the expert-communication comparison, slide 7
run_bench.sh one        # a single run, driven by ONE_CFG and ONE_ARGS
```

**The ladder on slide 5**, one shape, one change per row:

```bash
SHAPE="--model.config.num_hidden_layers 12 --step_scheduler.local_batch_size 2 \
       --step_scheduler.global_batch_size 16 --dataset.seq_len 2048"
ONE_CFG=moonlight_v4_16b_torch.yaml ONE_ARGS="$SHAPE --model.backend.experts torch \
       --model.backend.dispatcher torch" run_bench.sh one          # row 1
```

**Flags added for this work**, all off by default:

| flag | effect | slide |
| --- | --- | --- |
| `backend.compile_hc` | fuse the hyper-connection cores | 8 |
| `backend.lm_head_bf16` | vocabulary projection in bf16 | 9 |
| `backend.hc_proj_bf16` | mixer projection in bf16 | 9 |
| `backend.hc_proj_kernel` | mixer projection on the hand-written kernel | appendix B |
| `benchmark.torch_profile_start/end/dir` | capture a kernel trace | 4 |

**Profiling.** `scripts/analyze_trace.py` turns a trace into the category tables in this deck; add
`--shapes` for the operator attribution that appendix A needed.

**Gotchas.** Inside a container, point the TileLang, Triton and Inductor cache directories at
writable storage. A cold TileLang cache costs about 15 minutes on the first run. If the TileLang
import fails, the framework silently falls back to the stock model class and then rejects its own
arguments, which is a confusing way to learn that a cache directory was read-only.
