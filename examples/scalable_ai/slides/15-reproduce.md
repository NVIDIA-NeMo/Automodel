# Reproducing this

**Configs**, in `examples/scalable_ai/configs/`:

| file | what it is |
| --- | --- |
| `moonlight_v4_16b_hf.yaml` | stage 0, stock `transformers` |
| `moonlight_v4_16b_torch.yaml` | portable Automodel: eager attention, torch dispatcher |
| `moonlight_v4_16b_tilelang_deepep.yaml` | Hopper kernels: TileLang attention, DeepEP |
| `moonlight_v4_16b_mfu.yaml` | everything in this deck, stacked |

**Runner.** `examples/scalable_ai/run_bench.sh` drives all of it and expects `$WORK` to hold
`Automodel/`, `models/Moonlight-V4-16B-A3B/` and `logs/`.

```bash
run_bench.sh journey    # the 4-layer stock-to-optimised comparison (slide 4)
run_bench.sh moe_ab     # the MoE A/B (slide 7)
run_bench.sh one        # a single run, driven by ONE_CFG / ONE_ARGS
```

**Flags added for this work**, all defaulting to off so nothing changes silently:

| flag | effect |
| --- | --- |
| `backend.compile_hc` | `torch.compile` the mHC mixer, collapse and expand cores |
| `backend.hc_proj_bf16` | mixer projection in bf16 on cuBLAS |
| `backend.hc_proj_kernel` | mixer projection through the fused Triton kernel |
| `backend.lm_head_bf16` | vocabulary projection in bf16 instead of fp32 |
| `benchmark.torch_profile_start/end/dir` | capture a kernel trace of chosen steps |

**Profiling.** `scripts/analyze_trace.py` turns a chrome trace into the category tables used
throughout this deck.

**Operational notes.** Inside the container, point `TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR` and
`TORCHINDUCTOR_CACHE_DIR` at writable storage; a failed TileLang import makes Automodel fall back to
the `transformers` class, which then rejects the `backend` argument. A cold TileLang cache costs
about 15 minutes on the first run.
