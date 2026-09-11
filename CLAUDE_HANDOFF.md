# Handoff — Qwen3.6-35B-A3B SFT on `vuhaian/v4_88k` (2 × B300 RunPod)

Written 2026-09-11 ~09:20 by Claude Code (session `dc22bf29-146a-4a63-8910-37a0e87d9eb4`).
Everything a fresh session (or a human) needs to continue without re-deriving anything.

- **Restore my memory + transcript on a new pod:** `bash /workspace/claude_handoff/restore.sh`
- **Refresh the backup before shutting down:** `bash /workspace/claude_handoff/backup.sh`
- `/workspace` is a local RAID (`/dev/md1`): it survives a pod stop/start, **not** pod
  termination/migration. Push the branch and/or download `/workspace/claude_handoff/` +
  this file if the pod may be replaced.

---

## 0. Current status — BLOCKED on hardware

**GPU 0 (PCI `00000000:9A:00.0`) has a hardware fault.** `nvidia-smi -q` shows
`GPU Recovery Action: Reset`, `Channel Repair Pending` / `TPC Repair Pending: GPU requires reset`,
5 uncorrectable retired pages pending, and **all NVLinks on GPU 0 inactive**
(`nvidia-smi nvlink -s`). GPU 1 is healthy. A DDP probe crashed with
`CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error` during
checkpoint load. NVLink worked earlier the same day (FSDP2 + DeepEP runs), so the fault
appeared around 08:30–09:10.

Needs a GPU reset (host-side) or a new pod. **Before any training on the fixed/new pod:**

```bash
nvidia-smi -q -d ECC,ROW_REMAPPER | grep -iE 'pending|uncorrectable'   # no pending repairs
nvidia-smi nvlink -s                                                     # all links active on both GPUs
. /workspace/automodel_env.sh
cd /workspace/claude_handoff/scratch/DeepEP/tests && python test_intranode.py --num-processes 2   # ~1 min, must pass
# (clone of github.com/deepseek-ai/DeepEP at rev 42144303, the uv.lock pin)
```

Nothing is running. Nothing is committed.

---

## 1. Goal and hard requirements

Full-parameter SFT of `Qwen/Qwen3.6-35B-A3B` (VLM, frozen vision tower; text backbone =
30 GatedDeltaNet linear-attn + 10 full-attn layers, 256 experts / 8 active, head_dim 256,
vocab 248,320, 1 MTP layer) on the private turn-exploded corpus `vuhaian/v4_88k`
(87,552 rows / 9,835 trajectories). Runbook:
`Automodel/examples/vlm_finetune/qwen3_5_moe/RUNBOOK_v4_88k.md` (written for 8 × H200;
updated during this session with B300 notes).

Requirements (never trade away): (1) supervise **only the final assistant turn**;
(2) **drop** over-length rows, never truncate; (3) exported checkpoint loads in plain
`transformers` as `Qwen3_5MoeForConditionalGeneration` (keep MTP).

---

## 2. User preferences and decisions (binding)

- **Masking:** the empty `<think>\n\n</think>\n\n` block that opens every final turn is
  **masked** (user decision; documented in the runbook). Model is trained for
  `enable_thinking=False` inference.
- **Optimizer:** `torchao.optim.AdamW8bit` (user request). Required a small fix in
  `components/optim/scheduler.py` (torchao keeps lr as a tensor; scheduler now `fill_()`s it).
- **Minimal changes to the codebase's attention mechanism.** A right-padding SDPA patch was
  reverted on request; performance must come from packages/config. → TE 2.18 + cuDNN 9.26.
- **Prefer predefined samplers/components**; a custom length-sorted sampler was reverted.
- **Learnability first**: flag anything that changes numerics (FP8/MXFP8, length-grouped
  batching that alters step composition). Don't enable silently.
- Measure one knob at a time; don't commit unless asked.

---

## 3. Environment (how it was built — reproduce in this order)

Machine: 2 × NVIDIA B300 SXM6 (SM 10.3, 268 GiB each), driver 580.126.09 (host, read-only
libs), Ubuntu 24.04, 344 logical CPUs, 4 TB RAM. Repo at `/workspace/Automodel`, branch
`trungvd-zenai/feat/qwen3-6-v4-88k-sft`. HF token in `/workspace/.env` (`HF_TOKEN=...`).

**Always:** `. /workspace/automodel_env.sh` (sets HF_TOKEN/HF_HOME=/workspace/hf,
UV_CACHE_DIR, CUDA 13.0 paths, strips toolkit lib64 from LD_LIBRARY_PATH, CUDNN_HOME +
venv cuDNN/NCCL first, TORCH_CUDA_ARCH_LIST="10.0;10.3", CPATH cccl, OMP_NUM_THREADS=64,
expandable_segments, activates `.venv`). Just activating the venv is NOT enough.

1. **CUDA 13.0 toolkit via apt** (`cuda-toolkit-13-0`; base image had 12.8) to match
   torch 2.10.0+cu130. `update-alternatives --set cuda /usr/local/cuda-13.0`.
   Also `rdma-core libibverbs-dev`.
   **Never** apt-install `libnvidia-ml-dev` (or anything pulling `libnvidia-compute-*` /
   `nvidia-persistenced`): it drags driver packages that fail in the container. CUDA 13's
   `cuda-nvml-dev-13-0` already provides `nvml.h`.
2. **uv sync in two phases** (torch must exist before the no-build-isolation builds):
   `uv sync --locked --extra vlm --extra vlm-media --extra fla`, then
   `uv sync --locked --extra vlm --extra vlm-media --extra moe`, then
   `uv sync --locked --inexact --extra vlm --extra vlm-media --extra moe --group dev`
   (dev group = `cut-cross-entropy`, needed by FusedLinearCrossEntropy).
   **Don't use `--all-groups`** (pulls `magi` → MagiAttention source build, unused).
3. **Outside `uv.lock`** (an exact `uv sync` reverts these — reinstall after any sync):
   - `uv pip install nvidia-ml-py` (DeepEP imports `pynvml`).
   - **TE 2.18.0 + cuDNN 9.26.0.51** (needed for TE fused attention at head_dim 256 on
     SM 10.x — TE PR #3056 needs cuDNN ≥ 9.23 for BSHD). Install from OUTSIDE the repo dir
     (pyproject's `[tool.uv]` build config breaks it otherwise):
     ```bash
     . /workspace/automodel_env.sh; cd /tmp
     uv pip install --python /workspace/Automodel/.venv/bin/python --no-deps \
       "nvidia-cudnn-cu13==9.26.0.51" "transformer-engine==2.18.0" "transformer-engine-cu13==2.18.0"
     NVTE_WITH_NCCL_EP=0 uv pip install --python /workspace/Automodel/.venv/bin/python --no-deps \
       --no-build-isolation-package transformer-engine-torch "transformer-engine-torch==2.18.0"
     ```
     (`NVTE_WITH_NCCL_EP=0`: TE's optional NCCL-EP code includes a header torch 2.10 lacks.)
     Rollback list: `claude_handoff/scratch/venv_freeze_before_te218.txt` (TE 2.15, cuDNN 9.15.1.9).
4. Weights: `hf download Qwen/Qwen3.6-35B-A3B` (HF_HOME=/workspace/hf, 67 GB, snapshot 995ad96e).
5. Data: `python scripts/prefilter_v4_88k.py --max-seq-len 40960 --out data/v4_88k_filtered`
   → 87,279 train rows (99.69%) + 512 val. Worst-case probe set:
   `data/v4_88k_filtered/longest256.parquet` (256 longest rows, 38.8k–40.9k tokens; built
   with HF `datasets` sort/select — pyarrow `take` overflows).
6. `.git/info/exclude` has `/data/` and `/checkpoints/`.

### Gotchas that cost time (don't repeat)
- **Base image ships apt cuDNN 9.8 + NCCL 2.25 (CUDA 12) in /usr/lib.** TE's loader prefers a
  "system" cuDNN (globs `$CUDNN_HOME`, else `dlopen("libcudnn.so")` = apt 9.8) → TE silently
  uses `UnfusedDotProductAttention` (~259 GiB at 32k tokens) or you get
  `undefined symbol: ncclCommWindowDeregister`. Fixed by `CUDNN_HOME` + LD_LIBRARY_PATH in the
  env script. Verify: `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2` → `Selected backend = FusedAttention`.
- Toolkit cuBLASLt 13.1 on LD_LIBRARY_PATH shadows pip's 13.4 → TE import fails
  (`cublasLtGroupedMatrixLayoutInit_internal`). Env script strips it.
- **Killing runs:** rank processes re-parent to PID 1 in their own process group; `kill -- -PGID`
  misses them and they keep ~170 GB GPU → next run OOMs falsely. Kill
  `recipes/vlm/finetune.py` PIDs explicitly; wait for GPUs ≈ 0 MiB.
  `pgrep -f` / `pkill -f` patterns also match your own shell — filter `$$` with ps+awk.
- Shell `grep` is ugrep: use `command grep`; avoid `.{0,N}`.
- `uv pip install` into the venv from inside the repo applies pyproject build config — run from elsewhere.
- DeepEP works on B300 (test_intranode passes); the GB200 `hybridep` workaround isn't needed.
- `OMP_NUM_THREADS` does not speed up checkpoint loading (single-threaded Python).

---

## 4. Repo changes (uncommitted) — `claude_handoff/uncommitted.patch` + untracked YAML

| File | Change |
|---|---|
| `examples/.../v4_88k.py` | `_resolve_markers` derives thinking-on and thinking-off generation-prompt suffixes; collate trims the longest that matches (masks the empty think block) |
| `examples/.../check_masking_v4_88k.py` | accepts either suffix |
| `examples/.../qwen3_6_35b_v4_88k_ep8.yaml` | optimizer → `torchao.optim.AdamW8bit` |
| `nemo_automodel/components/optim/scheduler.py` | tensor lr updated in place (`fill_`) |
| `tests/unit_tests/optim/test_scheduler.py` | test for tensor lr (57 tests pass) |
| `examples/.../RUNBOOK_v4_88k.md` | masking decision, do-not-undo row, smoke warmup override, dev-group note, Blackwell TE/cuDNN note, 8-bit memory math |
| `examples/.../qwen3_6_35b_v4_88k_ddp2.yaml` (new) | pure DDP variant (see §6) |

Reverted on request (not in the tree): attention right-padding SDPA patch; custom length sampler.

---

## 5. Runbook rungs (2 × B300; run with `--nproc-per-node 2 --distributed.ep_size 2`)

| Rung | Result |
|---|---|
| 1 kernels | ok (fla 0.4.2, causal_conv1d 1.6.0, deep_ep + HybridEP, TE 2.18, grouped_gemm, cut-cross-entropy) |
| 2 masking | failed 8/8 (think-block trim never fired) → fixed → 8/8 ok |
| 3 prefilter | 87,279 kept (99.69%), 273 dropped |
| 4 config parse | ok |
| 5 tiny proxy | ok (6 steps, finite) |
| 6 smoke | ok: step-0 loss 0.94 (weights loaded), val 0.68, no DeepEP fallback, backend te/deepep kept |
| 7 200-step probe | **not started** |

Runs shorter than 51 steps need `--lr_scheduler.lr_warmup_steps 1` (scheduler asserts warmup < decay).
`moe_metrics.enabled` is ignored by the VLM recipe (only LLM train_ft reads it; W&B only).

---

## 6. Performance findings

Setup: FSDP2 + EP2 + DeepEP, TE 2.18 fused attention, `experts: te`, activation checkpointing on,
AdamW8bit, real shuffled data, global batch 32.

| config | s/step | real tokens/s | peak GiB/GPU |
|---|---|---|---|
| lbs 2, PyTorch SDPA attn (before TE fix) | ~165 | 2.0–3.6k | 136 |
| lbs 1, TE fused | ~79 | ~5k | 139 |
| **lbs 2, TE fused** | **~64** | **4.8–7.0k** | 166 |
| lbs 4, TE fused | not measured | — | worst case ~192 (fits) |
| lbs 8 | OOM on worst case | — | — |

- Worst-case ceiling (256 longest rows): lbs 2 ok (168 GiB), lbs 4 ok (~192 GiB), lbs 8 OOM.
- Padding with random batching: lbs 2 32%, lbs 4 49%, lbs 8 58% (tps counts real tokens only).
- ETA at lbs 2: ~48 h/epoch → ~4 days for 2 epochs.
- TE fused vs SDPA flash at 32k tokens (attention fwd+bwd): 26.6 ms vs 93.5 ms, same memory.
- grad_norm 3–7 vs clip 1.0 in the first 5 steps; runbook rule: halve LR if persistent in rung 7.
- Not applicable: fused RoPE (force-disabled globally, #3027), `compile_attn` (needs sdpa+torch linear),
  CUDA graphs (variable sequence lengths), `rms_norm` backend (model hard-codes Qwen3NextRMSNorm).
- Knob sweep (AC off/selective, defer_fsdp_grad_sync, experts gmm/torch_mm, hybridep,
  async dispatch, dispatcher SMs, MXFP8) was **planned but stopped** by the user; harness
  `/workspace/perf_sweep.sh` is ready.

**Pure DDP** (`qwen3_6_35b_v4_88k_ddp2.yaml`: `strategy: ddp`, ep 1, `dispatcher: torch`,
`experts: torch_mm`, `gradient_as_bucket_view: true`, lbs 1): each rank builds the full model
on CPU (~13 min startup, single-threaded). Fit unknown — crashed on the GPU-0 fault. Memory
estimate ~210 GB/GPU before activations. FSDP2 `dp_replicate_size: 2` is rejected on 2 GPUs.

---

## 7. Next steps (after GPUs are healthy)

1. Health checks (§0). Re-run rung 1 imports and a 5-step smoke (lbs 2) to confirm the stack.
2. Optional: finish the lbs-4 real-data speed run (`OUT=/workspace/lbs bash /workspace/lbs_sweep.sh "" "4"`)
   and/or the DDP fit probe (`OUT=/workspace/ddp CFG=examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ddp2.yaml EXTRA_COMMON="" bash /workspace/lbs_sweep.sh "1 2 4" "1 2 4"`).
3. Rung 7: `automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml --nproc-per-node 2 --distributed.ep_size 2 --step_scheduler.local_batch_size <2|4> --step_scheduler.max_steps 200`
   — watch loss trend, grad_norm vs clip, val at 100/200; kill ~150 and resume via `checkpoint.restore_from`.
4. Rung 8 full run, rung 9 export check (copy `preprocessor_config.json` + `video_preprocessor_config.json` from the base snapshot).
5. Commit when the user asks.

---

## 8. File map

| Path | What |
|---|---|
| `/workspace/automodel_env.sh` | environment (source it every shell) |
| `/workspace/claude_handoff/` | memory, transcript, scratch probes, patch, restore/backup scripts |
| `/workspace/perf_sweep.sh`, `/workspace/lbs_sweep.sh` | benchmark harnesses |
| `/workspace/lbs/`, `/workspace/ddp/`, `/workspace/perf/` | sweep logs and summaries |
| `/workspace/Automodel/data/v4_88k_filtered/` | train/val/longest256 parquet |
| `/workspace/hf/` | HF cache (weights) |
| `/workspace/*.log` | install / rung logs |
