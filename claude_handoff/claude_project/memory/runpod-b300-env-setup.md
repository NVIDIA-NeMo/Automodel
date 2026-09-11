---
name: runpod-b300-env-setup
description: "How the 2xB300 RunPod box was set up for Automodel (CUDA 13 via apt, driver-package trap, caches on /workspace, env file)"
metadata: 
  node_type: memory
  type: project
  originSessionId: dc22bf29-146a-4a63-8910-37a0e87d9eb4
  modified: 2026-09-11T09:14:17.667Z
---

2xB300 (SM 10.3, 275 GB each) RunPod container, driver 580.126 (host-mounted, read-only libs). Set up 2026-09-11 for the Qwen3.6-35B-A3B v4_88k SFT runbook ([[v4-88k-masking-decision]]).

- CUDA 13.0 toolkit installed via apt (`cuda-toolkit-13-0`) to match torch 2.10.0+cu130 in uv.lock; `/usr/local/cuda` alternative points to 13.0. Base image shipped 12.8.
- **Never apt-install `libnvidia-ml-dev`** (or anything pulling `libnvidia-compute-*`/`nvidia-persistenced`): it drags in driver userspace pkgs that fail to unpack ("Invalid cross-device link") and leave dpkg broken. CUDA 13's `cuda-nvml-dev-13-0` already provides nvml.h. Those driver pkgs are apt-mark held.
- Root overlay is only 30 GB: HF_HOME=/workspace/hf, UV_CACHE_DIR=/workspace/.uv_cache.
- `/workspace/automodel_env.sh` sets everything (HF_TOKEN from /workspace/.env, CUDA_HOME, TORCH_CUDA_ARCH_LIST="10.0;10.3", CPATH cccl, venv activate).
- Runtime must NOT have the CUDA toolkit lib64 on LD_LIBRARY_PATH (base image adds /usr/local/cuda/lib64): toolkit cuBLASLt 13.1 shadows the pip wheel's 13.4 and TE 2.15 fails with undefined symbol `cublasLtGroupedMatrixLayoutInit_internal`. The env script strips it.
- DeepEP (unpatched, built by uv) imports `pynvml`, which is not in uv.lock: `uv pip install nvidia-ml-py` into .venv. An exact `uv sync` removes it again — reinstall after every sync.
- TE fused attention on B300 (head_dim 256) needs TE >= 2.18 + cuDNN >= 9.23. Installed outside uv.lock (an exact `uv sync` reverts them): nvidia-cudnn-cu13 9.26.0.51 + transformer-engine{,-cu13,-torch} 2.18.0, with -torch built via `NVTE_WITH_NCCL_EP=0` (its NCCL-EP code needs torch>2.10 headers). Run `uv pip install` from OUTSIDE the repo with `--python .venv/bin/python`, else pyproject's [tool.uv] build config breaks it. Rollback list: scratchpad venv_freeze_before_te218.txt (TE 2.15.0, cuDNN 9.15.1.9).
- Base image has apt cuDNN 9.8 + NCCL 2.25 (held). TE's loader picks them over the venv unless `CUDNN_HOME=<venv>/nvidia/cudnn` and venv nvidia/cudnn/lib + nvidia/nccl/lib lead LD_LIBRARY_PATH — the env script sets both. Symptom when wrong: TE selects UnfusedDotProductAttention (~259 GiB at 32k tokens) or `undefined symbol: ncclCommWindowDeregister`.
- `OMP_NUM_THREADS=64` exported in the env script (torchrun otherwise forces 1). It does NOT speed up checkpoint loading (single-threaded Python), only CPU tensor ops.
- Killing a run: the launcher's rank processes re-parent to PID 1 in their own process group, so `kill -- -PGID` misses them and they keep ~170 GB of GPU memory → later runs OOM falsely. Always kill `recipes/vlm/finetune.py` PIDs explicitly and wait for GPUs to read ~0 MiB. `pgrep -f`/`pkill -f` with a pattern also matches the calling shell — filter by `$$` via ps+awk.
- `grep` in this shell is ugrep; use `command grep`, avoid `.{0,N}` bounded repeats.
- Sweep harnesses: /workspace/perf_sweep.sh (knob sweep), /workspace/lbs_sweep.sh (batch-size ceiling + speed; env OUT/CFG/EXTRA_COMMON). Worst-case data: data/v4_88k_filtered/longest256.parquet.
- uv sync in two phases (torch first, then `--extra moe` source builds); do NOT use `--all-groups` — it pulls the `magi` group (MagiAttention source build, unused). DO add `--group dev`: the recipe's FusedLinearCrossEntropy imports `cut_cross_entropy`, which lives only in the dev group. Use `--inexact` so the non-locked nvidia-ml-py survives.
- The shell's `grep` is ugrep: bounded repeats like `.{0,120}` fail with "exceeds complexity limits" (silently breaks log monitors). Use `command grep` and simple patterns.

**Why:** B300 + CUDA-13 + container driver constraints are not recorded in the repo.
**How to apply:** Re-use the env file; if rebuilding kernels, keep the arch list and CPATH.
