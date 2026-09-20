#!/bin/bash
# Regenerate the end-to-end nsys profiles for the Automodel guest lecture (2026 September edition,
# Moonlight-V4-16B-A3B).  Portable profiles run on any CUDA GPU; set RUN_HOPPER=1 to also produce the
# DeepEP / TileLang profiles, which need Hopper-class GPUs and the tilelang, tile_kernels and deep_ep
# packages.
#
# Per-module profiles are not produced here.  Run examples/scalable_ai/profile_layer.py directly for
# those; `--nvtx true` below annotates every submodule's forward and backward inside the end-to-end
# runs, so per-module timings can also be read straight out of these reports with
# `nsys stats --report nvtx_gpu_proj_sum <profile>.nsys-rep`.
#
# A profile is regenerated when its .nsys-rep is missing or was produced by a different command --
# the resolved argv is recorded next to it -- so a rerun picks up new stages and edited flags without
# re-running the rest.  FORCE=1 regenerates everything.
set +e   # keep going if one profile fails

# Do not embed credentials in the profiles
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$SCRIPT_DIR"
cd "$REPO_ROOT"

# torchrun sets sys.path[0] to the *script's* directory, not the repo root, so without this a
# container's pre-installed nemo_automodel shadows the checkout being profiled.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# tilelang, triton and inductor abort at import when they cannot create their cache directory.
# Automodel then silently falls back to the transformers model class, which rejects `backend`,
# so a stage that looks like it ran would actually be profiling the wrong implementation.
CACHE_ROOT="${TMPDIR:-/tmp}/automodel_profile_cache"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$CACHE_ROOT/tilelang}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
mkdir -p "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "=== Regenerating profiles in $OUT (repo root: $REPO_ROOT) ==="

# ``--nvtx true`` makes the recipe run ``nemo_automodel.autonvtx.patch`` over the model, which wraps every
# submodule's forward and backward in an NVTX range, so the timeline is labelled by module instead of by raw
# kernel.  The hooks are installed once at setup, so they are live for every step, not just the
# captured window; the cost is one hook pair per module.
#
# The capture window is iterations 3-5.  It takes three settings that only work together: the recipe calls
# cudaProfilerStart/Stop at nsys_start/nsys_end, but only for ranks listed in `nsys_ranks` -- which the
# configs default to `[]`, so the calls never fire -- and nsys only honours them under `-c cudaProfilerApi`
# below.  With any one missing, nsys captures the whole run instead, iteration 0 included.  That matters
# because DeepEP and TileLang JIT-compile on the first iteration: stage4's iteration 0 carries 2777 ms of
# kernel time against 46.7 ms in a steady step, which swamps any per-module total taken over the capture.
E2E_COMMON=(--benchmark.nsys_start 3 --benchmark.nsys_end 5 --benchmark.nsys_ranks '[0,1]'
            --step_scheduler.max_steps 6 --benchmark.warmup_steps 1
            --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024
            --nvtx true)

# --trace=osrt records the OS-runtime calls the compute thread makes, which is how a "the host is blocked"
# hypothesis gets tested rather than assumed: on these captures it showed the main thread blocked for only
# 8% of the GPU stall, the rest being userspace execution.  --python-sampling periodically captures the
# Python stack, which is the only way to attribute that userspace time to frames.  Note it needs CPU
# IP/backtrace sampling to be available on the host: where `kernel.perf_event_paranoid` forbids it (or the
# container lacks SYS_ADMIN) the injection still loads but collects nothing, and the report records
# "Unable to configure the collection of CPU IP/backtrace samples" in its diagnostics.  Check with
# `nsys status --environment`.
NSYS_ARGS=(--force-overwrite true -c cudaProfilerApi --trace=cuda,nvtx,osrt --cuda-memory-usage=true
           --python-sampling=true --python-sampling-frequency=2000)

e2e() {  # name config extra-args...
  local name=$1 config=$2; shift 2
  local report="$OUT/$name.nsys-rep" stamp="$OUT/.$name.cmd"
  # Cache on the command, not just the filename: an edited stage flag or a new E2E_COMMON entry has to
  # invalidate the report, or a rerun silently serves one captured with different settings.
  local cmd="${NSYS_ARGS[*]} $config ${E2E_COMMON[*]} $*"
  if [[ -s "$report" && "${FORCE:-0}" != "1" && "$(cat "$stamp" 2>/dev/null)" == "$cmd" ]]; then
    echo "=== e2e: $name -- cached, skipping (FORCE=1 to regenerate) ==="
    return 0
  fi
  echo "=== e2e: $name ==="
  # nsys finalizes a report even when the target crashes, so a failed run must not leave one behind:
  # it would look like a cache hit forever and hide the stage that needs attention.
  if nsys profile "${NSYS_ARGS[@]}" --output="$report" \
      torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py --config "$config" "${E2E_COMMON[@]}" "$@"; then
    printf '%s' "$cmd" > "$stamp"
  else
    echo "=== e2e: $name FAILED, discarding partial report ==="
    rm -f "$report" "$stamp"
  fi
}

CFG_HF=examples/scalable_ai/configs/moonlight_v4_16b_hf.yaml
CFG_TORCH=examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml
CFG_TILELANG=examples/scalable_ai/configs/moonlight_v4_16b_tilelang_deepep.yaml
STAGE_MODEL=(--model.config.num_hidden_layers 3 --distributed.ep_size 2)

# The 2-layer Automodel run: the only like-for-like comparison against stage 1's stock transformers,
# since sliding-window is the one attention kind both implementations compute the same way.
e2e moonlight_v4_torch_small "$CFG_TORCH" --model.config.num_hidden_layers 2 --distributed.ep_size 1

# ---------------------------------------------------------------------------
# Stage ladder: one optimization per step, everything else held fixed, so each
# profile differs from the previous one by exactly one backend knob and every
# delta is attributable.  All stages use the same 3-layer model (the schedule's
# first three entries: SWA, SWA, CSA) at ep_size 2, except stage 1 -- stock
# transformers cannot train CSA layers, so it runs the 2 SWA layers only.
#
#   stage 1 -> 2   stock transformers            -> NeMo Automodel native
#   stage 2 -> 3   per-expert loop               -> GroupedGEMM (torch._grouped_mm)
#   stage 3 -> 4   torch dispatcher (all-gather) -> DeepEP
#   stage 4 -> 5   eager dense attention         -> TileLang sparse attn + indexer
#
# Each stage passes only the knob that differs from the stage above it; every other backend is left
# to the config, so the ladder cannot silently pin a stale value if the configs move.  Stages 3 and 5
# therefore need no flags at all -- they are their config's defaults.  See nsys_profiles/README.md
# "Stage ladder" for why rms_norm is not a rung and what stage 5 matches.
#
# Stages 4-5 need deep_ep / tilelang, so they live behind RUN_HOPPER=1.
# ---------------------------------------------------------------------------
e2e stage1_hf_ootb       "$CFG_HF"    --model.config.num_hidden_layers 2
e2e stage2_am_expertloop "$CFG_TORCH" "${STAGE_MODEL[@]}" --model.backend.experts torch
e2e stage3_groupedgemm   "$CFG_TORCH" "${STAGE_MODEL[@]}"


if [[ "${RUN_HOPPER:-0}" == "1" ]]; then
  # stage 3 -> 4 swaps the token dispatcher; 4 -> 5 the attention backend.  Stage 5 is the TileLang
  # config's defaults, which differ from stage 4 only in `attn`.
  e2e stage4_deepep   "$CFG_TORCH" "${STAGE_MODEL[@]}" --model.backend.dispatcher deepep
  e2e stage5_tilelang "$CFG_TILELANG" "${STAGE_MODEL[@]}"
fi

echo "=== done ==="
ls -lh "$OUT"/*.nsys-rep 2>/dev/null
