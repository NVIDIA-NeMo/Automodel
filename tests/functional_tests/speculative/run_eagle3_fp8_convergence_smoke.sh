#!/bin/bash
# EAGLE-3 fp8 draft-training convergence smoke (SM89+ hardware, 1 GPU).
#
# Discharges the "Validate fp8 draft convergence on SM89+ hardware" validation
# debt from issue #2958. The fp8 feature (#2963) shipped with unit tests and an
# `emulate: true` numerics path, but was never run through a REAL fp8 GEMM on
# fp8-capable silicon, so a genuine convergence check was still owed.
#
# What it does: trains the SAME tiny EAGLE-3 draft twice from the same seed and
# data order -- once with real hardware fp8 (`fp8.emulate: false`) and once in
# bf16 (`fp8.enabled: false`) -- then asserts that
#   1) the fp8 run reaches steps and logs FINITE loss (fp8 didn't NaN/inf),
#   2) the fp8 loss TRENDS DOWN over the run (the draft actually learns in fp8),
#   3) the fp8 loss trajectory TRACKS bf16 within tolerance (fp8 quantization
#      did not break convergence versus the bf16 reference).
# (2)+(3) together are the convergence validation the issue asks for.
#
# PREREQUISITES:
#   * ONE GPU with compute capability >= 8.9 (H100 / H200 / L40S / Ada). The
#     script refuses to certify on anything older -- pre-Hopper has no fp8 GEMM,
#     so `emulate: false` would fall back and prove nothing. (Set
#     ALLOW_NON_SM89=1 to run anyway for a PIPELINE-ONLY check; it will NOT be
#     treated as discharging the validation debt and says so loudly.)
#   * torchao installed (the fp8 path imports torchao.float8).
#   * flash-attn (the draft uses flash_attention_2).
#   * A local or hub Qwen3-8B target. Override with TARGET=/path/to/Qwen3-8B.
#
# USAGE (on an H100 node):
#   TARGET=/path/to/Qwen3-8B \
#     bash tests/functional_tests/speculative/run_eagle3_fp8_convergence_smoke.sh
#
# Knobs (env): TARGET, WORK (scratch dir), N_ROWS (default 64),
#   TOL (fp8-vs-bf16 final-loss relative tolerance, default 0.15).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TARGET="${TARGET:-Qwen/Qwen3-8B}"
WORK="${WORK:-$REPO/.eagle3_fp8_smoke_work}"
BASE_YAML="examples/speculative/eagle3/qwen3_eagle3_fp8_smoke.yaml"
N_ROWS="${N_ROWS:-64}"
TOL="${TOL:-0.15}"
mkdir -p "$WORK"

echo "### 0/5  hardware gate: require compute capability >= 8.9 for a REAL fp8 GEMM"
python - "${ALLOW_NON_SM89:-0}" <<'PY'
import sys
import torch

allow = sys.argv[1] == "1"
if not torch.cuda.is_available():
    print("FATAL: no CUDA device visible; fp8 convergence must run on an fp8-capable GPU.")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"GPU 0: {name}  compute capability sm_{major}{minor}")
if (major, minor) >= (8, 9):
    print("OK: SM89+ -> real hardware fp8 GEMM available; this run CAN discharge the fp8 validation debt.")
elif allow:
    print("WARN: pre-SM89 GPU with ALLOW_NON_SM89=1 -> PIPELINE-ONLY run.")
    print("      torchao has no fp8 GEMM here; `emulate: false` proves NOTHING about hardware fp8.")
    print("      This run does NOT discharge issue #2958's fp8 validation debt.")
else:
    print("FATAL: this GPU is pre-SM89. Real fp8 needs H100/H200/L40S/Ada (sm_89+).")
    print("       Re-run on fp8-capable hardware, or set ALLOW_NON_SM89=1 for a pipeline-only check.")
    sys.exit(1)
PY

echo "### 1/5  CPU unit tests for the fp8 draft wiring (helper swap + recipe hook)"
python -m pytest tests/unit_tests/recipes/llm/test_spec_draft_fp8_lora.py -q

echo "### 2/5  build $N_ROWS tiny smoke rows (OpenAI 'messages' jsonl)"
python - "$WORK/train.jsonl" "$N_ROWS" <<'PY'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
rows = [{"id": i, "messages": [
    {"role": "user", "content": f"In one sentence, state fact number {i}."},
    {"role": "assistant", "content": f"Fact {i}: a short teacher-forced answer used only for a pipeline smoke, not for accuracy."},
]} for i in range(n)]
open(out, "w").write("\n".join(json.dumps(r) for r in rows))
print("wrote", len(rows), "rows ->", out)
PY

echo "### 3/5  materialize the two run configs (fp8 real vs bf16 baseline, same seed/data)"
python - "$BASE_YAML" "$WORK" "$TARGET" "$WORK/train.jsonl" <<'PY'
import sys, yaml
src, work, target, data = sys.argv[1:5]
base = yaml.safe_load(open(src))
base["recipe_args"]["target_model_name_or_path"] = target
base["recipe_args"]["train_data_path"] = data
base["recipe_args"]["val_data_path"] = None
base["recipe_args"]["train_split"] = None

def dump(cfg, path):
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    print("wrote", path)

# fp8 variant: real hardware GEMM (emulate false), compile on.
import copy
fp8 = copy.deepcopy(base)
fp8["recipe_args"]["output_dir"] = f"{work}/out_fp8"
fp8["checkpoint"]["checkpoint_dir"] = f"{work}/out_fp8/checkpoints"
fp8["fp8"]["enabled"] = True
fp8["fp8"]["emulate"] = False
dump(fp8, f"{work}/smoke_fp8.yaml")

# bf16 baseline: fp8 off, compile off (fair reference trajectory).
bf16 = copy.deepcopy(base)
bf16["recipe_args"]["output_dir"] = f"{work}/out_bf16"
bf16["checkpoint"]["checkpoint_dir"] = f"{work}/out_bf16/checkpoints"
bf16["fp8"]["enabled"] = False
bf16["compile"]["enabled"] = False
dump(bf16, f"{work}/smoke_bf16.yaml")
PY

echo "### 4/5  train both drafts (1 GPU each, sequential)"
FP8_LOG="$WORK/fp8.log"
BF16_LOG="$WORK/bf16.log"
echo "  -- fp8 (real GEMM) ..."
torchrun --standalone --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_eagle3 -c "$WORK/smoke_fp8.yaml" 2>&1 | tee "$FP8_LOG"
echo "  -- bf16 baseline ..."
torchrun --standalone --nproc_per_node=1 \
  -m nemo_automodel.recipes.llm.train_eagle3 -c "$WORK/smoke_bf16.yaml" 2>&1 | tee "$BF16_LOG"

echo "### 5/5  assert: fp8 finite + decreasing, and tracks bf16 within TOL=$TOL"
python - "$FP8_LOG" "$BF16_LOG" "$TOL" <<'PY'
import re, sys, math

pat = re.compile(r"step=(\d+)\s+train_loss=([0-9.eE+-]+)")

def losses(path):
    out = []
    for line in open(path):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    # de-dup by step, keep last, sort by step
    d = {}
    for s, v in out:
        d[s] = v
    return [d[s] for s in sorted(d)]

fp8_log, bf16_log, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
fp8 = losses(fp8_log)
bf16 = losses(bf16_log)

assert fp8, "no fp8 train_loss logged -> fp8 run never reached a step (inspect the log)"
assert bf16, "no bf16 train_loss logged -> baseline run never reached a step (inspect the log)"
assert all(math.isfinite(x) for x in fp8), f"NON-FINITE fp8 loss (fp8 GEMM produced NaN/inf): {fp8}"

# window-mean the first/last few steps to smooth per-step noise
k = max(1, min(3, len(fp8) // 2))
fp8_head, fp8_tail = sum(fp8[:k]) / k, sum(fp8[-k:]) / k
bf16_tail = sum(bf16[-k:]) / max(1, min(3, len(bf16) // 2))

print(f"fp8 : steps={len(fp8):>3}  head={fp8_head:.4f}  tail={fp8_tail:.4f}  min={min(fp8):.4f}")
print(f"bf16: steps={len(bf16):>3}  tail={bf16_tail:.4f}  min={min(bf16):.4f}")

ok = True
if fp8_tail > fp8_head:
    print(f"FAIL: fp8 loss did NOT decrease ({fp8_head:.4f} -> {fp8_tail:.4f}); draft is not learning in fp8.")
    ok = False
else:
    print(f"PASS: fp8 loss decreased {fp8_head:.4f} -> {fp8_tail:.4f}.")

rel = abs(fp8_tail - bf16_tail) / max(abs(bf16_tail), 1e-6)
if rel <= tol:
    print(f"PASS: fp8 tail tracks bf16 within tol (rel diff {rel:.3f} <= {tol}).")
else:
    print(f"FAIL: fp8 tail diverges from bf16 (rel diff {rel:.3f} > {tol}); fp8 hurt convergence.")
    ok = False

print("\nFP8 CONVERGENCE SMOKE: " + ("OK" if ok else "FAILED"))
sys.exit(0 if ok else 1)
PY
echo
echo "Logs: fp8=$FP8_LOG  bf16=$BF16_LOG"
echo "Paste the '### 5/5' block back on issue #2958 to record the fp8 hardware validation."
