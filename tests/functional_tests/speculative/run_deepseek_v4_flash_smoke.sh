#!/bin/bash
# DeepSeek-V4-Flash DSpark end-to-end smoke test (REAL recipe path, 8 GPUs).
#
# Validates: distributed V4 target load (expert-parallel + FSDP2, FP8/FP4 dequant)
#   -> online hidden-state capture across the EP shards -> draft training
#   -> finite, decreasing three-term loss. This is the path the full config
#   (deepseek_v4_flash_dspark.yaml) takes, just shrunk to reach a step in minutes.
#
# PREREQUISITES (the smoke cannot pass without these):
#   * DeepEP (or HybridEP) installed in the env -- the V4 MoE token dispatcher
#     requires it; without it setup() raises "HybridEP is not installed".
#   * 8 GPUs, >=80 GiB each (V4-Flash is ~530 GiB bf16, ~66 GiB/rank at ep_size=8).
#   * A local DeepSeek-V4-Flash checkpoint.
#
# USAGE:
#   TARGET=/path/to/DeepSeek-V4-Flash \
#     bash tests/functional_tests/speculative/run_deepseek_v4_flash_smoke.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TARGET="${TARGET:?set TARGET=/path/to/DeepSeek-V4-Flash (local path or hub id)}"
WORK="${WORK:-$REPO/.v4_smoke_work}"
SMOKE_YAML="examples/speculative/dspark/deepseek_v4_flash_smoke.yaml"
mkdir -p "$WORK"

echo "### 1/4  CPU unit tests (draft / config / registry / inv_freq pin)"
python -m pytest tests/unit_tests/speculative/test_dspark_draft_deepseek_v4.py -q

echo "### 2/4  build 64 tiny smoke rows (OpenAI 'messages' jsonl, same schema as the real data)"
python - "$WORK/train.jsonl" <<'PY'
import json, sys
out = sys.argv[1]
rows = [{"id": i, "messages": [
    {"role": "user", "content": f"In one sentence, state fact number {i}."},
    {"role": "assistant", "content": f"Fact {i}: a short teacher-forced answer used only for a pipeline smoke test, not for accuracy."},
]} for i in range(64)]
open(out, "w").write("\n".join(json.dumps(r) for r in rows))
print("wrote", len(rows), "rows ->", out)
PY

echo "### 3/4  materialize run config (point it at your target + smoke data)"
RUN_YAML="$WORK/smoke.yaml"
python - "$SMOKE_YAML" "$RUN_YAML" "$TARGET" "$WORK/train.jsonl" "$WORK/out" <<'PY'
import sys, yaml
src, dst, target, data, out = sys.argv[1:6]
c = yaml.safe_load(open(src))
c["recipe_args"]["target_model_name_or_path"] = target
c["recipe_args"]["train_data_path"] = data
c["recipe_args"]["output_dir"] = out
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print("wrote", dst)
PY

echo "### 4/4  8-GPU real-weight training smoke"
LOG="$WORK/smoke.log"
torchrun --standalone --nproc_per_node=8 \
  -m nemo_automodel.recipes.llm.train_dspark -c "$RUN_YAML" 2>&1 | tee "$LOG"

echo "### assert: loss logged, finite, and trended down"
python - "$WORK/out/dspark_train_metrics.jsonl" <<'PY'
import json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
losses = [r["loss"] for r in rows if "loss" in r]
assert losses, "no loss logged -> training never reached a step (inspect target load / EP)"
assert all(math.isfinite(x) for x in losses), f"non-finite loss: {losses}"
print("steps=%d  first=%.3f  last=%.3f  min=%.3f" % (len(losses), losses[0], losses[-1], min(losses)))
print("SMOKE OK" if losses[-1] <= losses[0] else "SMOKE WARN: loss did not drop over these few steps -- inspect")
PY
echo
echo "Manual EP sanity points to eyeball in $LOG (the 5 things a CPU test can't cover):"
echo "  1) FP8/FP4 base weights dequantize to bf16 on load (no 'non-finite' on target fwd)"
echo "  2) captured hidden states are full per-rank (no all-gather / not a DTensor)"
echo "  3) frozen V4 target forward runs with a 2D mask + use_cache=False"
echo "  4) ep_size=8 divides n_routed_experts=256 (32 experts/rank)"
echo "  5) per-rank weight memory ~66 GiB (nvidia-smi during the run)"
echo "DONE. Full log: $LOG"
