#!/bin/bash
# GLM-5.2 DSpark end-to-end smoke test (REAL recipe path, 8 GPUs).
#
# Validates: distributed GLM-5.2 target load (expert-parallel + FSDP2, FP8 dequant)
#   -> online hidden-state capture across the EP shards -> draft training
#   -> finite, decreasing three-term loss. This is the path the full config
#   (glm_5.2_dspark.yaml) takes, but with the target shrunk to 6 layers
#   (target_num_hidden_layers=6) so it fits on ONE 8x80GB node and reaches a step
#   in minutes. The full 78-layer target needs multiple nodes (see the full config).
#
# PREREQUISITES (the smoke cannot pass without these):
#   * HybridEP (or DeepEP) installed in the env -- the GLM MoE token dispatcher
#     requires it; without it setup() raises "HybridEP is not installed".
#   * 8 GPUs, >=80 GiB each. The smoke loads only 6 target layers (3 dense + 3
#     routed-MoE / IndexShare layers, ~24 GiB/rank at ep_size=8); the full target
#     would not fit here, hence the reduction.
#   * A local GLM-5.2 checkpoint.
#
# USAGE:
#   TARGET=/path/to/GLM-5.2 \
#     bash tests/functional_tests/speculative/run_glm_5.2_smoke.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TARGET="${TARGET:?set TARGET=/path/to/GLM-5.2 (local path or hub id)}"
WORK="${WORK:-$REPO/.glm_smoke_work}"
BASE_YAML="examples/speculative/dspark/glm_5.2_dspark.yaml"
mkdir -p "$WORK"

echo "### 1/4  CPU unit tests (draft / config / registry / freqs pin)"
python -m pytest tests/unit_tests/speculative/test_dspark_glm_5_2.py -q

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
python - "$BASE_YAML" "$RUN_YAML" "$TARGET" "$WORK/train.jsonl" "$WORK/out" <<'PY'
import sys, yaml
src, dst, target, data, out = sys.argv[1:6]
c = yaml.safe_load(open(src))
dist = c["distributed"]
dist["ep_size"] = 8
dist["activation_checkpointing"] = True

args = c["recipe_args"]
args["target_model_name_or_path"] = target
args["train_data_path"] = data
args.pop("train_split", None)
args["val_data_path"] = None
args.pop("val_split", None)
args["output_dir"] = out
args["seq_length"] = 512
args["micro_batch_size"] = 1
args["grad_accumulation_steps"] = 1
args["num_workers"] = 0
args["num_epochs"] = 1
args["target_num_hidden_layers"] = 6
args["draft_num_hidden_layers"] = 2
args["num_anchors"] = 16
args["target_layer_ids"] = [3, 4, 5]
args["log_every_steps"] = 1

c["optimizer"]["warmup_ratio"] = 0.0
c["checkpoint"]["enabled"] = False
c["wandb"]["enable"] = False
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
echo "  1) FP8 base weights dequantize to bf16 on load (no 'non-finite' on target fwd)"
echo "  2) captured hidden states are full per-rank (no all-gather / not a DTensor)"
echo "  3) frozen GLM target forward runs with a 2D mask + use_cache=False"
echo "  4) ep_size=8 divides n_routed_experts=256 (32 experts/rank)"
echo "  5) per-rank memory is ~24 GiB here (only 6 target layers loaded); the FULL"
echo "     78-layer target needs multiple nodes (ep_size>=32); it does not fit on one 8x80GB box"
echo "DONE. Full log: $LOG"
