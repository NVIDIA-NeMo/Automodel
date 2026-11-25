#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: %s CHAT_JSONL OUTPUT_DIR [--section.field=value ...]\n' "$0"
    exit 0
fi
if [[ $# -lt 2 ]]; then
    printf 'Specify CHAT_JSONL and OUTPUT_DIR. Run %s --help for usage.\n' "$0" >&2
    exit 2
fi

data=$1
output=$2
shift 2
exec python -m nemo_automodel.recipes.speculative_decoding.train_eagle3 \
    -c examples/speculative/parallel_reconsideration/qwen3_0_6b_mvp.yaml \
    "--recipe_args.train_data_path=$data" "--recipe_args.output_dir=$output" \
    "--checkpoint.checkpoint_dir=$output/checkpoints" "$@"
