# Train Parallel Reconsideration

[Parallel Reconsideration](https://arxiv.org/abs/2611.05724v1) trains an EAGLE-3.1
draft to revise its next-token prediction. The ordinary and reconsideration
rollouts share an initial feature, the decoder layer, and the LM head. Each
reconsideration pass uses the learned self-query at the same position and reads
the committed prefix plus its own key. The sampled halt feature and KV always
come from the same depth.

The objective uses Eq. (6)'s unconditional one-pass CE, halt-weighted deeper CE,
gate loss, and entropy of the minibatch halt marginal. Both rollouts sample
their own token continuations. Hard teacher labels remain aligned to the
recorded context because the paper leaves rescoring diverged paths unspecified.
For an existing EAGLE-3 cache, the labels are argmaxes of its stored draft-vocab
probabilities with the recorded eligibility mask. Cache precision governs that
argmax.

The starter config uses Qwen3-0.6B, a 512-token cap, three rollout steps, and the
paper's depth cap of two and entropy coefficient of 0.01. CUDA training uses
BF16 autocast with FP32 draft parameters. Supply an
[UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) subset
in JSONL format with a `messages` list for each row. The example keeps the first
assistant turn so the stock Qwen3 template has stable prefixes for the
recipe's answer-only loss mask.

```bash
mkdir -p data
python - <<'PY'
import json
from datasets import load_dataset

rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
with open("data/ultrachat_train.jsonl", "w", encoding="utf-8") as stream:
    for row in rows.take(2048):
        messages = row["messages"]
        end = next(i for i, message in enumerate(messages) if message["role"] == "assistant")
        stream.write(json.dumps({"messages": messages[: end + 1]}) + "\n")
PY
```

```bash
set -o pipefail
run_dir=outputs/qwen3_0_6b_parallel_reconsideration
mkdir -p "$run_dir"
CUDA_VISIBLE_DEVICES=0 bash examples/speculative/parallel_reconsideration/run_qwen3_0_6b.sh \
  data/ultrachat_train.jsonl "$run_dir" 2>&1 | tee "$run_dir/train.log"
```

Pass `--checkpoint.restore_from=LATEST` after the two positional arguments to
resume. An intermediate step checkpoint resumes from the start of its recorded
epoch. The native consolidated export includes the query, gate, and vocabulary
map. This path uses the full-parameter eager draft on unpacked sequences.
Engine integration is required to serve a reconsidered tree.
