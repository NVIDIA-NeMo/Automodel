# CoderForge Convergence Pipeline (CP validation)

End-to-end SFT pipeline on [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
to validate **context parallelism (CP)** in NeMo AutoModel, then evaluate on
[SWE-bench Verified](https://www.swebench.com/verified). Tracked in AM-492.

CoderForge ships OpenHands agent **trajectories** (multi-turn assistant/tool
exchanges) in OpenAI chat format. Trajectories are long (median ~38K tokens),
which is exactly why CP matters — and why length handling is the crux of the
data stage.

## Phase 1 — Data pipeline (this directory)

```
data/
  prefilter_dataset.py   Parse + clean + tokenize-once + coverage curve + length filter -> JSONL
  prefilter.sh           Runner with CoderForge + Gemma4 defaults
  validate_data.py       Token-level correctness assertions on the ChatDataset output
```

### Why prefilter (don't truncate)

When `apply_chat_template(truncation=True)` truncates a trajectory, the terminal
turn marker (`<turn|>`, token 106) is silently dropped. The model never sees a
complete turn ending and learns to never stop → death-looping at inference. We
therefore **drop** over-length trajectories rather than truncate, so every
training sample ends on a complete turn.

### Gemma4 specifics (verified)

- **Tokenizer/template** come from the local checkpoint dir (`chat_template.jinja`
  + `tokenizer.json`). Point `--model` at it.
- The Gemma4 template has **no real `{% generation %}` block**, so
  `return_assistant_tokens_mask` is all-zeros. `ChatDataset` detects this and
  falls back to `_build_multiturn_assistant_mask`, which supervises every
  assistant turn (including tool-call turns). Correct, but O(turns) re-tokenization
  per sample — pre-tokenization is a future optimization for large runs.
- The stop token the model must learn is **`<turn|>` (id 106)**, listed in
  `generation_config.eos_token_id=[1,106,50]` — **not** the tokenizer
  `eos_token_id` (1 = `<eos>`). `validate_data.py` checks 106.
- CoderForge messages use a union schema (`tool_calls: null` on plain turns);
  the preprocessor strips those, and the cache is **JSONL** (Parquet's Arrow
  struct unification would re-add the null keys and break `ChatDataset`).

### Run it

```bash
# 1. Analyze: tokenize once, print the coverage curve (retention vs seq_length),
#    cache the analyzed JSONL. Pick seq_length from the curve at your retention target.
MODEL=/path/to/hf_gemma4_31b_it bash data/prefilter.sh

# 2. Produce a training-ready cache at the chosen seq_length (cheap re-filter, no
#    re-tokenization — so a larger seq_length later is a quick second run).
MODEL=/path/to/hf_gemma4_31b_it SEQ_LENGTH=32768 bash data/prefilter.sh

# 3. Validate the cache through the exact ChatDataset training path.
python data/validate_data.py \
    --dataset data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq32768/data.jsonl \
    --model /path/to/hf_gemma4_31b_it \
    --seq_length 32768 --num-samples 200
```

The output `data.jsonl` plugs into a training config:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq32768/data.jsonl
  seq_length: 32768
```

## Phase 2 — Training recipe (Gemma4 31B + CP) — *next*

`gemma4_31b_coderforge_cp<N>_<seq>k.yaml`. Pilot (1–5K trajectories,
`max_steps=200`, monitor loss/throughput/memory) then a full run, with
`save_consolidated: True` so the HF checkpoint is SWE-bench-evaluable.

## Phase 3 — SWE-bench Verified evaluation — *next*

Outside AutoModel: serve the SFT model (vLLM/SGLang), run mini-SWE-agent →
`predictions.jsonl` → `swebench.harness.run_evaluation`. Baseline: base Gemma4
vs the CP-trained SFT model.
