# CoderForge CP validation — Qwen3-32B (dense) at 128K

End-to-end SFT pipeline on [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
to validate **context parallelism (CP)** in NeMo AutoModel on the **dense Qwen3-32B**
at a **128K** context length, then evaluate on
[SWE-bench Verified](https://www.swebench.com/verified) (eval in a follow-up PR).

CoderForge ships OpenHands agent **trajectories** (multi-turn assistant/tool
exchanges) in OpenAI chat format. Trajectories are long (median ~38K tokens with the
Qwen3 tokenizer), which is exactly why CP matters — and why length handling is the
crux of the data stage.

## Phase 1 — Data pipeline (this directory)

```
data/
  prefilter_dataset.py   Parse + clean + tokenize-once + coverage curve + length filter -> JSONL
  prefilter.sh           Runner with CoderForge + Qwen3 defaults (128K, the tools+generation template)
```

### Why prefilter (don't truncate)

When `apply_chat_template(truncation=True)` truncates a trajectory, the terminal
turn/stop token is silently dropped. The model never sees a complete turn ending and
learns to never stop → death-looping at inference. We therefore **drop** over-length
trajectories rather than truncate, so every training sample ends on a complete turn.

### Choosing the sequence length (why 128K)

Because we drop rather than truncate, `seq_length` directly sets how much data
survives. `prefilter_dataset.py` tokenizes every trajectory once — with the **Qwen3
tokenizer** and the **tools+generation chat template**, so `n_tokens` matches the exact
training render — and prints the retention curve. On the 155,144 `filtered_reward1`
trajectories (Qwen3 tokenizer; median ~38K, p95 ~71K, max ~184K tokens):

| seq_length | retention |
|---|---|
| 16K | ~0% (112 trajectories) |
| 32K | 30% |
| 49K | 76% |
| 64K | 92.5% |
| 96K | 99.3% |
| **128K** | **99.9%** |

We chose **128K (131072)** so almost no trajectory is dropped (99.9% retention). At
128K the CP topology (`cp16`, 8192 tokens/rank) still matches the per-rank load of the
proven gemma4-31B `cp8 @ 64K` run.

**To train at a different context length, just re-run the data stage at a different
`SEQ_LENGTH`.** The analyzed cache stores each trajectory's token count, so a new
length is a cheap re-filter (no re-tokenization) that emits a new `data.jsonl`; point
the recipe's `dataset.path_or_dataset_id` + `dataset.seq_length` at it and set `cp_size`
so `seq_length` stays divisible by `2 * cp_size`.

### Qwen3 specifics

- **Dense Qwen3-32B (`Qwen3ForCausalLM`) has no custom NeMo model class**, so it loads
  as the stock HuggingFace model through `NeMoAutoModelForCausalLM`. CP shards each
  sequence on the sequence dimension via **SDPA** (torch `context_parallel` intercepts
  `F.scaled_dot_product_attention`) — do not force `flash_attention_2` (bypasses CP).
- **No sequence packing.** Packing + CP>1 over SDPA is unsupported for a stock HF model
  (*"Packed sequence is only supported with CP size 1"*). We train **unpacked**:
  `default_collater` pads each batch to a multiple of `2 * cp_size`, and
  `local_batch_size=1` means a short trajectory only computes its own length, not a
  padded 128K.
- **Assistant-only masking** comes from `qwen3_coderforge_chat_template.jinja` — a
  tools-aware template with `{% generation %}` blocks, so the tokenizer returns
  `return_assistant_tokens_mask` directly. The stock Qwen3 template rewrites earlier
  turns (drops `<think>` from turns before the last user message) and can't produce a
  stable prefix-consistent mask.
- CoderForge messages use a union schema (`tool_calls: null` on plain turns); the
  preprocessor strips those, and the cache is **JSONL** (Parquet's Arrow struct
  unification would re-add the null keys and break `ChatDataset`).

### Run it

```bash
# 1. Analyze: tokenize once, print the retention curve, cache the analyzed JSONL.
MODEL=Qwen/Qwen3-32B bash data/prefilter.sh

# 2. Produce the 128K training cache (a cheap re-filter of the analyzed cache).
MODEL=Qwen/Qwen3-32B SEQ_LENGTH=131072 bash data/prefilter.sh
```

Run this inside an environment with `nemo_automodel` + `transformers` + `datasets`
(the nemo-automodel container or a matching venv) — `prefilter_dataset.py` imports
`nemo_automodel` for the exact chat-template token count. The output `data.jsonl`
plugs into the recipe (carve a small held-out `val.jsonl` from it for the validation
loss):

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq131072/data.jsonl
  seq_length: 131072
  chat_template: examples/long_context_validation/qwen3_32b/qwen3_coderforge_chat_template.jinja
```

## Phase 2 — Training recipe (Qwen3-32B + CP)

`qwen3_32b_coderforge_cp16_128k_lowerLR.yaml` — SFT on the base `Qwen/Qwen3-32B`, on
16 nodes / 128 GPUs, `cp16 × dp8`, `gbs=16`, 128K sequence length, **unpacked**,
`FusedLinearCrossEntropy` (fuses `lm_head`+CE and consumes the final hidden state,
avoiding the `[batch, seq, vocab]` fp32 logit upcast that OOMs on 32B), and the
tools+generation chat template for assistant-only masking. `cp16` spans 2 nodes and
gives 8192 tokens/rank at 128K.

It uses `lr 5e-6` — gentler, half of a first `1e-5` attempt that peaked then regressed
— with a 60-step warmup and cosine over `max_steps=800`, and `clip_grad_norm=1.0`.
`save_consolidated: false` keeps checkpointing fast (DCP-only; consolidate offline for
eval), and `ckpt_every_steps=100` makes the run resumable across 4h windows and lets
you evaluate intermediate checkpoints — the gentler LR kept improving through the early
ones. Launch on 16 nodes with your multi-node launcher
(`torchrun ... examples/llm_finetune/finetune.py -c <this recipe>`).

_Training / validation loss curve (wandb): link to be added._

## Phase 3 — SWE-bench Verified evaluation — *next*
