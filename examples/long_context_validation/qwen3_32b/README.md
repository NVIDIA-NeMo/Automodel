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

Run the above inside an environment with `nemo_automodel` + `transformers` + `datasets`
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

Each checkpoint's `model/` dir is written as DCP shards (one per rank, fast to save) plus
an auto-generated `model/consolidate.sh` helper. To get an HF-loadable checkpoint for
eval, run that helper once per step on a CPU node — it calls `tools/offline_hf_consolidation.py`
to merge the shards into standard HF safetensors under `model/consolidated/`:

```bash
# from the AutoModel repo root; scale NPROC/threads to the CPU node
NPROC_PER_NODE=16 NUM_THREADS=5 bash <ckpt>/epoch_0_step_<N>/model/consolidate.sh
# -> <ckpt>/epoch_0_step_<N>/model/consolidated/  (config.json + tokenizer + model-*.safetensors)
```

_Training / validation loss curve (wandb):
<https://wandb.ai/Nemo-automodel/long_context_validation_qwen3_32b/workspace?nw=nwuserathittenaman>._

## Phase 3 — SWE-bench Verified evaluation

### Goal

Does CoderForge SFT lift Qwen3-32B's **SWE-bench Verified** resolve rate above the base?
Qwen3-32B already tool-calls, so the base already resolves ~18% — the question is whether
SFT gives a real **uplift**. Uses the shared, model-agnostic [`../eval/`](../eval) harness
(see its [README](../eval/README.md) for how it works); this page only adds the Qwen3
knobs and results.

### Scaffold

Same 3-tool OpenHands agent (`execute_bash` / `str_replace_editor` / `finish`) over vLLM +
Docker-less enroot, graded locally with the official `swebench` spec. Two Qwen3 specifics:
it emits **hermes**-style tool calls (`<tool_call>{...}</tool_call>`) and runs with its
native **thinking** mode on. Serve base and SFT identically so the delta is the model.

### Steps (from `../eval/`)

```bash
bash setup_eval_tooling.sh                    # one-time: venv + agent + swebench
SUBSET=verified sbatch prewarm_images.sub     # pre-import instance images once (CPU)

# Serve each checkpoint + run the agent -> preds.json (one 8-GPU node, TP=4 x DP=2).
# MODEL = a *consolidated* HF ckpt dir; NOPARSER=1 + thinking-on (see Note).
RUN_TAG=qwen3_base NAME=qwen3 MODEL=<base Qwen/Qwen3-32B> \
  SLICE=0:500 SUBSET=verified NOPARSER=1 MAX_TOKENS=16384 TP=4 DP=2 WORKERS=16 \
  OH3_ENABLE_THINKING=1 OH3_TEMPERATURE=0.6 sbatch --gpus-per-node=8 openhands3_run.sub
RUN_TAG=qwen3_sft  NAME=qwen3 MODEL=<CoderForge-SFT consolidated> \
  SLICE=0:500 SUBSET=verified NOPARSER=1 MAX_TOKENS=16384 TP=4 DP=2 WORKERS=16 \
  OH3_ENABLE_THINKING=1 OH3_TEMPERATURE=0.6 sbatch --gpus-per-node=8 openhands3_run.sub

# Grade both (SUBSET must match; confirm enroot-errs=0 before trusting a 0.0 resolve)
PREDS=<runs>/qwen3_base/preds.json SUBSET=verified RUN_TAG=grade_qwen3_base sbatch grade_enroot.sub
PREDS=<runs>/qwen3_sft/preds.json  SUBSET=verified RUN_TAG=grade_qwen3_sft  sbatch grade_enroot.sub
```

### Note

**1. Why `NOPARSER=1` (not `PARSER=hermes`):** Qwen3 emits hermes `<tool_call>{...}</tool_call>`
calls, but vLLM's built-in `hermes` parser does a *strict* `json.loads` that chokes on control
characters in code arguments and silently dropped ~1,873 calls. `NOPARSER=1` routes the raw text
to `content`, where `oh3_run.py`'s `parse_hermes_tool_calls` (`json.loads(strict=False)`) recovers
them.

**2. Thinking-on:** `OH3_ENABLE_THINKING=1` + `OH3_TEMPERATURE=0.6` turn on Qwen3's native
`<think>` reasoning (Qwen3's recommended thinking sampling). This was the biggest lever — it
lifted the **base** from 10.8% → **18.2%** on the full 500.

**3. Serving:** one 8-GPU node, `TP=4 × DP=2` (2 replicas), `MAX_TOKENS=16384` (thinking needs
headroom). Point `MODEL` at a **consolidated** HF checkpoint (Phase 2's `consolidate.sh`); if the
SFT dir lacks the tokenizer / chat-template files, copy them from `Qwen/Qwen3-32B` before serving.

### Result — CoderForge SFT lifts the resolve rate

Base vs the CoderForge-SFT (gentler `lr 5e-6`) checkpoints, same harness, all 500 Verified:

| Checkpoint | Resolved / 500 | Resolve rate |
|---|---|---|
| Base `Qwen3-32B` (thinking-on) | 91 | **18.2%** |
| SFT step 99  | 109 | 21.8% |
| SFT step 199 | 100 | 20.0% |
| SFT step 299 | 109 | 21.8% |
| SFT step 399 | 120 | 24.0% |
| **SFT step 499** | **122** | **24.4%** |

CoderForge SFT lifts Qwen3-32B **+6.2 points** (18.2% → 24.4%) — a real SWE-bench gain, and the
gentler LR kept improving through the early checkpoints (the goal of this long-context CP-SFT
validation). _(A first `lr 1e-5` run peaked then regressed, which is why the recipe uses `5e-6`.)_

