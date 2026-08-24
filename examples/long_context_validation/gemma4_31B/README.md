# CoderForge Convergence Pipeline (CP validation)

End-to-end SFT pipeline on [togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
to validate **context parallelism (CP)** in NeMo AutoModel, then evaluate on
[SWE-bench Verified](https://www.swebench.com/verified).

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
  check_masking.py       Eyeball the assistant-only label mask on real training samples
gemma4_coderforge_chat_template.jinja
                         Gemma4 template with {% generation %} blocks for direct assistant masking
```

### Why prefilter (don't truncate)

When `apply_chat_template(truncation=True)` truncates a trajectory, the terminal
turn marker (`<turn|>`, token 106) is silently dropped. The model never sees a
complete turn ending and learns to never stop → death-looping at inference. We
therefore **drop** over-length trajectories rather than truncate, so every
training sample ends on a complete turn.

### Choosing the sequence length (why 64K)

Because we drop rather than truncate, `seq_length` directly sets how much data
survives. `prefilter_dataset.py` tokenizes every trajectory once and prints the
retention curve so the choice is data-driven. On the 155,144 `filtered_reward1`
trajectories (Gemma4 tokenizer; median ~43.5K, p95 ~80K, max ~187K tokens):

| seq_length | retention |
|---|---|
| 16K | ~0% (18 trajectories) |
| 32K | 17% |
| 49K | 64% |
| **64K** | **87.1%** |
| 96K | 98.5% |
| 128K | 99.8% |

We chose **64K** as the balance between data retention (87%) and the CP/memory
topology that fits on 128 GPUs (`cp8`, 8K tokens/rank).

**To train at a longer context than 64k, just re-run the data stage at a higher
`SEQ_LENGTH`.** The analyzed cache stores each trajectory's token count, so a
higher-length pass is a cheap re-filter (no re-tokenization) that emits a new
`data.jsonl`; point the recipe's `dataset.path_or_dataset_id` +
`dataset.seq_length` at it and raise `cp_size` so `seq_length` stays divisible by
`2 * cp_size`.

### Gemma4 specifics (verified)

- **Tokenizer/template** come from the local checkpoint dir (`chat_template.jinja`
  + `tokenizer.json`). Point `--model` at it. The base `google/gemma-4-31B` ships
  **no `chat_template.jinja`** — copy it from `google/gemma-4-31B-it` into the base
  checkpoint dir before running data prep or training (the tokenizer is identical).
- **Assistant masking uses `gemma4_coderforge_chat_template.jinja`** (in this
  directory), a copy of the stock Gemma4 template with every assistant turn's body
  wrapped in `{% generation %}...{% endgeneration %}`. The recipe, `validate_data.py`,
  and `check_masking.py` point `chat_template` at it, so
  `apply_chat_template(return_assistant_tokens_mask=True)` returns the assistant
  mask directly (one `apply_chat_template` call per sample). The rendered token
  stream is byte-identical to the stock template.
  The stock Gemma4 template has **no `{% generation %}` block**, so `ChatDataset`
  would fall back to `_build_multiturn_assistant_mask` — an O(turns) per-sample
  re-tokenization that #3024's prefix-consistency guard rejects for multi-turn
  conversations whose rendered prefixes are not exact token prefixes of the full
  render. The generation-block template is the supported path.
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
MODEL=/path/to/hf_gemma4_31b bash data/prefilter.sh

# 2. Produce a training-ready cache at the chosen seq_length (cheap re-filter, no
#    re-tokenization — so a larger seq_length later is a quick second run).
MODEL=/path/to/hf_gemma4_31b SEQ_LENGTH=65536 bash data/prefilter.sh

# 3. Validate the cache through the exact ChatDataset training path.
python data/validate_data.py \
    --dataset data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq65536/data.jsonl \
    --model /path/to/hf_gemma4_31b \
    --seq_length 65536 --num-samples 200
```

### Check the label mask (assistant-only supervision)

`validate_data.py` asserts token-level invariants; `check_masking.py` is the
human-eyeball companion. Run it **after building the cache**, and especially
whenever a training run shows *"loss down but downstream capability down"* — that
symptom is only a valid finding if the mask is correct. If masking is broken
(≈0% supervised → nothing learned; ≈100% → the model also imitates user/tool
text; or the wrong spans), then loss-down/capability-down is a **bug, not a
finding**. For a few samples it prints the supervised fraction (`labels != -100`),
whether the Gemma4 stop token `<turn|>` (106) lands inside supervised spans, and a
decoded supervised span vs a masked span so you can confirm assistant content
(incl. `tool_calls`) is supervised while system/user/tool-result content is masked.

The tokenizer and default JSONL are module constants (`GEMMA4`, `JSONL`) at the top of the
script — edit them if your paths differ, or pass `--jsonl`:

```bash
python data/check_masking.py \
    --jsonl data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq65536/data.jsonl \
    --seq-length 65536 --n 4
```

The output `data.jsonl` plugs into a training config:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: data/cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq65536/data.jsonl
  seq_length: 65536
```

## Phase 2 — Training recipe (Gemma4 31B + CP)

`gemma4_31b_base_coderforge_cp8_64k_1e5_800steps.yaml` — SFT on the **base**
`google/gemma-4-31B`, on 16 nodes / 128 GPUs, `cp8 × dp16`, `gbs=16`, 64K
sequence length, `FusedLinearCrossEntropy`, and the `ChatDataset` + THD collate
path (via `packed_sequence_thd_collater_vlm`) that preserves tools. It sets
`save_consolidated` so the resulting HF checkpoint is SWE-bench-evaluable.

The base model must learn the Gemma4 tool-call special tokens (`<|tool_call>`=48
/ `<tool_call|>`=49) from scratch, so this uses `lr 1e-5` with a 60-step warmup
over `max_steps=800` (~0.5B tokens) and `clip_grad_norm=1.0` (the base has
volatile early grads). `freeze_language_model: false` keeps the embeddings + tied
LM head trainable — required to learn 48/49.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/long_context_validation/gemma4_31B/gemma4_31b_base_coderforge_sft.png" alt="Gemma4-31B base SFT training loss curve on CoderForge" width="700">
</p>

## Phase 3 — SWE-bench Verified evaluation

Goal: check whether CoderForge SFT gives the **raw base** `google/gemma-4-31B` (a
pretrained model with *no* instruction/agent tuning) any **agentic** ability at
all, versus the un-SFT'd base. The scripts referenced below live in
[`eval/`](./eval); parameterize the `/path/to/...` placeholders and `<account>`
for your cluster.

**Scaffold.** The [OpenHands](https://github.com/All-Hands-AI/OpenHands) v0.52.1
3-tool surface (`execute_bash` / `str_replace_editor` / `finish`) — the exact tools
CoderForge trajectories were generated on — is presented to the served checkpoint
(vLLM, with the built-in `gemma4` tool-call parser **disabled** — `NOPARSER=1`; see
"Why NOPARSER=1" below) and executed against each SWE-bench instance's repo inside a
**Docker-less enroot** container. Grading is local-in-enroot with the official
`swebench` spec (gold patches → 5/5 resolved, so the grader is trusted).

**Steps** (from `eval/`):

```bash
bash 00_setup_eval_tooling.sh                    # py3.10 venv + agent + swebench harness
SUBSET=verified sbatch 06_prewarm_images.sub     # pre-import instance images once (CPU)

# Serve each checkpoint + run the 3-tool agent -> preds.json (serve+agent on one 8-GPU node).
# NOPARSER=1 is REQUIRED for these checkpoints (see "Why NOPARSER=1" below); serve both the base
# and the SFT identically so the base-vs-SFT delta reflects the model, not the serving config.
RUN_TAG=oh3_base NAME=gemma4base MODEL=<base google/gemma-4-31B consolidated> \
  SLICE=0:500 SUBSET=verified NOPARSER=1 MAX_TOKENS=16384 TP=2 DP=4 WORKERS=16 \
  sbatch --gpus-per-node=8 07_openhands3_run.sub
RUN_TAG=oh3_sft  NAME=gemma4cf   MODEL=<CoderForge-SFT consolidated> \
  SLICE=0:500 SUBSET=verified NOPARSER=1 MAX_TOKENS=16384 TP=2 DP=4 WORKERS=16 \
  sbatch --gpus-per-node=8 07_openhands3_run.sub

# Grade both locally (always confirm enroot-errs=0 before trusting a 0.0 resolve)
PREDS=<runs>/oh3_base/preds.json RUN_TAG=grade_base sbatch 05_grade_enroot.sub
PREDS=<runs>/oh3_sft/preds.json  RUN_TAG=grade_sft  sbatch 05_grade_enroot.sub
```

**Why `NOPARSER=1`.** A tool call is only usable if something parses the model's raw
text into a structured `{name, arguments}` call. Both the raw base and the CoderForge
SFT emit their arguments in the **base model's pretrained JSON prior** —
`call:name{"command":"...","path":"..."}` (quoted JSON) — because 800 SFT steps don't
overwrite that prior. vLLM's `--tool-call-parser gemma4` expects the *native* unquoted
syntax that the **gemma4-`it` (instruction-tuned)** model emits (`{command:<value>}`),
and **mangles** the JSON form: it folds the
leading `{"` into the key (`{"command"`), so every argument comes out corrupted
(measured: 727/727 `execute_bash` returned non-zero, 349/349 editor calls rejected —
0% usable). `NOPARSER=1` disables that parser (and sets `tool_choice=none`, so vLLM
still renders the tool schemas into the prompt but doesn't try to parse/force a call);
the raw `call:name{...}` then reaches the response `content`, where `oh3_run.py`'s
`json.loads`-first fallback recovers the JSON dict cleanly. Because base and SFT share
the same JSON prior, **both** must run with `NOPARSER=1` — and serving them identically
is what makes the base-vs-SFT comparison fair. (An `-it`/instruct checkpoint that emits
Gemma-native args is the opposite case: leave the parser **on**, i.e. omit `NOPARSER`.)

**Base serve note.** The raw base checkpoint is missing the `-it` model's serving
fields, so two things must be set up before it can tool-call at all:

- **`response_schema`** — a field in the `-it` `tokenizer_config.json` that declares the
  tool-call *text format* (the `call:name{...}` envelope, regex
  `call:(?P<name>\w+)(?P<arguments>\{.*\})`) so vLLM can recognize a tool call in the
  output stream. The base config doesn't have it → **copy the `response_schema` block
  from the `-it` config into the base's `tokenizer_config.json`**. (`oh3_run.py` mirrors
  this same regex host-side, so the `NOPARSER=1` fallback recovers calls too.)
- **`stop_token_ids=[1,106,50]`** — Gemma4's per-turn stop is `<end_of_turn>` = token
  **106**, but the base's `generation_config` only lists `eos` = token **1**; without
  forcing 106 the model runs past its turn boundary and never yields control back to the
  agent. So `oh3_run.py` forces the stop set `[1, 106, 50]` (eos, turn-stop, and one more
  Gemma stop marker) so each turn ends cleanly. Models with a correct `generation_config`
  (e.g. Qwen3) instead set `OH3_STOP_TOKEN_IDS=""` to use their own `eos`.

**`probe_indist.py`** is a one-call served smoke test: it hits the running vLLM **once**
with the 3-tool schema and checks whether the checkpoint emits a real structured
`call:name{...}` tool call versus just planning in prose — i.e. it confirms the serving
config (`response_schema` + `stop_token_ids` + parser choice) is correct, and that the
checkpoint has crossed over from text-only planning to structured tool calls, **before**
you commit to a full 500-instance run.

### Result — SFT installs agentic behavior; the raw base has none

Base vs the **step-799** CoderForge SFT checkpoint, same OpenHands 3-tool scaffold,
on all **500 SWE-bench Verified** tasks (7,972 assistant turns, 14,011 tool calls):

| | Base `gemma-4-31B` (un-SFT'd) | CoderForge SFT (step 799) |
|---|---|---|
| Structured `tool_calls` | **0** (never emits one) | **7972 / 7972 turns = 100%** |
| Tools used | none | `think` + `execute_bash` + `str_replace_editor` |
| Turns per task | ~33 identical planning turns, then aborts | mean **15.9**, median **15** (range 2–56) |
| Arg fidelity | — | **clean** (0% `\uXXXX` garbage, 0/500 trajectories) |
| Terminates (`finish`) | never acts (`never_toolcalled`) | 0 / 500 (candidate patches produced, none applied) |
| Resolved (official `swebench`) | — | **0 / 500 = 0.0%** (grading valid, enroot-errs 0) |

**Base — never acts.** The raw base emits **zero** tool calls and repeats the same
planning checklist verbatim every turn ("Phase 1. READING… 1.1 … 1.2 …") until the
harness aborts (`never_toolcalled`) — a property of the raw pretrained model, not the
checkpoint. It plans forever and never acts.

**SFT — real agentic behavior.** After CoderForge SFT the step-799 model opens each
task with a `think` that diagnoses the bug, then drives real exploration: across all
500 Verified tasks **100% of turns (7972/7972) are structured OpenHands tool calls**
(`execute_bash` ×9423, `str_replace_editor` ×4088, `think` ×500; mean 15.9 turns/task,
args clean — 0/500 garbled). It does not yet emit `finish`, so runs end without a
landed patch: candidate diffs are produced but do not apply cleanly, and the official
`swebench` grader resolves **0/500** (grading valid, enroot-errs 0).

**Takeaway.** CoderForge SFT **installs the agentic process** (tool-calling,
think-first, correct targeting) into a base model that had none — the goal of this
long-context CP-SFT validation.
