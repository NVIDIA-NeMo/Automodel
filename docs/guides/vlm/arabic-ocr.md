# Fine-Tune Qwen3.8-27B for Arabic OCR

## Introduction

This guide covers LoRA fine-tuning of a Qwen3.5-family VLM to transcribe
scanned Arabic pages into text. It specifies the on-disk dataset format,
the conversation schema the trainer expects, and the configuration
choices that a page-image OCR task forces — most of which differ from
the VQA defaults shipped in the other VLM recipes.

The reference corpus is 2000 page-image / transcription pairs of Arabic
print — books, exam papers, official forms — with diacritics (tashkeel)
present on 57% of pages.

**Task at a glance**

- Input: one page image per sample, native resolution, aspect ratio preserved.
- Output: the page's full text, verbatim, as plain text.
- Adapted: the language backbone via LoRA. The vision tower is frozen.
- Supervised: only the assistant turn — roughly 9% of the tokens in a sample.

---

## Dataset Format

### On disk

A single flat directory of `<id>.png` / `<id>.md` pairs:

```
accepted/
├── 1003.png      # page image
├── 1003.md       # ground-truth transcription
├── 1008.png
├── 1008.md
└── ...
```

The `<id>` is any string; pairing is by exact stem match. Files without
a counterpart are skipped with a warning rather than failing the run.

Despite the `.md` extension the transcriptions are **plain text, not
structured Markdown**. In the reference corpus, Markdown headings appear
in 0.2% of files and tables in 0.4%. Do not write a prompt that promises
Markdown structure the references do not contain.

### Transcription conventions

The reference transcriptions follow four conventions. They are stated in
the system prompt so the prompt and the targets agree:

| Convention | Meaning | Frequency |
|---|---|---|
| `image : N` on its own line | A figure/photo appears here; numbered from 1 down the page | 37.0% |
| `[REDACTED]` | Personal information struck out of the source | 11.4% |
| Blank line between blocks | Paragraph separation | 76.8% |
| Digits as printed | Arabic-Indic (٠-٩) or Latin (0-9), never converted | 68.2% Latin / 37.5% Arabic-Indic |

Transcriptions preserve source errors. OCR fidelity means reproducing
what is on the page, including misspellings, so the system prompt
explicitly forbids normalizing or correcting.

### Validation rules

`_collect_pairs` drops a sample when:

- the matching `.png` is missing, or
- the `.md` is empty or whitespace-only.

The second rule matters more than it looks. An empty transcription
produces an assistant turn with no tokens, every label in the sample is
masked to `-100`, and the sample contributes no loss while still costing
a full forward pass. The reference corpus has 2 such files, leaving 1998
usable pairs.

---

## Conversation Schema

`ArabicOcrDataset.__getitem__` returns the same shape as the other VLM
datasets in this package, so `default_collate_fn` handles it unchanged:

```python
{
    "conversation": [
        {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [{"type": "image", "image": <PIL.Image>},
                                          {"type": "text",  "text": <sampled prompt>}]},
        {"role": "assistant", "content": [{"type": "text",  "text": <transcription>}]},
    ]
}
```

Images are opened lazily — `Image.open` reads only the PNG header, and
the pixel decode happens in the DataLoader worker on access. Building the
dataset therefore holds neither 2000 decoded pages nor 2000 open file
handles.

### Prompts

One system prompt, fixed across every sample, states the task and the four
transcription conventions above. Because it is fixed, **inference must use
the same system prompt** or the model is off-distribution.

Nine user prompts (six English, three Arabic) are sampled per example so
the model does not overfit to a single phrasing. Selection is
`random.Random((seed, idx))` — deterministic per index, so a page draws
the same instruction on every epoch and across a resume, rather than being
re-rolled each pass.

Both are overridable from YAML (`dataset.system_prompt`, `dataset.prompts`)
without editing the module.

---

## Configuration

### Image resolution

Qwen uses native dynamic resolution: aspect ratio and size are preserved,
and what is bounded is the **number of visual tokens**. One visual token
covers `patch_size × merge_size` = 16 × 2 = 32 px square, i.e. 1024 native
pixels. `size.longest_edge` is a total pixel budget, not a side length.

The checkpoint default of `16777216` (16.7 MP ≈ 16384 visual tokens) is
far too generous for page scans, whose sizes vary enormously:

| Cap | median | p90 | max |
|---|---|---|---|
| 16.7 MP (default) | 1900 | 11750 | 16428 |
| **2 MP (recommended)** | **1900** | **2040** | **2091** |
| 1.31 MP | ~1250 | ~1280 | ~1290 |

At the default, step time and memory swing by an order of magnitude
between samples. Capping at 2 MP leaves the median page untouched — 74%
of the corpus is already below it — while pulling 8000×6000 outliers down
from 16428 tokens to 2091:

```yaml
processor:
  size:
    longest_edge: 2097152   # 2048 visual tokens
    shortest_edge: 65536
```

Do not cap more aggressively without checking your corpus. Diacritics are
small marks and are the first thing lost to downsampling.

### Sequence length and padding

Total sequence = vision tokens + transcription tokens + ~237 template
overhead (system prompt ~200, user prompt 9–17, `<think>` marker 4, role
markers). For the reference corpus at a 2 MP cap:

| | tokens |
|---|---|
| median | 2244 |
| p90 | 2762 |
| p99 | 3471 |
| max | 5789 |

**Do not set `max_length` on the collate function.** Setting it switches
`apply_chat_template` to `padding="max_length"`, which:

1. pads *every* sample to the cap — at 4096 against a 2006-token mean,
   51% of every step is spent on padding; and
2. right-truncates anything over the cap, cutting the *end* off the
   transcription and training the model to stop mid-page.

Without it, batches pad to their own longest and nothing is truncated
(`tokenizer.model_max_length` is 262144). Removing a 4096 cap measured
~50% more throughput per sample.

The worst case is text-driven, not image-driven: the longest sample in the
reference corpus is 1998 vision + 3554 transcription tokens.

### Loss

The vocabulary is 248320. With multi-thousand-token sequences, a plain
cross-entropy materializes a `[seq, 248320]` fp32 logits tensor — 3.79 GiB
per micro-batch at 4096 — which OOMs on top of the sharded weights. Use
the fused kernel, which applies the `lm_head` and the CE in chunks and
never builds it:

```yaml
model:
  text_config:
    use_cache: false
    output_hidden_states: true   # required: the loss consumes hidden states

loss_fn:
  _target_: nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
  reduction: sum
```

`ChunkedCrossEntropy` does not solve this — it chunks the softmax but
still materializes the logits.

`FusedLinearCrossEntropy` requires `cut-cross-entropy`, which this repo
declares in the `dev` dependency group pinned to an Apple `ml-cross-entropy`
git revision. Install it with `uv sync --locked --all-groups`, not
`uv pip install cut-cross-entropy` — the PyPI package is a different build
from the pinned revision.

### Thinking mode

Qwen3.5's chat template defaults to `reasoning_effort='xhigh'` and prepends
"Reasoning effort is set to xhigh. Please think carefully…" to the system
turn, while the assistant turn it builds for a plain text target contains
an *empty* `<think>` block. Training on that pairing teaches the model to
ignore an explicit instruction to reason, and spends ~50 tokens of every
sample's context saying so.

`arabic_ocr_collate_fn` passes `enable_thinking=False`, which drops the
preamble. The empty `<think>\n\n</think>` marker on the assistant turn
remains — the template emits it for the final assistant turn regardless of
`enable_thinking`, `preserve_thinking`, or `reasoning_effort`. That marker
is supervised, which is correct: it is Qwen's native non-thinking format,
and the model must learn to emit it for train/serve consistency.

**When serving, strip the leading `<think>\n\n</think>\n\n` from generated
output** before treating it as the transcription.

### LoRA scope

```yaml
peft:
  match_all_linear: false
  exclude_modules: ["*vision_tower*", "*vision*", "*visual*",
                    "*image_encoder*", "*lm_head*", "*audio*", "*mtp*"]
  dim: 16
  alpha: 32
```

With `match_all_linear: false` and a non-empty `exclude_modules`, the
matcher falls through to "all `nn.Linear` except those excluded". Without
`*mtp*`, that sweeps in the multi-token-prediction head used for
speculative decoding, which is irrelevant here and complicates the merge
and serve steps.

The vision tower is frozen. For a model already competent at Arabic script
this is the right default — initial loss on the reference corpus is ~0.5,
not the ~2.0 of a task being learned from scratch. If transcription quality
plateaus on difficult scans, unfreezing the vision tower is the first lever.

---

## Running

```bash
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

automodel --nproc-per-node=8 \
  examples/vlm_finetune/qwen3_8/qwen3_8_27b_lora_arabic_ocr.yaml
```

On a node whose system CUDA is newer than the one PyTorch bundles, the
`LD_LIBRARY_PATH` prefix is required. Otherwise the system `libcublasLt`
wins and the first bias-epilogue GEMM fails with
`CUBLAS_STATUS_NOT_INITIALIZED` — which surfaces at the vision tower's
`attn.qkv`, the first `nn.Linear` with a bias, and looks nothing like a
library-version problem.

### Measured behavior

8 × A100-80GB, `local_batch_size: 2`, `global_batch_size: 32`
(gradient accumulation 2), dynamic padding:

| | |
|---|---|
| Memory | 28–34 GiB allocated, 46–61 GB reserved |
| Throughput | ~1500 tok/s aggregate |
| Step time | ~44 s (32 samples per optimizer step) |
| Steps | 63 per epoch (1998 samples / 32) |

`nvidia-smi` reports substantially more than the trainer's allocated
figure — the difference is allocator reserve, fragmentation, and NCCL
buffers. Size batches against the reserved number, not the logged one.

`local_batch_size: 4` fits on average (~66 GiB extrapolated) but is not
safe with dynamic padding, because memory tracks tokens rather than
samples and sequences reach 5789 against a 2244 median. Prefer gradient
accumulation: it gives an identical effective batch and an identical step
count at half the peak.

Set `num_workers: 0`. Forked DataLoader workers inherit the rank's CUDA
context and appear as a second ~40 GiB process per GPU. GPU utilization
stays at 98–100% without them, so they buy nothing here.

---

## Checkpoints and Export

`checkpoint.enabled` defaults to `false` in several example configs — with
it off, nothing is written and the adapter is lost when the process exits.
Set it, and let the step scheduler place the saves:

```yaml
step_scheduler:
  max_steps: null              # derive from num_epochs * epoch_len
  ckpt_every_steps: null       # defaults to epoch_len -> one save per epoch
  save_checkpoint_every_epoch: true

checkpoint:
  enabled: true
  checkpoint_dir: <a durable path, not /tmp>
```

A run auto-resumes from `checkpoint_dir`. Reusing a directory silently
continues the previous run — and if its step count already exceeds
`max_steps`, the new run exits immediately having done nothing.

PEFT checkpoints bypass consolidation and always write a HuggingFace-format
adapter directly under `model/`:

```
epoch_0_step_62/model/
├── adapter_config.json
├── adapter_model.safetensors
├── automodel_peft_config.json
└── tokenizer / processor artifacts
```

`save_consolidated` has no effect for PEFT runs.

### Merging

```bash
python tools/merge_lora.py \
  --base-model Qwen/Qwen3.8-27B \
  --adapter-path <ckpt>/model/ \
  --output-dir <merged>/ \
  --dtype bfloat16 --device cpu \
  --model-class AutoModelForImageTextToText
```

`--model-class` is **required** for a VLM adapter. The adapter records
`task_type: CAUSAL_LM`, so without it PEFT loads the text-only backbone
and every `model.language_model.*` target fails to resolve.

Merging is optional for HuggingFace inference — `AutoPeftModelForCausalLM`
loads the adapter directly. It is needed for runtimes whose LoRA support
does not cover these module types: the adapter spans gated-DeltaNet
projections (`in_proj_a/b/z/qkv`), which are well outside what serving
stacks typically patch.

---

## Metrics

Loss starts near 0.5 and only 9% of tokens carry loss (median 199
supervised of 2244), so per-step loss is noisy — read a moving average,
not the raw series.

Per-step metrics land in `<checkpoint_dir>/training.jsonl`. That file is
appended, never truncated, so a directory reused across runs holds several
runs back to back with the step counter resetting; split on a
non-increasing step before plotting.
