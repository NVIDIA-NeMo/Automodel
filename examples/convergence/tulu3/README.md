# Tulu-3 Convergence Pipeline

End-to-end SFT convergence validation on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture).

**Roadmap**: [e2e-robustness-roadmap.md](../../../e2e-robustness-roadmap.md) (Theme 1)

## Pipeline

The pipeline has five stages. Each stage gates the next.

```
Data Pre-filtering → Model Verification → Training → Eval → Inference Quality
```

### 1. Data Pre-filtering

**Why**: When `apply_chat_template(truncation=True)` truncates a sample, the terminal `<|im_end|>` token is silently dropped. The model never sees a complete conversation ending, so it never learns to stop generating — this directly causes death looping at inference time. Truncation is a double-edged sword: without EOS the model death-loops, but appending EOS after truncation teaches the model to end responses abruptly mid-sentence since the content was cut off at an arbitrary point. Pre-filtering avoids both problems by removing over-length samples entirely, so every training sample has a natural, complete conversation ending.

For Tulu-3 with Qwen3 tokenizer at `seq_length=2048`, roughly 5% of samples exceed the limit. Filtering these out and caching the result as Parquet avoids re-filtering on every training run.

The validation step then checks that the data pipeline is correct: attention masks properly mark padding, labels are masked at padding positions, every sample has at least one supervised token, and the EOS token is present in the content.

```bash
# Check truncation rates at multiple seq_lengths
bash data/check_truncation.sh

# Pre-filter dataset and cache as Parquet
bash data/prefilter.sh

# Validate attention mask, label masking, supervised tokens, EOS presence
python data/validate_data.py \
    --dataset allenai/tulu-3-sft-mixture \
    --model Qwen/Qwen3-30B-A3B \
    --seq_length 1024
```

### 2. Model Verification

**Why**: If our model implementation loads weights incorrectly, has a RoPE mismatch, or uses different normalization precision than the reference HF implementation, all downstream results are meaningless. Comparing layer-by-layer activations against HF Transformers on the pretrained checkpoint before any fine-tuning catches these mismatches early — before spending GPU hours on training.

The comparison registers forward hooks on every decoder layer in both models and compares:
- **Per-layer hidden states** — cosine similarity and max absolute difference at the last token position
- **Final logits** — cosine similarity, max absolute difference, and top-1 token agreement

HF Transformers loads with `device_map="auto"`. NeMo loads via torchrun using the same config and code path as training (EP, FSDP, backend).

```bash
# Qwen3 MoE 30B with EP=8 — same config as training
bash model-verification/run.sh

# Or directly:
python model-verification/compare_activations.py \
    --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
    --num-prompts 3 \
    --threshold 0.99
```

Example output (Qwen3 MoE 30B, EP=8, 1 prompt, 48 decoder layers):

```
  Layer-by-Layer Activation Comparison: NeMo AutoModel vs HF Transformers

  Threshold (cosine sim): 0.99
  Prompts: 1

  Prompt 0: 33 tokens | Solve for x: 3x + 7 = 22. Show your work step by step.
    [PASS] Layers: mean_cos=0.999745  worst_cos=0.999363 (layer_3)
           mean_max_diff=0.109105  max_max_diff=1.125000
           Logits: cos=0.999507  max_diff=0.5625  top1_agree=Y

  RESULT: PASS — all prompts above threshold 0.99
```

### 3. Training

```bash
# Single-node torchrun
bash training/launch.sh \
    --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml

# Multi-node SLURM
bash training/launch_slurm.sh \
    --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
    --nodes 2 --partition batch
```

### 4. Eval

**Why — IFEval with thinking mode**: Tulu-3 is a non-reasoning SFT mixture (no chain-of-thought traces). Base thinking models (e.g., Qwen3 with `enable_thinking=True`) perform poorly on IFEval because the model spends tokens in `<think>` blocks rather than following the instruction format constraints. After SFT on Tulu-3, the model should improve on IFEval because it learns to produce direct, format-compliant responses. This makes IFEval a good regression signal — a drop in IFEval accuracy after SFT likely indicates a pipeline bug, not a modeling issue.

We evaluate with `enable_thinking=False` by default since Tulu-3 doesn't train thinking behavior. To verify that thinking-mode performance also improves (or at least doesn't regress), run with `--thinking`.

```bash
# One-time setup
bash eval/setup_lm_eval.sh

# Run evaluation (non-thinking, default)
bash eval/run_eval.sh \
    --model-path checkpoints/LATEST/model/consolidated \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --tasks ifeval

# Optional: compare thinking-mode performance against base model
bash eval/run_eval.sh \
    --model-path checkpoints/LATEST/model/consolidated \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --tasks ifeval \
    --thinking
```

### 5. Inference Quality

**Why**: Benchmark accuracy alone doesn't catch all failure modes. A model can achieve reasonable IFEval scores while still exhibiting pathological behavior on a subset of prompts. The inference quality analysis detects specific failure modes:

- **Death looping** — The model repeats the same sentence or phrase indefinitely. Detected by duplicate sentence ratio (> 30%), single sentence repeating 5+ times, or a contiguous block of 4+ identical sentences. This is the most common failure mode from truncation or EOS bugs.
- **Missing EOS** — The response hits the generation token limit without producing a stop token. Indicates the model didn't learn to terminate.
- **Empty response** — The model produces nothing (or only `<think>` tags with no content). Suggests the SFT data had samples with no supervised tokens.
- **Abrupt ending** — The response cuts off mid-sentence. May indicate the model learned from truncated training samples.

The quality gate loads thresholds from `eval/thresholds.yaml` and exits non-zero if any failure rate exceeds the limit, blocking the pipeline.

```bash
python inference/analyze_quality.py results/ \
    --export results/quality_report.json \
    --threshold-file eval/thresholds.yaml
```

## Model Results

Per-model configs, results, and training curves:

- [Qwen3-4B](models/qwen3-4b/README.md) — Dense 4B, CP=1, FSDP
- [Qwen3 MoE 30B](models/qwen3-moe-30b/README.md) — MoE 30B (3B active), EP=8, FSDP

## Directory Layout

```
data/
  check_truncation.sh    Truncation rate analysis at multiple seq_lengths
  prefilter.sh           Pre-filter and cache dataset as Parquet
  validate_data.py       Token-level correctness assertions

model-verification/
  run.sh                      Stage 2 runner script
  compare_activations.py      Orchestrates NeMo vs HF layer-by-layer comparison
  extract_hf_activations.py   HF Transformers activation extraction (device_map=auto)
  extract_nemo_activations.py NeMo activation extraction (torchrun, same code path as training)

training/
  launch.sh              torchrun launch wrapper
  launch_slurm.sh        SLURM sbatch launch wrapper

eval/
  setup_lm_eval.sh       One-time lm-evaluation-harness setup
  run_eval.sh            lm_eval with vLLM backend

inference/
  analyze_quality.py     Inference failure mode detection and reporting

models/
  qwen3-4b/             Qwen3-4B configs, results, training curves
  qwen3-moe-30b/        Qwen3 MoE 30B configs, results, MoE metrics
```
