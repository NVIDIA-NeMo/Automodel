# Qwen3-MoE-30B (A3B) — Tulu-3 SFT Run (TE FusedAdam)

Reproducible runbook for the `qwen3_moe_30b_ep8_te_fusedadam.yaml` convergence run:
pre-filter → train (1000 steps, EP=8, 8×GPU) → eval (IFEval).

Run **inside the automodel container** (interactive, 8×H100, `/opt/venv` active) from the
repo root. Assumes `HF_HOME`, `WANDB_API_KEY`, and network access are available.

> **Differences from the Moonlight runbook** (this is where Qwen bites you):
> 1. Standard Qwen tokenizer — **no TikToken gotcha** (still pass `HF_HUB_OFFLINE=0` only
>    because the tokenizer isn't cached yet; it loads cleanly, no slow-tokenizer fallback).
> 2. Eval MUST use `--thinking` + `think_end_token=</think>`, else ~40% empty responses
>    (see step 4). No `trust_remote_code` needed (Qwen3 is native in transformers).
> 3. The 30B (~57 GB) on 1 GPU needs `max_model_len=8192` or vLLM engine init OOMs the KV cache.

---

## 1. Pre-filter the dataset (seq_length=2048)

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
MODEL=Qwen/Qwen3-30B-A3B-Base SEQ_LENGTH=2048 \
  bash examples/convergence/tulu3/data/prefilter.sh
```

Keeps ~893,536 / 939,343 samples (~4.9% removed). Note the printed cache path, e.g.:

```
examples/convergence/tulu3/data/cached/allenai_tulu-3-sft-mixture_train_seq2048_Qwen3-30B-A3B-Base_<hash>
```

## 2. Train

```bash
export NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 NVTE_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Point this at the cache dir printed by step 1:
CACHED="$(pwd)/examples/convergence/tulu3/data/cached/allenai_tulu-3-sft-mixture_train_seq2048_Qwen3-30B-A3B-Base_<hash>"

torchrun --nproc-per-node 8 --tee 3 examples/llm_finetune/finetune.py \
    --config examples/convergence/tulu3/models/qwen3-moe-30b/qwen3_moe_30b_ep8_te_fusedadam.yaml \
    --model.pretrained_model_name_or_path Qwen/Qwen3-30B-A3B-Base \
    --dataset.path_or_dataset_id "$CACHED" \
    --validation_dataset.path_or_dataset_id "$CACHED" \
    --validation_dataset.split "train[:128]" \
    --wandb.enable true \
    --wandb.project convergence \
    --wandb.entity nvidia \
    --wandb.name qwen3_moe_30b_fused_te \
    --wandb.dir /tmp/wandb
```

Key config values (from the YAML): `global_batch_size=128`, `local_batch_size=2`,
`max_steps=1000`, `val_every_steps=100`, `seq_length=2048`, `ep_size=8`, TE attn+linear,
`rms_norm: torch_fp32`, `FusedLinearCrossEntropy` loss (avoids the bf16→fp32 logit upcast OOM),
TE FusedAdam (`lr=1e-5`, betas `[0.9, 0.95]`), `save_consolidated: final`.

- Model weights (~57 GB, 16 shards) download at startup on first run, then cache.
- Consolidated checkpoint written automatically at the final step to:
  `checkpoints_convergence/qwen3_moe_30b_te_fusedadam/epoch_0_step_999/model/consolidated`

**Reference (single 8×H100 node):** converged loss 0.94→0.66, val_loss ≈ 0.82, no NaN/spikes.

## 3. One-time eval setup

`/opt` is **not persistent** across container recycles/day boundaries — rebuild each new session.

```bash
# (a) Bootstrap uv (absent in the container; `pip install uv` does not yield a working binary)
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# (b) Install lm-evaluation-harness (fresh clone + isolated venv + vLLM) at /opt

bash examples/convergence/tulu3/eval/setup_lm_eval.sh /opt/lm-evaluation-harness

# (c) Remove torchcodec — its .so fails to load (missing FFmpeg libs) and crashes vLLM's
#     guarded video import (raises OSError, not ImportError).
uv pip uninstall --python /opt/lm-evaluation-harness/.venv/bin/python torchcodec
```

## 4. Evaluate (IFEval)

```bash
CKPT_ROOT="$(readlink -f checkpoints_convergence/qwen3_moe_30b_te_fusedadam/LATEST)"
CKPT="$CKPT_ROOT/model/consolidated"

bash examples/convergence/tulu3/eval/run_eval.sh \
    --model-path "$CKPT" \
    --tokenizer Qwen/Qwen3-30B-A3B \
    --tasks ifeval \
    --tp-size 1 --dp-size 1 \
    --thinking \
    --extra-model-args "max_model_len=8192,think_end_token=</think>" \
    --gen-kwargs "until=<|im_end|>"
```

Why each Qwen-specific flag matters:
- **`--thinking`** (enable_thinking=True): without it, the Qwen3 chat template pre-injects an
  empty `<think>\n\n</think>` block. The model was trained on non-thinking Tulu-3 (the training
  template only emits `<think>` when `reasoning_content` is present), so that prefill is OOD and
  ~40% of responses come back empty → IFEval collapses to ~0.27 (below the pretrained base). With
  `--thinking` the prompt is a bare `<|im_start|>assistant\n`, matching training.
- **`think_end_token=</think>`**: required by newer lm-eval whenever `enable_thinking=True`.
- **`max_model_len=8192`**: the 30B on 1 GPU can't fit KV cache for the default 32768 context.
- **`until=<|im_end|>`**: explicit stop token for the current vLLM/lm-eval stack.

**Reference IFEval (this run, TE FusedAdam SFT):**

All "measured" rows below use the identical stack (`--thinking`, `think_end_token=</think>`,
`until=<|im_end|>`, `max_model_len=8192`, tokenizer `Qwen/Qwen3-30B-A3B`, tp=1/dp=1).

| Model | prompt_strict | prompt_loose | inst_strict | inst_loose |
|-------|-------------:|-------------:|------------:|-----------:|
| Qwen3-30B-A3B-Base — pretrained (README) | 0.318 | 0.420 | 0.441 | 0.543 |
| README — TE FusedAdam SFT | 0.545 | 0.590 | 0.714 | 0.750 |
| **This run — TE FusedAdam SFT (measured)** | **0.5767** | **0.6100** | **0.6894** | **0.7182** |

Note: the README-quoted pretrained row (0.318) was measured without `--thinking` and understates
the base; the true same-stack SFT gain is **+0.14 absolute**, uniform across all four metrics.
