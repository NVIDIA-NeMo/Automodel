# Adding a New Model to the Convergence Pipeline

This guide walks through adding a new model to the Tulu-3 convergence pipeline. We use Qwen3-4B as the running example.

## 1. Create a Model Directory

```
models/<model-name>/
  README.md
  chat_template.jinja           # Custom chat template (if needed)
  <model>_cp1_flashoptim.yaml   # FlashAdamW config
  <model>_cp1_te_fusedadam.yaml # TE FusedAdam config
```

## 2. Get the Chat Template

Thinking models need a custom chat template that strips `<think>` tags from training labels. Without it, SFT on non-reasoning data produces high death loop rates.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Your/Model", trust_remote_code=True)
print(tokenizer.chat_template)
```

Save as `models/<model-name>/chat_template.jinja`. The key modification: wrap thinking content in `{% generation %}` / `{% endgeneration %}` tags so it appears in training but the model learns to produce direct answers.

## 3. Establish Baselines

Run eval on the **pretrained model** (no SFT) before doing anything else:

```bash
bash eval/setup_lm_eval.sh  # one-time

bash eval/run_eval.sh \
    --model-path Your/Model-Base \
    --tokenizer Your/Model-Base \
    --tasks ifeval --tp-size 1 --dp-size 1 \
    --output-path /tmp/eval_baseline
```

Run inference quality analysis on the baseline:

```bash
python inference/analyze_quality.py /tmp/eval_baseline/ --export /tmp/baseline_quality.json
```

Record the baseline numbers — IFEval accuracy, death loop rate, abrupt ending rate.

## 4. Check Truncation and Pre-filter

```bash
python data/check_truncation.py \
    --dataset allenai/tulu-3-sft-mixture \
    --model Your/Model \
    --seq_length 1024 2048 4096 8192
```

Choose `seq_length` where truncation < 5%. Then pre-filter and cache:

```bash
python data/prefilter_dataset.py \
    --dataset allenai/tulu-3-sft-mixture \
    --model Your/Model \
    --seq_length 2048 \
    --cache_dir /tmp/tulu3_filtered
```

The cached Parquet directory is passed to training via `--dataset.path_or_dataset_id`.

## 5. Write Training Configs

Start from an existing config (e.g., `models/qwen3-4b/qwen3_4b_cp1_flashoptim.yaml`).

**Must change:**
- `model.pretrained_model_name_or_path`
- `checkpoint.checkpoint_dir`
- `chat_template` paths in `dataset` and `validation_dataset`

**Consider changing:**
- `dataset.seq_length` — based on truncation analysis
- `step_scheduler.local_batch_size` — based on GPU memory. Use 2 for TE FusedAdam (FP32 master weights need more memory for checkpoint consolidation)
- `distributed.cp_size` / `ep_size` — based on model size and architecture
- `optimizer.lr` — 1e-5 is a good starting point

**For MoE models:**
- Set `distributed.ep_size` (typically 8 for single-node)
- Add `model.backend.experts: torch_mm` and `model.backend.dispatcher: deepep`
- Add `model.backend.rms_norm: torch_fp32` for numerical stability
- Set `model.backend.rope_fusion: false` if using CP

## 6. Validate Data Pipeline

```bash
python data/validate_data.py \
    --dataset allenai/tulu-3-sft-mixture \
    --model Your/Model \
    --seq_length 2048 \
    --num-samples 500
```

All 5 assertions must pass:
- `attention_mask_shape` — contiguous 1s then 0s
- `labels_masked_in_padding` — labels == -100 where attention_mask == 0
- `has_supervised_token` — at least 1 supervised token per sample
- `no_eos_in_padding` — no eos_token_id in padding positions
- `eos_in_content` — eos_token_id present in content

## 7. Run Model Verification

Compare your NeMo implementation against HF Transformers before training:

```bash
python model-verification/compare_activations.py \
    --config models/<model-name>/<config>.yaml \
    --num-prompts 3 \
    --threshold 0.99
```

All decoder layers should have cosine similarity > 0.99. If not, investigate weight loading, RoPE, or normalization differences.

## 8. Train

```bash
CACHED=/tmp/tulu3_filtered/<cached_dir>

source /opt/venv/bin/activate
torchrun --nproc-per-node 8 --tee 3 \
    examples/llm_finetune/finetune.py \
    --config examples/convergence/tulu3/models/<model-name>/<config>.yaml \
    --model.pretrained_model_name_or_path Your/Model-Base \
    --dataset.path_or_dataset_id "$CACHED" \
    --validation_dataset.path_or_dataset_id "$CACHED" \
    --validation_dataset.split "train[:128]"
```

Monitor: loss should decrease from ~0.8 to ~0.5-0.6 over 1000 steps. Watch for NaN/Inf in loss or grad_norm.

**Important:** Use absolute paths for `--checkpoint.checkpoint_dir`. If using TE FusedAdam, set `local_batch_size: 2` to leave memory headroom for checkpoint consolidation.

## 9. Evaluate

Run IFEval with thinking off and on:

```bash
CKPT="$(readlink -f checkpoints/<run>/LATEST)/model/consolidated"

bash eval/run_eval.sh \
    --model-path "$CKPT" \
    --tokenizer Your/Model \
    --tasks ifeval \
    --tp-size 1 --dp-size 1 \
    --output-path /tmp/eval_sft_off

bash eval/run_eval.sh \
    --model-path "$CKPT" \
    --tokenizer Your/Model \
    --tasks ifeval \
    --tp-size 1 --dp-size 1 \
    --thinking \
    --output-path /tmp/eval_sft_on
```

**Important:** Use `readlink -f` to resolve LATEST symlink. Use absolute paths. For dense models, use `--dp-size 1`.

## 10. Run Inference Quality Analysis

```bash
python inference/analyze_quality.py \
    "$(find /tmp/eval_sft_off -name 'samples_*.jsonl')" \
    --export /tmp/quality_off.json

python inference/analyze_quality.py \
    "$(find /tmp/eval_sft_on -name 'samples_*.jsonl')" \
    --export /tmp/quality_on.json
```

Compare against baselines (step 3). Check all 4 failure modes:
- **Death loop** — should be < 20%, ideally < 10%
- **Abrupt ending** — should improve significantly vs pretrained baseline
- **Missing EOS** — should be 0%
- **Empty response** — should be 0%

## 11. Write the Model README

Create `models/<model-name>/README.md` with:
- Model details (size, architecture, parallelism)
- Config table with optimizer, lr, notes
- IFEval results (pretrained baseline vs SFT, thinking off/on)
- Inference quality table (all 4 failure modes, pretrained vs SFT)
- Training loss table (step 0, step 999, val loss)
- Training curves (from W&B)
- Key takeaways

See `models/qwen3-4b/README.md` for the format.


## Checklist

- [ ] Baselines established (pretrained model eval + inference quality)
- [ ] Model directory created with chat template and configs
- [ ] Truncation rates checked, data pre-filtered and cached
- [ ] Data validation passes (5/5 assertions)
- [ ] Model verification passes (cosine sim > 0.99 vs HF)
- [ ] Training converges (loss decreasing, no NaN)
- [ ] SFT eval results (thinking off + on)
- [ ] Inference quality analysis (all 4 failure modes vs baseline)
- [ ] Model README with results and takeaways
