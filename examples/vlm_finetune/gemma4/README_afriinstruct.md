# Gemma 4 E2B text-only SFT on AfriInstruct

This pipeline fine-tunes `google/gemma-4-E2B-it` without adding model, recipe,
or dataset classes. Gemma 4 E2B remains on the existing
`FinetuneRecipeForVLM` and `NeMoAutoModelForImageTextToText` paths; only the
dataset is materialized into AutoModel's existing conversation format.

A notebook is not required. Use a notebook only for optional exploratory
analysis. The data preparation command, generated manifests, recipe YAML, and
checkpoint metrics are the reproducible pipeline artifacts.

## 1. Install and authenticate

```bash
uv sync --locked --all-groups --extra vlm
export HF_TOKEN="hf_..."
export HF_HOME="/shared/hf-cache"
```

Accept the Gemma license on Hugging Face before loading
`google/gemma-4-E2B-it`.

## 2. Prepare AfriInstruct

The preparation utility streams the source dataset, removes malformed and
exact-duplicate instruction/output pairs, creates a deterministic validation
split, partitions records by language and task, and writes bounded JSONL
shards plus meta manifests.

Start with a bounded schema smoke test:

```bash
uv run python tools/prepare_afriinstruct.py \
  --output-dir /data/afriinstruct/e2b-schema-smoke-v1 \
  --max-samples 1000 \
  --validation-fraction 0.05 \
  --shard-size 250
```

The source stream is ordered by language, task, and source. Consequently,
`--max-samples` selects the first matching records and is suitable only for
checking schema and pipeline behavior. Do not train a model on that output.

For a representative training subset, scan the full stream and select records
with deterministic hash sampling:

```bash
uv run python tools/prepare_afriinstruct.py \
  --output-dir /data/afriinstruct/e2b-representative-v1 \
  --sample-fraction 0.01 \
  --validation-fraction 0.01 \
  --shard-size 50000
```

`--sample-fraction` preserves the source distribution without materializing the
full dataset in memory. The complete source must still be streamed once.

Select languages or tasks when the training objective is narrower:

```bash
uv run python tools/prepare_afriinstruct.py \
  --output-dir /data/afriinstruct/e2b-swa-hau-yor-v1 \
  --languages swa hau yor \
  --validation-fraction 0.02 \
  --shard-size 50000
```

The output directory contains:

```text
train_meta.json
validation_meta.json
summary.json
processed/train/*.jsonl
processed/validation/*.jsonl
```

Always inspect `summary.json` before training. It records source settings,
filters, rejection counts, and the retained language/task/source distribution.
Use a new versioned output directory for each preparation run; the utility
refuses to overwrite a nonempty directory.

The meta dataset loader materializes its selected records. Keep qualification
runs bounded to a size that fits host memory, and grow the prepared set only
after measuring setup memory and time on the target nodes.

## 3. Validate the recipes

```bash
uv run python tools/lint_example_yamls.py
uv run pytest tests/unit_tests/tools/test_prepare_afriinstruct.py -q
```

Point both recipes at the prepared dataset:

```bash
export AFRIINSTRUCT_DATA_DIR=/data/afriinstruct/e2b-representative-v1
```

Optional output locations:

```bash
export AFRIINSTRUCT_CHECKPOINT_DIR=/checkpoints/gemma4-e2b-afriinstruct
export AFRIINSTRUCT_WANDB_DIR=/logs/wandb
```

## 4. Run the LoRA qualification gate

The LoRA recipe is
`examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml`.
Run a 20-step gate before a full job. The warmup override keeps the scheduler
valid for the shortened run.

```bash
uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps 20 \
  --step_scheduler.ckpt_every_steps 10 \
  --step_scheduler.val_every_steps 10 \
  --lr_scheduler.lr_warmup_steps 2
```

The gate passes when training and validation losses are finite, gradients are
finite, a checkpoint is written, and the checkpoint can resume. Resume by
overriding `checkpoint.restore_from` with the checkpoint directory.

Run the qualified configuration on all GPUs in a node:

```bash
uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 8
```

Enable W&B only after credentials are configured:

```bash
uv run automodel \
  examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_peft.yaml \
  --nproc-per-node 8 \
  --wandb.enable true
```

## 5. Promote to full SFT only when justified

The full language-backbone recipe is
`examples/vlm_finetune/gemma4/gemma4_e2b_afriinstruct_full.yaml`. It keeps the
vision tower, audio tower, and embeddings frozen while updating the complete
language backbone. It uses a lower learning rate and writes a consolidated
SafeTensors export at the end.

Do not start full SFT until the LoRA pipeline has demonstrated correct data,
stable optimization, checkpoint resume, and measurable held-out gains.

## 6. Evaluation gates

Compare the pretrained model and each candidate checkpoint on fixed,
source-disjoint sets. Report results by language and task rather than only an
aggregate validation loss. Include task metrics, instruction following, and
generation-quality checks for repetition, empty output, missing EOS, and
abrupt endings.

The staged convergence and inference-quality process under
`examples/convergence/tulu3/` is the repository-native template for these
gates. Reuse its process, while keeping the Gemma 4 model and recipe settings
from the YAML files in this directory.
