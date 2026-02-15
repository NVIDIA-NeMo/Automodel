# Quickstart: Fine-Tune a Model in 5 Minutes

:::{tip}
**TL;DR** -- Install with `pip install nemo-automodel`, run one command to fine-tune [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (ungated, fits on 8 GB VRAM), then load your checkpoint with standard Hugging Face APIs.
:::

## Prerequisites

- **GPU**: NVIDIA, 8 GB+ VRAM
- **CUDA**: 11.8+
- **Python**: 3.10+
- **Internet**: to download model + dataset from Hugging Face

:::{note}
Not sure which model fits your GPU? See [Choose a Model and Method](choosing-model-and-method.md).
:::

## 1 -- Install

```bash
pip install nemo-automodel
python -c "import nemo_automodel; print('AutoModel ready')"
```

:::{tip}
Other methods (Docker, editable, `uv`) are in the [Installation Guide](installation.md).
:::

## 2 -- Fine-Tune

```bash
automodel finetune llm \
  -c examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml \
  --checkpoint.enabled true \
  --checkpoint.model_save_format safetensors \
  --checkpoint.save_consolidated true \
  --step_scheduler.max_steps 20 \
  --step_scheduler.ckpt_every_steps 20 \
  --step_scheduler.val_every_steps 20
```

:::{tip}
**Multiple GPUs?** Add `--nproc-per-node=4` (or however many you have).
:::

**Expected output:**

```text
INFO:root:step 4  | epoch 0 | loss 3.55 | mem: 5.66 GiB | tps 6924
INFO:root:step 8  | epoch 0 | loss 2.79 | mem: 6.58 GiB | tps 9328
INFO:root:step 12 | epoch 0 | loss 2.43 | mem: 6.48 GiB | tps 9068
INFO:root:step 16 | epoch 0 | loss 2.20 | mem: 6.47 GiB | tps 9148
INFO:root:step 20 | epoch 0 | loss 2.25 | mem: 5.35 GiB | tps 9196
Saving checkpoint to checkpoints/epoch_0_step_20
```

- **loss** should decrease over time
- Checkpoint saved to `checkpoints/epoch_0_step_20/model/consolidated/`

## 3 -- Run Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "checkpoints/epoch_0_step_20/model/consolidated/"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

:::{tip}
Checkpoints are standard Hugging Face format -- they work with [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ollama](https://ollama.com), and more. See [Deployment](deployment.md).
:::

## 4 -- Use Your Own Data

1. **Format** as JSONL: `{"instruction": "...", "output": "..."}`
2. **Update YAML** `dataset` section:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /path/to/your/data.jsonl
  column_mapping:
    question: instruction
    answer: output
  answer_only_loss_mask: true
```

3. **Run** the same command from step 2.

:::{note}
More dataset types (conversations, VLM, streaming, pretraining) are in the [Dataset Overview](dataset-overview.md).
:::

## YAML Config at a Glance

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained  # _target_ = which Python class to call
  pretrained_model_name_or_path: Qwen/Qwen3-0.6B    # any HF model ID

dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train

step_scheduler:
  global_batch_size: 64       # total samples per gradient step
  local_batch_size: 8         # samples per GPU per micro-batch
  num_epochs: 1

distributed:
  strategy: fsdp2             # FSDP2, megatron_fsdp, or ddp
  tp_size: 1                  # tensor parallel GPUs

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5                  # see Hyperparameters Guide for recommendations
```

**Override anything from CLI:** `--model.pretrained_model_name_or_path Qwen/Qwen3-8B --optimizer.lr 5e-6`

:::{note}
Full config reference: [YAML Configuration](configuration.md).
:::

## What's Next?

| Goal | Guide |
|------|-------|
| Use LoRA for large models | [Fine-Tuning Guide (PEFT)](llm/finetune.md) |
| Fine-tune a vision-language model | [Gemma 3n VLM](omni/gemma3-3n.md) |
| Choose the right model and method | [Choose a Model and Method](choosing-model-and-method.md) |
| Tune learning rate, epochs, LoRA rank | [Hyperparameters Guide](hyperparameters.md) |
| Deploy to vLLM / Ollama / HF Hub | [Deployment](deployment.md) |
| Run on a Slurm cluster | [Run on a Cluster](../launcher/cluster.md) |
| Train your own architecture | [Custom Model Guide](custom-model.md) |
| Not sure if you should fine-tune? | [Is Training Right for Me?](is-training-right-for-me.md) |
