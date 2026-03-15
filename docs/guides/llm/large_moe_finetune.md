# Fine-Tune Large MoE LLMs

## Introduction

Mixture-of-Experts (MoE) architectures activate only a fraction of their total parameters per token, delivering strong performance at reduced compute cost. Fine-tuning MoE models follows the same recipe as dense models (see the [SFT/PEFT guide](finetune.md)), but introduces additional parallelism dimensions:

- **Expert Parallelism (`ep_size`)**: Distributes individual experts across GPUs. Required because MoE models have far more total parameters than a single GPU can hold, even though only a subset is active per token.
- **Pipeline Parallelism (`pp_size`)**: Splits model layers across pipeline stages to further reduce per-GPU memory. Larger models need more stages.
- **Activation Checkpointing**: Trades recomputation for memory — enabled for the largest models (GLM-5, DeepSeek-V3.2) to fit within GPU memory.

This guide covers four validated MoE models. For the full list of supported architectures, see the [LLM model coverage](../../model-coverage/llm.md) page.

## Models

| Model | HF Checkpoint | GPUs | `pp_size` | `ep_size` | Activation Checkpointing |
|-------|--------------|------|-----------|-----------|--------------------------|
| MiniMax-M2.5 | [`MiniMaxAI/MiniMax-M2.5`](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) | 64 H100 (8 nodes x 8) | 2 | 32 | No |
| Step-3.5 Flash | [`stepfun-ai/Step-3.5-Flash`](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | 64 H100 (8 nodes x 8) | 2 | 32 | No |
| GLM-5 | [`zai-org/GLM-5`](https://huggingface.co/zai-org/GLM-5) | 256 H100 (32 nodes x 8) | 4 | 64 | Yes |
| DeepSeek-V3.2 | [`deepseek-ai/DeepSeek-V3.2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | 256 H100 (32 nodes x 8) | 4 | 64 | Yes |

All four use FSDP2 with interleaved 1F1B pipeline schedule and 2 layers per stage. A representative `distributed:` config (GLM-5) is shown below — the other models differ only in the values from the table above:

```yaml
distributed:
  strategy: fsdp2
  pp_size: 4              # pipeline stages (2 for 64-GPU models, 4 for 256-GPU models)
  ep_size: 64             # expert parallelism degree (32 for 64-GPU models, 64 for 256-GPU models)
  activation_checkpointing: true  # required for larger models; omit or set false for smaller ones
  pipeline:
    pp_schedule: interleaved1f1b
    layers_per_stage: 2
```

Full recipe YAMLs:

- MiniMax-M2.5: [`examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml)
- GLM-5: [`examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml)
- Step-3.5 Flash: [`examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml)
- DeepSeek-V3.2: [`examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml)

To set up your environment, follow the [installation guide](../installation.md).

## Data

### HellaSwag Dataset

All four recipes use the [HellaSwag](https://huggingface.co/datasets/rowan/hellaswag) dataset, a commonsense natural language inference benchmark where the model must predict the most plausible continuation of a given scenario.

- **Source**: `rowan/hellaswag`
- **Split**: `train` (used for both training and validation in these recipes)
- **Task**: Next-token prediction on commonsense sentence completions

To swap in your own dataset, see [Integrate Your Own Text Dataset](dataset.md) and the [Dataset Overview](../dataset-overview.md).

## Launch Training

NeMo Automodel supports several launch methods — the Automodel CLI, Slurm, interactive sessions, and `torchrun`. For full details on multi-node configuration, Slurm batch jobs, and environment variables, see the [Run on a Cluster](../../launcher/cluster.md) guide.

### Automodel CLI

```bash
automodel finetune llm -c examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml
```

Replace the recipe path with the one for your target model.

### torchrun

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HOME=your/path/to/hf_cache
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=your_wandb_key

torchrun --nproc_per_node=8 \
         --nnodes=8 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
  nemo_automodel/recipes/llm/train_ft.py \
    -c examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml \
    --model.pretrained_model_name_or_path=/your/local/model_weights
```

Replace the `-c` path, `--nnodes`, and `--model.pretrained_model_name_or_path` for your target model.

**Before you start**:
- Hugging Face applies rate limits on downloads. Clone the model repository to your local filesystem beforehand.
- Ensure your Hugging Face cache (`HF_HOME`) is configured and that the dataset is already cached locally.
- To enable Weights & Biases logging, set your `WANDB_API_KEY` and configure the `wandb` section in the YAML file.
