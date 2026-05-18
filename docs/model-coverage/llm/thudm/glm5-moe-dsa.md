# GLM-5 / GLM-5.1 (MoE + DSA)

[GLM-5](https://huggingface.co/zai-org/GLM-5) and [GLM-5.1](https://huggingface.co/zai-org/GLM-5.1) are Zhipu AI's latest open-source large Mixture-of-Experts models featuring a DeepSeek-style MLA (Multi-head Latent Attention) + DSA (Dynamic Sparse Attention) architecture. GLM-5.1 shares the `glm_moe_dsa` architecture with GLM-5, with updated weights.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `GlmMoeDsaForCausalLM` |
| **Parameters** | 256 routed experts, 8 active per token |
| **HF Org** | [zai-org](https://huggingface.co/zai-org) |
:::

## Key Features

- **Mixture of Experts (MoE)**: 256 routed experts with 8 active per token
- **78 layers**, hidden size 6144, with MLA using KV compression (kv_lora_rank=512) and head_dim=64
- **~200k context window** (max_position_embeddings=202,752)
- **3 dense layers** followed by MoE layers (first_k_dense_replace=3)

## Available Models

- **GLM-5** (`GlmMoeDsaForCausalLM`)
- **GLM-5.1** (`GlmMoeDsaForCausalLM`): updated weights

## Example HF Models

| Model | HF ID |
|---|---|
| GLM-5 | [`zai-org/GLM-5`](https://huggingface.co/zai-org/GLM-5) |
| GLM-5.1 | [`zai-org/GLM-5.1`](https://huggingface.co/zai-org/GLM-5.1) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`glm_5_hellaswag_pp.yaml <../../../../examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml>` | SFT — GLM-5 with EP=64, PP=4 on 32 nodes |
| {download}`glm_5.1_hellaswag_pp.yaml <../../../../examples/llm_finetune/glm/glm_5.1_hellaswag_pp.yaml>` | SFT — GLM-5.1 with EP=64, PP=4 on 32 nodes |

## Parallel Setup

The recipe scales training using Expert Parallelism and Pipeline Parallelism (EP=64, PP=4 across 32 nodes of 8× H100 GPUs).

```yaml
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 4
  ep_size: 64
  sequence_parallel: false
  activation_checkpointing: true
  pipeline:
    pp_schedule: interleaved1f1b
    pp_microbatch_size: 1
    round_virtual_stages_to_pp_multiple: down
    scale_grads_in_schedule: false
    patch_inner_model: false
    patch_causal_lm_model: false
    layers_per_stage: 2
  moe:
    reshard_after_forward: false
    wrap_outer_model: false
```

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

:::{note}
This recipe was validated on **32 nodes × 8 GPUs (256 H100s)**. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
- [zai-org/GLM-5.1](https://huggingface.co/zai-org/GLM-5.1)
