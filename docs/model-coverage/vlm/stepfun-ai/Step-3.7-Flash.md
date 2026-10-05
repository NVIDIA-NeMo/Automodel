---
title: "stepfun-ai/Step-3.7-Flash"
description: "Fine-tune stepfun-ai/Step-3.7-Flash on MedPix with full or LoRA vision-language recipes in NeMo AutoModel for supervised image-text training."
slug: model-coverage/vision-language-models/stepfun-ai/Step-3.7-Flash
---

[Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) is StepFun AI's mixture-of-experts (MoE) vision-language model, described upstream as 198B parameters with approximately 11B active per token. The upstream model card describes native image understanding. Use the NeMo AutoModel recipes below for full or LoRA image-text fine-tuning on MedPix-VQA.

## Quick Start

Complete the [recipe setup](#recipe-setup) for full fine-tuning on MedPix-VQA. This example uses 16 nodes with 8 GPUs each. From the repository root on every node, run with `NODE_RANK` set to 0-15 and `MASTER_ADDR` set to the reachable address of node 0.

```bash
uv run torchrun --nnodes 16 --nproc-per-node 8 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29500 \
  -m nemo_automodel.cli.app examples/vlm_finetune/stepfun/step3p7_medpix_200b_ep32pp4.yaml
```

## Choose a Workflow

| Workflow | Example Setup | Recipe |
| --- | --- | --- |
| Full Image-Text Fine-Tuning | MedPix-VQA; 16 nodes; PP4; EP32 | [View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_ep32pp4.yaml) |
| LoRA Image-Text Fine-Tuning | MedPix-VQA; 8 nodes; PP8; EP8 | [View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_lora_pp8ep8_8node.yaml) |

## Model Context

The native processor accepts images and text. Both recipes load pretrained weights and fine-tune on MedPix-VQA image-text pairs while freezing the vision tower.

| Property | Value |
|---|---|
| **Task** | Image-Text-to-Text (upstream); MedPix-VQA image-text fine-tuning (recipes) |
| **Architecture** | `Step3p7ForConditionalGeneration` — MoE vision-language model |
| **Language Module** | Step-3.5-Flash-derived backbone, 45 layers, 288 experts, top-8 routing |
| **Vision Module** | 1.8B ViT, 47 layers, 728x728 image size |
| **Checkpoint Context Window** | 256K tokens (262,144); recipe inputs are capped at 2,048 tokens |
| **Recipe Precision** | BF16 (`torch_dtype: bfloat16` in both recipes) |
| **HF Org** | [stepfun-ai](https://huggingface.co/stepfun-ai) |
| Parameters | 201.37B in the pinned checkpoint metadata; upstream reports 198B total and approximately 11B active per token |
| Decoder Layers | 45 |
| Hidden Size | 4,096 |
| Attention | 64 query heads in full-attention layers; 96 in sliding-attention layers; 8 key-value heads in both |
| Context Length | 262,144 tokens (checkpoint configuration); recipe inputs capped at 2,048 tokens |
| Vocabulary Size | 128,896 |

### Positioning

The [upstream model card](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/5f6244077ac62e04eec3f320501ff8c2b293373a/README.md) describes native image understanding. The NeMo AutoModel workflows on this page cover supervised image-text fine-tuning on MedPix-VQA with full fine-tuning or LoRA.

### Architecture

- **Language backbone:** derived from Step-3.5-Flash with 45 layers, 288 experts, 8 activated experts per token, and a 256K checkpoint context length. Both recipes cap inputs at 2,048 tokens.
- **Vision backbone:** 1.8B-parameter ViT with 47 layers and 728x728 image inputs.

### Key Strengths

- **Native image understanding.** Described by the upstream model card; the NeMo AutoModel processor accepts images and text.
- **MedPix-VQA fine-tuning.** Both recipes train on image-text pairs with the vision tower frozen.
- **Full and LoRA recipes.** Choose either workflow using the distributed configurations listed above.

### Example Recipes

- [Full SFT — MedPix, EP32 + PP4](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_ep32pp4.yaml)
- [LoRA — MedPix, PP8 + EP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_lora_pp8ep8_8node.yaml)

See the [Step-3.7-Flash fine-tuning guide](../../../guides/vlm/step-3-7.md) for the expected training setup and launch notes.

### Agent Frameworks

The upstream model card lists OpenClaw, Hermes Agent, and Kilo Code as agent platforms for Step-3.7-Flash.

### Setup and Additional Examples

Review the recipes in [Choose a Workflow](#choose-a-workflow), including their
hardware and data requirements. Follow the [installation instructions](/get-started/installation)
before launching a recipe.

### Recipe Setup

Read [step3p7_medpix_200b_ep32pp4.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_ep32pp4.yaml) for checkpoint access, data paths, and backend dependencies. Use the [Slurm launcher guide](/job-launchers/slurm-cluster) to prepare the distributed allocation.

Both recipes select Transformer Engine for attention and linear layers. The full recipe uses HybridEP; the LoRA recipe uses DeepEP. Source installations need the `moe` extra from the [installation instructions](/get-started/installation); the standard `all` extra omits DeepEP. The pinned `deep_ep` package supplies both dispatchers. For the full recipe, build HybridEP with `HYBRID_EP_MULTINODE=1` and the matching CUDA and RDMA dependencies, as configured in the [repository Dockerfile](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docker/Dockerfile).

Make the checkpoint, processor, and MedPix-VQA data accessible from every node. The example topologies use pipeline parallelism (PP) and expert parallelism (EP); they are recipe configurations, not minimum hardware requirements.

### Workflow Notes

Fine-tune stepfun-ai/Step-3.7-Flash: [step3p7_medpix_200b_ep32pp4.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/stepfun/step3p7_medpix_200b_ep32pp4.yaml). 16 nodes, 8 GPUs per node; see [Recipe Setup](#recipe-setup).

The full recipe saves checkpoints under `vlm_checkpoints/step3p7_medpix_ep32pp4/`. The LoRA recipe sets `checkpoint.enabled: false`, so it does not save trained adapters. Enable checkpoint saving and configure its interval and output directory before running LoRA when you need to retain the result.

## Available Models

- **Step-3.7-Flash** — registered as `Step3p7ForConditionalGeneration`, with the checkpoint-facing alias `Step3p6ForConditionalGeneration` mapping to the same model class.

| Model | HF ID |
|---|---|
| Step-3.7-Flash | [`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash) |

## Related Resources

- [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash)

- [Checkpoint Configuration](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/5f6244077ac62e04eec3f320501ff8c2b293373a/config.json)
- [Upstream Model Card](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/5f6244077ac62e04eec3f320501ff8c2b293373a/README.md)
