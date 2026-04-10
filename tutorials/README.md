# NeMo AutoModel Tutorials

End-to-end tutorials covering the LLM customization lifecycle using
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).

| Tutorial | Dataset | Description |
|----------|---------|-------------|
| [Domain Adaptive Pre-Training (DAPT)](./dapt) | Domain-specific text corpus | Continued pre-training of a foundation model on domain data to improve in-domain performance (inspired by [ChipNeMo](https://arxiv.org/abs/2311.00176)). |
| [Supervised Fine-Tuning (SFT)](./sft-peft) | [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Full-parameter SFT to adapt a pre-trained model to follow instructions. |
| [Parameter-Efficient Fine-Tuning (PEFT)](./sft-peft) | [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | Memory-efficient LoRA fine-tuning for task adaptation. |
| [Evaluation](./evaluation) | Standard benchmarks (MMLU, HellaSwag, IFEval, etc.) | Evaluate AutoModel checkpoints with lm-evaluation-harness. |
| [Reasoning SFT](./reasoning-sft) | Reasoning instruction data (OpenAI chat format) | Fine-tune a model to selectively enable chain-of-thought reasoning via system prompt control. |
| [Nemotron Parse Fine-Tuning](./nemotron-parse) | [Invoices](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) | Fine-tune Nemotron Parse v1.1 for structured document extraction. |

## Prerequisites

- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) installed
  (see the AutoModel README for setup instructions).
- NVIDIA GPU(s) with sufficient memory (specific requirements noted per tutorial).
- Hugging Face account and API token for gated models (e.g., Llama).

## Pipeline Overview

These tutorials cover four stages of the LLM customization lifecycle:

```
DAPT ──> SFT / PEFT ──> Evaluation
               │
               └──> Reasoning SFT ──> Evaluation
```

- **DAPT**: Inject domain knowledge via continued pre-training.
- **SFT / PEFT**: Teach the model to follow instructions or solve specific tasks.
- **Reasoning SFT**: Teach the model chain-of-thought reasoning with on/off control.
- **Evaluation**: Measure quality on standard benchmarks after each stage.

For reinforcement learning from human feedback (RLHF / DPO / PPO), see
[NeMo-RL](https://github.com/NVIDIA/NeMo-RL).
