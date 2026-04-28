# NeMo AutoModel Tutorials

End-to-end tutorials covering the LLM customization lifecycle using
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel).

| Tutorial | Dataset | Description | Launch on Brev |
|----------|---------|-------------|----------------|
| [Supervised Fine-Tuning (SFT)](./sft-peft) | [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | Full-parameter SFT to adapt a pre-trained model to follow instructions. | 🚧 |
| [Parameter-Efficient Fine-Tuning (PEFT)](./sft-peft) | [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | Memory-efficient LoRA fine-tuning for task adaptation. | 🚧 |
| [Evaluation](./evaluation) | Standard benchmarks (MMLU, HellaSwag, IFEval, etc.) | Evaluate AutoModel checkpoints with lm-evaluation-harness. | 🚧 |
| [Nemotron Parse Fine-Tuning](./nemotron-parse) | [Invoices](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) | Fine-tune Nemotron Parse v1.1 for structured document extraction. | [![Launch on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3C6LDKU2DfOvpVTFhjw3YQ4djPM) |

## Prerequisites

- [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) installed
  (see the AutoModel README for setup instructions).
- NVIDIA GPU(s) with sufficient memory (specific requirements noted per tutorial).
- Hugging Face account and API token for gated models (e.g., Llama).

## Pipeline Overview

These tutorials cover four stages of the LLM customization lifecycle:

```
Foundation Model ──> DAPT ──> SFT / PEFT ──> Evaluation
       |
       └──────────> Reasoning SFT ──────────> Evaluation
```

- **DAPT**: Inject domain knowledge via continued pre-training.
- **SFT / PEFT**: Teach the model to follow instructions or solve specific tasks.
- **Reasoning SFT**: Teach the model chain-of-thought reasoning with on/off control.
- **Evaluation**: Measure quality on standard benchmarks after each stage.

For reinforcement learning from human feedback (RLHF / DPO / PPO), see
[NeMo-RL](https://github.com/NVIDIA/NeMo-RL).
