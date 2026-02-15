# Is Fine-Tuning Right for Me?

:::{tip}
**TL;DR** -- Try prompt engineering first. Fine-tune only when prompting isn't enough. This page helps you decide.
:::

## Decision Table

| Your Goal | Try First | If That Fails |
|-----------|-----------|---------------|
| **Follow instructions better** | Prompt engineering / few-shot examples | SFT or LoRA on instruction data |
| **Domain-specific knowledge** (medical, legal) | RAG (retrieval-augmented generation) | SFT on domain corpus |
| **Specific output format** (JSON, SQL, XML) | Few-shot prompting with format examples | SFT with format-specific training data |
| **Smaller / faster model** | Quantization (GPTQ, AWQ) | Knowledge distillation from a larger model |
| **New language or modality** | Multilingual base model + prompting | Pretraining or continued pretraining |

## Minimum Requirements for Fine-Tuning

Before you start, make sure you have:

- **GPU**: At least one NVIDIA GPU with 8+ GB VRAM (see [Choose a Model and Method](choosing-model-and-method.md) for sizing)
- **Data**: At least ~100 high-quality examples for LoRA, ~1,000+ for SFT (more is better)
- **Time**: A small fine-tune (1B model, 1,000 examples) takes minutes; a large one (70B, 100K examples) takes hours to days
- **Evaluation plan**: How will you measure if the fine-tune worked? (e.g., accuracy on held-out set, human eval)

:::{warning}
**Common pitfall**: Fine-tuning with too little data or too many epochs leads to overfitting -- the model memorizes your examples but doesn't generalize. Start with 1-3 epochs and monitor validation loss.
:::

## When Fine-Tuning Wins

- **Style / tone**: The model needs to write in your brand voice
- **Task-specific accuracy**: Measurably better than prompting on your eval set
- **Latency**: A smaller fine-tuned model can match a larger prompted model at lower cost
- **Privacy**: Fine-tuning lets you use a self-hosted model instead of an API

## When Fine-Tuning Isn't Worth It

- **The base model already does it well** with the right prompt
- **Your data is noisy or unlabeled** -- fine-tuning amplifies data quality, good and bad
- **You need the model to learn brand-new facts** -- fine-tuning is better for *style* than *knowledge* (consider RAG instead)

## Ready to Start?

| Next Step | Link |
|-----------|------|
| Pick a model and method | [Choose a Model and Method](choosing-model-and-method.md) |
| Run your first fine-tune | [Quickstart](quickstart.md) |
| Tune hyperparameters | [Hyperparameters Guide](hyperparameters.md) |
