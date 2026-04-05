# Command-R

[Cohere Command-R](https://cohere.com/command) is a series of enterprise-grade language models optimized for retrieval-augmented generation (RAG) and tool use. Command-R7B uses the updated `Cohere2ForCausalLM` architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `CohereForCausalLM` / `Cohere2ForCausalLM` |
| **Parameters** | 7B – 104B |
| **HF Org** | [CohereForAI](https://huggingface.co/CohereForAI) |
:::

## Available Models

- **c4ai-command-r-v01**: 35B
- **c4ai-command-r-plus**: 104B
- **c4ai-command-r7b-12-2024**: 7B (`Cohere2ForCausalLM`)

## Architectures

- `CohereForCausalLM` — Command-R v01, Plus
- `Cohere2ForCausalLM` — Command-R7B

## Example HF Models

| Model | HF ID |
|---|---|
| Command-R v01 | `CohereForAI/c4ai-command-r-v01` |
| Command-R7B | `CohereForAI/c4ai-command-r7b-12-2024` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`cohere_command_r_7b_squad.yaml <../../../examples/llm_finetune/cohere/cohere_command_r_7b_squad.yaml>` | SFT — Command-R 7B on SQuAD |
| {download}`cohere_command_r_7b_squad_peft.yaml <../../../examples/llm_finetune/cohere/cohere_command_r_7b_squad_peft.yaml>` | LoRA — Command-R 7B on SQuAD |


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

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/cohere/cohere_command_r_7b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/cohere/cohere_command_r_7b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [CohereForAI/c4ai-command-r7b-12-2024](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)
