# StarCoder2

[StarCoder2](https://huggingface.co/blog/starcoder2) is BigCode's second-generation code language model, available in 3B, 7B, and 15B sizes, trained on 600+ programming languages from The Stack v2.

:::{card}
| | |
|---|---|
| **Task** | Code Generation |
| **Architecture** | `Starcoder2ForCausalLM` |
| **Parameters** | 3B – 15B |
| **HF Org** | [bigcode](https://huggingface.co/bigcode) |
:::

## Available Models

- **starcoder2-3b**: 3B
- **starcoder2-7b**: 7B
- **starcoder2-15b**: 15B

## Architecture

- `Starcoder2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| StarCoder2 3B | `bigcode/starcoder2-3b` |
| StarCoder2 7B | `bigcode/starcoder2-7b` |
| StarCoder2 15B | `bigcode/starcoder2-15b` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`starcoder_2_7b_squad.yaml <../../../examples/llm_finetune/starcoder/starcoder_2_7b_squad.yaml>` | SFT — StarCoder2 7B on SQuAD |
| {download}`starcoder_2_7b_hellaswag_fp8.yaml <../../../examples/llm_finetune/starcoder/starcoder_2_7b_hellaswag_fp8.yaml>` | SFT — StarCoder2 7B on HellaSwag with FP8 |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/starcoder/starcoder_2_7b_squad.yaml
```

:::{dropdown} Run with Docker
Pull the NeMo AutoModel container and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

Then inside the container:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/starcoder/starcoder_2_7b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b)
