# Baichuan / Baichuan2

[Baichuan](https://github.com/baichuan-inc/Baichuan2) is a Chinese-English bilingual language model series from Baichuan Inc., featuring strong Chinese language performance.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `BaiChuanForCausalLM` |
| **Parameters** | 7B – 13B |
| **HF Org** | [baichuan-inc](https://huggingface.co/baichuan-inc) |
:::

## Available Models

- **Baichuan2-13B-Chat**
- **Baichuan2-7B-Chat**
- **Baichuan-7B**

## Architecture

- `BaiChuanForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Baichuan2 13B Chat | `baichuan-inc/Baichuan2-13B-Chat` |
| Baichuan 7B | `baichuan-inc/Baichuan-7B` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`baichuan_2_7b_squad.yaml <../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml>` | SFT — Baichuan2 7B on SQuAD |
| {download}`baichuan_2_7b_squad_peft.yaml <../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml>` | LoRA — Baichuan2 7B on SQuAD |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
