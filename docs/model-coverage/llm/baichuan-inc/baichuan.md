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
| Baichuan2 13B Chat | [`baichuan-inc/Baichuan2-13B-Chat`](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) |
| Baichuan 7B | [`baichuan-inc/Baichuan-7B`](https://huggingface.co/baichuan-inc/Baichuan-7B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`baichuan_2_7b_squad.yaml <../../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml>` | SFT — Baichuan2 7B on SQuAD |
| {download}`baichuan_2_7b_squad_peft.yaml <../../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml>` | LoRA — Baichuan2 7B on SQuAD |


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
automodel --nproc-per-node=8 examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
