# Qwen3-VL / Qwen3-VL-MoE

[Qwen3-VL](https://qwenlm.github.io/blog/qwen3/) is Alibaba Cloud's third-generation vision language model series. The MoE variant activates a fraction of parameters per token for efficient large-scale inference.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen3VLForConditionalGeneration` |
| **Parameters** | 4B – 235B |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3-VL-8B-Instruct**: 8B
- **Qwen3-VL-4B-Instruct**: 4B
- **Qwen3-VL-MoE-30B**: 30B total (MoE)
- **Qwen3-VL-MoE-235B**: 235B total (MoE)

## Architecture

- `Qwen3VLForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3-VL 4B Instruct | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen3-VL 8B Instruct | `Qwen/Qwen3-VL-8B-Instruct` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`qwen3_vl_4b_instruct_rdr.yaml <../../../examples/vlm_finetune/qwen3/qwen3_vl_4b_instruct_rdr.yaml>` | rdr-items | SFT — Qwen3-VL 4B on RDR Items |
| {download}`qwen3_vl_8b_instruct_rdr.yaml <../../../examples/vlm_finetune/qwen3/qwen3_vl_8b_instruct_rdr.yaml>` | rdr-items | SFT — Qwen3-VL 8B on RDR Items |
| {download}`qwen3_vl_moe_30b_te_deepep.yaml <../../../examples/vlm_finetune/qwen3/qwen3_vl_moe_30b_te_deepep.yaml>` | MedPix-VQA | SFT — Qwen3-VL-MoE 30B with TE + DeepEP |
| {download}`qwen3_vl_moe_235b.yaml <../../../examples/vlm_finetune/qwen3/qwen3_vl_moe_235b.yaml>` | MedPix-VQA | SFT — Qwen3-VL-MoE 235B |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3/qwen3_vl_4b_instruct_rdr.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3/qwen3_vl_4b_instruct_rdr.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
