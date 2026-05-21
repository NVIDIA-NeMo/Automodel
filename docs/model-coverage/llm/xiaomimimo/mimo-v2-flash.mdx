# MiMo-V2-Flash

[MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) is Xiaomi's
hybrid attention Mixture-of-Experts language model. It alternates full and
sliding-window attention layers, uses a `sigmoid_with_bias` router with
group-limited expert routing, and ships as an FP8 HF checkpoint.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE, hybrid attention) |
| **Architecture** | `MiMoV2FlashForCausalLM` |
| **Parameters** | Approx. several hundred B total / much smaller active |
| **HF Org** | [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) |
:::

## Available Models

- **MiMo-V2-Flash**: hybrid full/sliding-window attention with FP8 weights.

## Architecture

- `MiMoV2FlashForCausalLM`
- Sliding-window attention via the `MiMoV2FlashAttention(is_swa=True)` path.
- MoE blocks use the shared `nemo_automodel.components.moe.layers.MoE`
  with `score_func="sigmoid_with_bias"` and `gate_precision=fp32` so
  routing decisions stay numerically stable when activations are bf16.
- FP8 round-trip in `MiMoV2FlashStateDictAdapter` covers the bulk of
  attention/expert weights; layer norms, the gate, `lm_head`, and
  `embed_tokens` stay in bf16 per `NON_QUANTIZED_KEY_PATTERNS`.

## Example HF Models

| Model | HF ID |
|---|---|
| MiMo-V2-Flash | [`XiaomiMiMo/MiMo-V2-Flash`](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`mimo_v2_flash_hellaswag.yaml <../../../../examples/llm_finetune/mimo_v2_flash/mimo_v2_flash_hellaswag.yaml>` | SFT — MiMo-V2-Flash on HellaSwag |


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
automodel --nproc-per-node=8 examples/llm_finetune/mimo_v2_flash/mimo_v2_flash_hellaswag.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2. Navigate to the AutoModel directory**:

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/mimo_v2_flash/mimo_v2_flash_hellaswag.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
