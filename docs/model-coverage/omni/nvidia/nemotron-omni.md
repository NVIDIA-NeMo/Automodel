# Nemotron-3-Nano-Omni

[Nemotron-3-Nano-Omni-30B-A3B-Reasoning](https://huggingface.co/nvidia) is NVIDIA's
omnimodal reasoning model. It pairs a NemotronH (hybrid Mamba-2 + Attention) MoE
language backbone with a RADIO v2.5-H vision encoder and a Parakeet (FastConformer)
sound encoder, supporting interleaved text, image, and audio inputs.

:::{card}
| | |
|---|---|
| **Task** | Omnimodal (Text·Image·Audio) |
| **Architecture** | `NemotronH_Nano_Omni_Reasoning_V3` |
| **Parameters** | 30B total / 3B active |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Nemotron-3-Nano-Omni-30B-A3B-Reasoning**: 30B total, 3B activated (MoE)

## Architecture

- `NemotronH_Nano_Omni_Reasoning_V3`

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`nemotron_omni_cord_v2.yaml <../../../../examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml>` | CORD-v2 | Full SFT — receipt parsing |
| {download}`nemotron_omni_cord_v2_peft.yaml <../../../../examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2_peft.yaml>` | CORD-v2 | LoRA PEFT — receipt parsing |

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

**3. Run the recipe** from inside the repo (8x H100 example):

```bash
automodel examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml --nproc-per-node 8
```

For a full walkthrough — dataset preparation, SFT vs. LoRA configs, and
post-training inference — see the
[Nemotron-Omni guide](../../../guides/vlm/nemotron-omni.md).
