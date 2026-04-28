# Hy3 (HunyuanLarge)

[Hy3-preview](https://huggingface.co/tencent/Hy3-preview) is a 295B Mixture-of-Experts language model from Tencent. It features 80 transformer layers (layer 0 dense, layers 1–79 MoE), 192 routed experts plus 1 shared expert with top-8 sigmoid routing, Grouped Query Attention (64 Q / 8 KV heads), per-head QK RMSNorm, RoPE, and an `e_score_correction_bias` gate buffer for expert-load correction. It supports a 256K context window.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `HYV3ForCausalLM` |
| **Parameters** | 295B total |
| **HF Org** | [tencent](https://huggingface.co/tencent) |
:::

## Available Models

- **Hy3-preview**: 295B total, top-8 routed experts activated per token

## Architectures

- `HYV3ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Hy3-preview | [`tencent/Hy3-preview`](https://huggingface.co/tencent/Hy3-preview) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`hy3_4layer_p0_smoke.yaml <../../../../examples/llm_finetune/hy_v3/hy3_4layer_p0_smoke.yaml>` | SFT — Hy3 4-layer smoke test (P0) |
| {download}`hy3_4layer_p1_ckpt.yaml <../../../../examples/llm_finetune/hy_v3/hy3_4layer_p1_ckpt.yaml>` | SFT — Hy3 4-layer checkpoint test (P1) |
| {download}`hy3_preview_deepep.yaml <../../../../examples/llm_finetune/hy_v3/hy3_preview_deepep.yaml>` | SFT — Hy3-preview with DeepEP |

## Try with NeMo AutoModel

**1. Install** ([NeMo AutoModel](../../../guides/installation.md)):

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
automodel --nproc-per-node=8 examples/llm_finetune/hy_v3/hy3_preview_deepep.yaml
```

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [tencent/Hy3-preview](https://huggingface.co/tencent/Hy3-preview)
