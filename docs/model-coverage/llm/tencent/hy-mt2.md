# Hy-MT2 (Hunyuan-MT2)

[Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) is Tencent's translation Mixture-of-Experts language model with 30B total parameters and 3B activated per token. It features 48 transformer layers (layer 0 dense, layers 1–47 MoE), 128 routed experts plus 1 shared expert with top-8 sigmoid routing, Grouped Query Attention (32 Q / 4 KV heads), per-head QK RMSNorm, RoPE, and an in-forward fp32 upcast on the language-model head (`enable_lm_head_fp32`). It supports a 256K context window.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE, translation) |
| **Architecture** | `HyMT2ForCausalLM` |
| **Parameters** | 30B total / 3B activated |
| **HF Org** | [tencent](https://huggingface.co/tencent) |
:::

## Available Models

- **Hy-MT2-30B-A3B**: 30B total, top-8 routed experts (out of 128) activated per token, plus 1 shared expert

## Architectures

- `HyMT2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Hy-MT2-30B-A3B | [`tencent/Hy-MT2-30B-A3B`](https://huggingface.co/tencent/Hy-MT2-30B-A3B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`hy_mt2_30b_a3b_sft.yaml <../../../../examples/llm_finetune/hy_mt2/hy_mt2_30b_a3b_sft.yaml>` | SFT — Hy-MT2-30B-A3B with FSDP2 + EP8 + fp32 LM head |

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
automodel --nproc-per-node=8 examples/llm_finetune/hy_mt2/hy_mt2_30b_a3b_sft.yaml
```

Refer to the [NeMo AutoModel Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tune the Model

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B)
