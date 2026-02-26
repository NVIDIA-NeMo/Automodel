# Troubleshooting Unsupported Models

Sometimes a model listed on the Hugging Face Hub may not work with NeMo AutoModel.
If you encounter any such model, please open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues) with the model ID and any stack trace you see.

## Common Issues

| Issue | Example Error | Solution |
|-------|---------------|----------|
| Model has explicitly disabled training in its definition code | — | Request support via a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues). We can add the model through our custom registry. |
| Model requires a newer `transformers` version | `The checkpoint you are trying to load has model type deepseek_v32 but Transformers does not recognize this architecture.` | Upgrade `transformers` (and NeMo AutoModel if needed), or open a GitHub issue. |
| Model upper-bounds `transformers`, requiring an older version | — | Open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues). |
| Unsupported checkpoint format | `OSError: meta-llama/Llama-2-70b does not appear to have a file named pytorch_model.bin, model.safetensors, ...` | Open a GitHub issue or request the model publisher to share a SafeTensors checkpoint. |

These cases typically stem from upstream packaging or dependency constraints. You would encounter the same issues when using `transformers` directly, as AutoModel mirrors the familiar load and fine-tune semantics.

## Steps to Try

1. **Upgrade NeMo AutoModel** to a release that supports the required `transformers` version. See [Installation](../guides/installation.md).
2. **Enable remote code** — if the model uses custom code, set `trust_remote_code: true` in your `model:` config. See [Hugging Face API Compatibility](../guides/huggingface-api-compatibility.md).
3. **Open a GitHub issue** with the model ID and error so we can prioritize support or add a registry-backed implementation.
