# Troubleshooting

:::{tip}
**TL;DR** -- Common errors and one-line fixes. Check the table below before opening an issue.
:::

## Error Reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| **CUDA out of memory** | Model too large for GPU | Reduce `local_batch_size`, use LoRA/QLoRA, or add `model.is_meta_device: true` for FSDP2 sharded init |
| **403 Forbidden from Hugging Face** | Gated model, no auth token | Run `huggingface-cli login` or switch to an ungated model like `Qwen/Qwen3-0.6B` |
| **NCCL timeout** | Network/firewall issue on multi-node | Increase `timeout_minutes`, verify NCCL env vars (`NCCL_DEBUG=INFO`), check firewall rules |
| **`uv` not found** | `uv` not installed | `pip install uv` or use `pip` directly instead of `uv run` |
| **Loss not decreasing** | Learning rate too low, data issue, or loss mask misconfigured | Increase LR by 5-10x, verify `answer_only_loss_mask`, check data format |
| **ImportError / ModuleNotFoundError** | Wrong nemo-automodel version | `pip install --upgrade nemo-automodel` |
| **Checkpoint won't load** | Mesh mismatch (different TP/FSDP config) | Use consolidated checkpoint (`save_consolidated: true`) or reshard to match new mesh |
| **`_target_` not found** | Typo in YAML `_target_` path | Verify the Python import path is correct: `python -c "from <path> import <class>"` |
| **Tokenizer warnings / bad output** | Chat template mismatch between training and inference | Verify with `print(tokenizer.chat_template)`, see [Chat Templates](chat-templates.md) |
| **Gradient NaN / Inf** | Learning rate too high or data contains bad values | Reduce LR, enable gradient clipping (`max_grad_norm: 1.0`), check for NaN in training data |

## Quick Debugging Checklist

1. **Check GPU memory**: `nvidia-smi` -- is memory near capacity?
2. **Check CUDA version**: `nvcc --version` -- matches your PyTorch build?
3. **Check install**: `python -c "import nemo_automodel; print(nemo_automodel.__version__)"` -- recent version?
4. **Check data**: Load your dataset in Python and inspect the first few examples
5. **Check config**: Print the resolved config: `automodel finetune llm -c config.yaml --print-config`

## Getting More Help

:::{note}
**Still stuck?** Open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues) with:
- Full error traceback
- Your YAML config (redact sensitive paths)
- Output of `pip list | grep nemo`
- GPU model and CUDA version (`nvidia-smi`)
:::
