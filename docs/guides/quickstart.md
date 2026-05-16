# Quickstart

This guide gets you from install to a local LoRA fine-tuning run with the least
configuration possible. You only need a Hugging Face model id and an
[OpenAI-format](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)
JSONL file of chat examples:

```bash
automodel <hf-model-id> <openai-chat-data.jsonl>
```

The command builds a default LoRA recipe config, detects the number of visible
GPUs on the node, launches with `torchrun` when more than one GPU is available,
and writes checkpoints under `./checkpoints/`.

## Install

For LLM fine-tuning:

```bash
pip install nemo-automodel
```

For VLM fine-tuning with image/video inputs, install the VLM extras:

```bash
pip install "nemo-automodel[vlm]"
```

If you are working from a source checkout, use your existing development
environment instead. For example:

```bash
uv sync --frozen
uv sync --frozen --extra vlm
```

See [Installation](installation.md) for Docker, source installs, and cluster
setup.

## Run an LLM LoRA Job

Use any Hugging Face causal language model and an OpenAI-format chat JSONL file:

```bash
automodel Qwen/Qwen2.5-1.5B-Instruct examples/quickstart/openai_chat.jsonl
```

Sample data:
[examples/quickstart/openai_chat.jsonl](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/quickstart/openai_chat.jsonl)

Each row contains a `messages` list using the same role/content structure as
OpenAI chat messages:

```json
{"messages":[{"role":"system","content":"You are a concise assistant."},{"role":"user","content":"Give one reason to use LoRA."},{"role":"assistant","content":"LoRA trains small adapters, so fine-tuning uses less memory."}]}
```

Multiturn rows use the same format. The full `messages` list is passed through
the tokenizer chat template.

## Run a VLM LoRA Job

Use the same command shape for OpenAI-style multimodal chat JSONL:

```bash
automodel Qwen/Qwen2.5-VL-3B-Instruct examples/quickstart/openai_vlm_chat.jsonl
```

Sample data:
[examples/quickstart/openai_vlm_chat.jsonl](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/quickstart/openai_vlm_chat.jsonl)

Rows with image or video content parts are routed to the VLM recipe:

```json
{"messages":[{"role":"user","content":[{"type":"text","text":"What color is the square?"},{"type":"image_url","image_url":{"url":"images/square.png"}}]},{"role":"assistant","content":"Red."}]}
```

The quick path accepts `image_url`, `input_image`, `image`, `video`, and
`input_video` content parts. Relative media paths are resolved relative to the
JSONL file. Inline `data:image/...` URLs also work for tiny self-contained
examples.

## GPU Selection

By default, `automodel` uses all visible GPUs on the local node. If more than
one GPU is visible, it starts a `torchrun` job with one worker per visible GPU.

Limit the run to specific GPUs with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 automodel Qwen/Qwen2.5-1.5B-Instruct examples/quickstart/openai_chat.jsonl
```

Or set the worker count explicitly:

```bash
automodel --nproc-per-node 2 Qwen/Qwen2.5-1.5B-Instruct examples/quickstart/openai_chat.jsonl
```

## Override Defaults

The quick path generates a normal recipe config, so CLI overrides still work:

```bash
automodel Qwen/Qwen2.5-1.5B-Instruct examples/quickstart/openai_chat.jsonl \
  --step_scheduler.max_steps=100 \
  --checkpoint.checkpoint_dir=./checkpoints/qwen-chat-lora
```

Default checkpoint locations:

| Data type | Default checkpoint directory |
|-----------|------------------------------|
| LLM chat | `./checkpoints/<model-name>-lora` |
| VLM chat | `./checkpoints/<model-name>-vlm-lora` |

## Move to YAML When Needed

Use an explicit YAML config when you need validation data, a custom optimizer,
full SFT instead of LoRA, custom dataset columns, packed sequences, launcher
sections, or multi-node settings.

Next steps:

- [Run on Your Local Workstation](../launcher/local-workstation.md)
- [Configuration](configuration.md)
- [LLM SFT and PEFT](llm/finetune.md)
- [VLM Dataset Guide](vlm/dataset.md)
