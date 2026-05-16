# Quickstart Samples

This directory contains small
[OpenAI-format chat](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)
JSONL datasets for the AutoModel quickstart flow.

## Files

| File | Use case |
|------|----------|
| [`openai_chat.jsonl`](openai_chat.jsonl) | Text-only LLM LoRA fine-tuning |
| [`openai_vlm_chat.jsonl`](openai_vlm_chat.jsonl) | VLM LoRA fine-tuning with image content parts |

## Run

Text-only LLM:

```bash
automodel Qwen/Qwen2.5-1.5B-Instruct examples/quickstart/openai_chat.jsonl
```

VLM:

```bash
automodel Qwen/Qwen2.5-VL-3B-Instruct examples/quickstart/openai_vlm_chat.jsonl
```

Both files use one JSON object per line. Each object has a `messages` list with
OpenAI-style chat roles such as `system`, `user`, and `assistant`.

Text-only row excerpt:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a concise assistant."
    },
    {
      "role": "user",
      "content": "Give one reason to use LoRA."
    },
    {
      "role": "assistant",
      "content": "LoRA trains small adapters, so fine-tuning uses less memory."
    }
  ]
}
```

Multiturn row excerpt:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Turn this into a title: memory efficient fine tuning"
    },
    {
      "role": "assistant",
      "content": "Memory-Efficient Fine-Tuning"
    },
    {
      "role": "user",
      "content": "Make it more specific."
    },
    {
      "role": "assistant",
      "content": "Memory-Efficient Fine-Tuning with LoRA"
    }
  ]
}
```

VLM row excerpt:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What color is the tiny square?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,..."
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "It is red."
    }
  ]
}
```

The VLM sample uses inline `data:image/...` URLs so it can run as a
self-contained format example. Real datasets can use relative image paths,
absolute paths, HTTP(S) URLs, or supported image content parts such as
`image_url`, `input_image`, and `image`.

## Next Steps

- [LLM SFT and PEFT guide](../../docs/guides/llm/finetune.md)
- [VLM dataset guide](../../docs/guides/vlm/dataset.md)
- [Quickstart guide](../../docs/guides/quickstart.md)
