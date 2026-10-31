# VLM fine-tuning with Zephon

[Zephon](https://zephon.io/) reads image conversations and packs whole documents
for AutoModel’s `FinetuneRecipeForVLM`.

## Train

Install AutoModel with its [VLM dependencies](../../../README.md#getting-started).
In [train.yaml](train.yaml), set `dataloader.zephon.sources[0].path` to your
JSONL shard directory, then run from the repository root:

```bash
uv run automodel examples/vlm_finetune/zephon/train.yaml --nproc-per-node 1
```

The example fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` on one CUDA GPU. The model
and processor are downloaded from Hugging Face. Each JSONL line uses AutoModel’s
conversation format, with image paths accessible to every training rank:

```json
{"conversation": [{"role": "user", "content": [{"type": "image", "image": "/data/photo.png"}, {"type": "text", "text": "Describe the image."}]}, {"role": "assistant", "content": [{"type": "text", "text": "A blue square."}]}]}
```

## How it works

AutoModel’s processor applies the chat template and expands image placeholders
before Zephon measures and bin-packs the documents. Training supervises assistant
tokens. Image documents stay whole; oversized documents are dropped by default.
The finite source is read once, and `max_steps` caps training.

Qwen2.5-VL, dense Qwen3-VL, non-sliding Mistral3 and dense Gemma4 image paths are
supported, subject to their model/backend restrictions. Video, audio and context
parallelism are unsupported.

See the [integration guide](../../llm_pretrain/zephon/README.md) for configuration,
output formats, cross-document attention and checkpointing. Zephon pipeline
options and APIs are covered in the [Zephon User Guide](https://docs.zephon.io/).
