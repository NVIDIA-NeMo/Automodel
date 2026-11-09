# Fine-Tune a VLM with Zephon

[Zephon](https://zephon.io/) reads image conversations and packs whole documents
for AutoModel’s `FinetuneRecipeForVLM`.

## Train

Install AutoModel with its [VLM dependencies](../../../README.md#getting-started).
In [train.yaml](train.yaml), set `dataset.zephon.sources[0].path` to your
self-contained shard directory, then run from the repository root:

```bash
uv run automodel examples/vlm_finetune/zephon/train.yaml --nproc-per-node 1
```

The example fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` on one CUDA GPU. The model
and processor are downloaded from Hugging Face. Each document contains its
conversation **and embedded image bytes**. Parquet is a convenient choice;
Zephon detects the format from the source. Existing embedded HF image columns
(`{bytes, path}`) and LitData records also work. Path-only image references need
an offline export first: see [preparing image data](data.md).

## How It Works

The recipe selects `ZephonVlmDatasetConfig` under `dataset`. When
`dataloader._target_` is omitted, AutoModel selects the matching
`ZephonVlmDataloaderConfig` automatically. This thin metadata adapter uses
AutoModel’s shared VLM construction path; `dataset.zephon`
configures the source, packing and runtime, while `processor` and
`dataset.chat_template` select the processor and conversation template.
The same single-source `path_or_dataset_id` and `paths` fields described in the
integration guide are available. They select sources without performing a native
dataset class's conversation/image transformations. The image adapter normalizes
`conversation`, `messages`, ShareGPT `conversations`, or image/question/answer
records. Per-source `input.column_mapping` handles different column names.

Zephon fetches source shards through its cache; decoding uses only the fetched
image bytes. The shared [cache settings](../../llm_pretrain/zephon/README.md)
also bound the Parquet decoded row-group cache separately from downloaded shards.

AutoModel’s processor applies the chat template and expands image placeholders
before Zephon measures and bin-packs the documents. Training supervises assistant
tokens. Packing defaults to `first_fit` with 64 bins; image documents stay whole,
and oversized documents are dropped by default. Unlike text mixtures, image
mixtures use sample-based weights: media-aware token estimation is unsupported.
The finite source is read once, and `max_steps` caps training.

Qwen2.5-VL, dense Qwen3-VL, non-sliding Mistral3, and dense Gemma4 image paths are
supported, subject to their model or backend restrictions. Video, audio, and context
parallelism are unsupported.

See the [integration guide](../../llm_pretrain/zephon/README.md) for configuration,
output formats, cross-document attention, and checkpointing. Zephon pipeline
options and APIs are covered in the [Zephon User Guide](https://docs.zephon.io/).
