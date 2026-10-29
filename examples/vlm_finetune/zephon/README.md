# Image training with Zephon

Use [Zephon](https://zephon.io/) with AutoModel's `FinetuneRecipeForVLM` to read
image conversations, prepare them with the model's processor, and pack whole
documents for training. The [main integration guide](../../llm_pretrain/zephon/README.md)
introduces configuration, output formats, cross-document attention, and
checkpointing. For Zephon concepts and API questions, see the
[Zephon User Guide](https://docs.zephon.io/).

Tokenization and packing are separate preparation stages: the text backend
supports raw, pretokenized, and prepacked input. The image backend currently
accepts raw conversations, with processor tokenization and image-placeholder
expansion before whole-document bin packing. Pretokenized/prepacked image
exports and `pack_algorithm: null` are not supported by this image path.

## Getting started

With AutoModel and its VLM dependencies installed, run from the repository root:

```bash
uv run python examples/vlm_finetune/zephon/prepare_images.py
uv run automodel examples/vlm_finetune/zephon/qwen2_5_vl.yaml --nproc-per-node 1
```

This example requires one CUDA GPU and access to the Qwen2.5-VL checkpoint named
in the recipe. It creates a small local image corpus and reads it once.
`step_scheduler.max_steps` caps training; the corpus may end sooner.

For a tiny Mistral3 model trained from random weights, with a local processor:

```bash
uv run python examples/vlm_finetune/zephon/prepare_mistral3.py
uv run automodel examples/vlm_finetune/zephon/mistral3_tiny.yaml --nproc-per-node 4
```

This example uses two-way tensor and pipeline parallelism on four CUDA GPUs.
It does not download pretrained weights. For your own checkpoint, select a
matching model and processor in the ordinary AutoModel recipe sections.

## Configuring image data

Use the image dataset and dataloader configs with the shared Zephon options:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonVlmSourceConfig

dataloader:
  _target_: nemo_automodel.components.datasets.zephon.ZephonVlmDataloaderConfig
  zephon:
    sources:
    - name: images
      path: /data/image-conversations
      fmt: jsonl
    packing:
      seq_len: 4096
      pack_algorithm: first_fit
      pack_num_bins: 16
    source:
      stop_after_passes: 1
    checkpoint:
      run_id: my-image-training-run
```

Each JSONL document uses AutoModel's conversation format, for example:

```json
{
  "conversation": [
    {"role": "user", "content": [
      {"type": "image", "image": "/data/photo.png"},
      {"type": "text", "text": "Describe the image."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "A blue square."}
    ]}
  ]
}
```

Store each document on one line in the actual JSONL file. Absolute image paths
avoid dependence on the workers' working directory. Documents may contain
multiple images, images of different resolutions, or text only.

The processor and chat template determine tokenization and assistant loss
masking. Packing uses the length after image placeholders have been expanded
into image tokens. Use `first_fit` or `best_fit`: images and their documents
must stay whole. Oversized documents are dropped by default; set
`packing.drop_oversized: false` to raise instead. Images are never truncated to
fit a bin. See [Zephon's packing guide](https://docs.zephon.io/pipelines/packing.html)
for bin-packing options.

`packing.seq_len` is optional. When omitted, it uses the text decoder's
configured input-token limit. Set a smaller value for shorter training inputs;
values above a known model limit are rejected in either cross-document attention
mode. Expanded image tokens count toward this limit. If the model does not
declare a limit, set `packing.seq_len` explicitly.

By default, all-text packs include an ignored synthetic image segment so the
vision tower still participates in training. The loader reserves space for that
segment when packing, reducing the available document length. Set
`dataset.inject_fake_images: false` only when your model and parallel setup
permit skipping the vision tower.

`dataset.on_media_error: drop` skips unreadable media and image-token mismatches;
the default, `raise`, stops on these errors. Malformed conversations and other
processor errors still raise. `workers.tokenize_parallelism` controls the
processor stage, including image decoding. Tune it for the CPU and memory
available to each training rank.

Image sources share the text integration's cache, execution, and checkpoint
settings. Token-aware source calibration is not supported for images yet.
When mixing multiple image sources, set `source.token_estimation: false`.

## Models and output formats

The integration supports Qwen2.5-VL, dense Qwen3-VL, Mistral3 with a
Mistral/Ministral3 text backbone without sliding attention, and dense Gemma4
images. Reuse the dataset and dataloader sections above with the appropriate
model and processor. Model-specific parallelism restrictions still apply;
for example, native dense Qwen3-VL does not support TP or PP. Gemma4 image
training uses its native SDPA/eager attention policy.

Keep `dataloader.output_format: batched` unless your model and attention backend
support native THD inputs. Batched output supports Flash Attention on compatible
models and is the supported format for VLM pipeline parallelism. THD requires
`dataset.inject_fake_images: false`, contiguous eligible tokens within each
document, and no VLM pipeline parallelism. Qwen3-VL currently does not support
THD. These choices affect the model input format, not Zephon's packing algorithm.

This integration covers raw image conversations. Video, audio, already encoded
or packed VLM inputs, context parallelism, and Zephon validation loaders are not
supported. AutoModel's existing validation loaders remain available.
