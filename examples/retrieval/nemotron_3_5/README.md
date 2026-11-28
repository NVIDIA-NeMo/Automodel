# Fine-Tune Nemotron 3.5 8B Embedding and Reranking Models

Use NeMo AutoModel retrieval recipes to fine-tune the released Nemotron 3.5 8B
embedding and reranking checkpoints on your data. Each example accepts text queries paired with documents
containing an image, text, or both.

Choose the configuration for your model:

| Model | Configuration | Fine-Tuning Scope |
| --- | --- | --- |
| [Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | [embed_8b_multimodal.yaml](embed_8b_multimodal.yaml) | Contrastive training of the vision encoder and projector with a frozen language model |
| [Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | [rerank_8b_multimodal.yaml](rerank_8b_multimodal.yaml) | Fine-tuning all parameters with listwise cross-entropy |

These examples adapt existing checkpoints. They do not reproduce the original
training pipeline, dataset mixtures, or reported results. Refer to the linked
model cards for the released models' training methods and data sources.

## Prerequisites

Follow the [installation guide](../../../docs/guides/installation.mdx), including
vision dependencies, Transformer Engine, and FlashAttention 2. The launch command
below uses eight NVIDIA GPUs with BF16 support on one node. Memory requirements
depend on the model and document lengths.

The configurations load the released Hugging Face checkpoints directly, including
their tokenizer and image processor. No checkpoint assembly or earlier text
training is required. To continue from a local checkpoint, override both
`model.pretrained_model_name_or_path` and `tokenizer.pretrained_model_name_or_path`
with its directory.

The embedding configuration sets `freeze_language_model: true` to preserve the
released text component while updating the vision encoder and multimodal
projector. Set it to `false` to also adapt the language model, for example when
fine-tuning on text-only data. This increases training memory requirements.

## Prepare Data

Prepare `train.json` using AutoModel's
[corpus ID-based JSON format](../../../docs/guides/llm/retrieval-dataset.mdx#corpus-id-based-json).
The file references your document corpus and lists queries with positive and
negative document IDs.

For image documents with optional text, save a Hugging Face `train` split with
`docid`, `image`, and `text` columns. In the corpus's `merlin_metadata.json`, set
`class` to `WikiSSNQDataset`, the existing loader for this image-and-text layout,
and set `corpus_id` to match the training records. Use empty `text` strings for
image-only documents and a Hugging Face `Image` feature that decodes the images.

Each query needs positive and negative document IDs. The configurations sample
one positive and three negatives, repeating negatives when necessary.
The setting `use_text_in_document: true` retains accompanying text.
Include image documents to train the embedding models' vision components.
Provide at least 128 queries for embedding training or 256 for reranker training.
Embedding training uses 100 warmup steps, and reranker training uses 200. Each run
must include more optimizer steps than warmup steps. For a short validation run, add
`--lr_scheduler.lr_warmup_steps 0` to the launch command.

The configurations use an image longest-edge setting of 1,120 pixels and one
query per GPU per microbatch. The embedding configuration limits document
sequences to 4,096 tokens. The reranker limits each query-document pair to 3,072
tokens. Each query has four document candidates. Gradient accumulation preserves
the configured global batch size.

## Run Training

Run from the repository root. Replace `<config>` with a filename from the table,
and replace the training file placeholder:

```bash
uv run --no-sync automodel 'examples/retrieval/nemotron_3_5/<config>' \
  --nproc-per-node 8 \
  --dataset.data_dir_list '<path/to/train.json>'
```

Training runs for one epoch using FSDP2 with BF16 model storage and compute.
Transformer Engine FusedAdam uses AdamW mode with FP32 master weights.
Under `checkpoint.checkpoint_dir`, check `training.jsonl` for finite losses and
advancing steps. The final checkpoint contains the model in `model/consolidated`.

## Adjust Memory and Distributed Training

The configurations set `reshard_after_forward: true` to reduce peak memory on an
eight-GPU node. FSDP2 reshards each layer after its forward pass, at the cost of
additional parameter all-gathers during recomputation. If your GPUs have enough
memory, setting this option to `false` can reduce communication overhead.

Keep `global_batch_size` divisible by `local_batch_size` times the data-parallel
world size. Increasing the world size reduces each FSDP2 shard and can reduce the
number of gradient-accumulation microbatches. Reducing `local_batch_size`,
`image_longest_edge`, and `p_max_length` or `rerank_max_length` lowers peak memory.
Changing the global batch size also changes optimization behavior.

Freezing a tower removes its gradients and optimizer state, but its weights and
forward activations still use memory. The embedding configuration freezes the
language model. The reranker configuration updates both towers and therefore
requires more memory. Adjust the learning rate and schedule for your data and
effective global batch size. Use `--checkpoint.checkpoint_dir` to select a new
output directory for each run.

## Original Training Approach

The released models were produced through earlier initialization and training
stages. The examples above start from the finished checkpoints for fine-tuning.
The following describes how to prepare a different starting point for the
multimodal stage.

### Embed 8B

The [embedding model card](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16#model-architecture)
describes combining the text component from
[Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)
with the vision encoder from
[Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16).
The language model remained frozen while the vision encoder and multimodal
projector were trained on image and combined image-text data.

To use that initialization, manually assemble a local Mistral3 checkpoint:
replace its language-model weights with the Nemotron 3 Embed weights and retain
the compatible vision encoder and multimodal projector. Save the configuration,
weights, tokenizer, and image processor together. Use `embed_8b_multimodal.yaml`
with `freeze_language_model: true`. Freezing an unmodified Ministral language
model does not provide the pretrained Nemotron text embeddings.

### Rerank 8B

The [reranker model card](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16#model-architecture)
describes two stages: text-only reranker training from the Ministral 3 8B
backbone, followed by training on image and combined image-text data.

For the multimodal stage, provide a complete Mistral3 reranker checkpoint from
the text-training stage, preserving its trained language component and scoring
head together with the vision encoder and projector. Use
`rerank_8b_multimodal.yaml`, which keeps both towers and the scoring head
trainable. This initialization does not use the Nemotron 3 Embed language weights.

### Use a Prepared Checkpoint

Replace `<config>` with the embedding or reranking YAML filename, and provide
your prepared checkpoint and training data:

```bash
uv run --no-sync automodel 'examples/retrieval/nemotron_3_5/<config>' \
  --nproc-per-node 8 \
  --model.pretrained_model_name_or_path '<path/to/prepared-checkpoint>' \
  --tokenizer.pretrained_model_name_or_path '<path/to/prepared-checkpoint>' \
  --dataset.data_dir_list '<path/to/train.json>'
```

Checkpoint assembly and the reranker's earlier text-training stage are not
implemented by these examples. The model cards describe data sources, but these
files do not supply the complete processed datasets, mixtures, or schedules
needed to reproduce the released checkpoints and reported results.
