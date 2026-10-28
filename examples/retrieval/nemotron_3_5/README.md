# Train Nemotron 3.5 8B Embedding and Reranking Models

Use NeMo AutoModel retrieval recipes to train the Nemotron 3.5 8B embedding and
reranking models. Each example accepts text queries paired with documents
containing an image, text, or both.

Choose the configuration for your model and training stage:

| Model | Configuration | Training Stage |
| --- | --- | --- |
| [Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | [embed_8b_multimodal.yaml](embed_8b_multimodal.yaml) | Contrastive training of the vision encoder and projector with a frozen language model |
| [Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | [rerank_8b_multimodal.yaml](rerank_8b_multimodal.yaml) | Multimodal training from a text-trained reranker, updating all parameters with listwise cross-entropy |

These are representative stage examples. The model cards document the released
models' training methods and data sources. The
[Nemotron-3-Embed-8B dataset overview](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16#dataset-overview)
describes the earlier text training.

## Prepare the Starting Checkpoint

Follow the [installation guide](../../../docs/guides/installation.mdx), including
vision dependencies, Transformer Engine, and FlashAttention 2. The launch command
below uses eight NVIDIA GPUs with BF16 support on one node. Memory requirements
depend on the model and document lengths.

Provide a local Hugging Face checkpoint with its configuration, weights, tokenizer,
and image processor. Select the initialization for the stage:

- **Embed 8B:** Manually assemble a local Mistral3 multimodal checkpoint before
  training. Start with
  [Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16)
  and replace its language-model component with weights from
  [Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16).
  Retain the compatible Mistral3 vision and multimodal components. This example
  does not perform checkpoint assembly.
- **Rerank 8B:** A complete Mistral3 reranker checkpoint from text training,
  including the trained scoring head, vision encoder, and projector.

The examples begin at these checkpoints. Checkpoint assembly and earlier text
training are separate steps. Using a released Nemotron 3.5 checkpoint as the
starting checkpoint performs continued training.

The embedding configuration assumes the assembled checkpoint and sets
`freeze_language_model: true` to preserve the trained Nemotron text component
while updating the vision encoder and multimodal projector. If you start from an
unmodified Mistral3 checkpoint, set `freeze_language_model: false` and fine-tune
the language model.

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
and replace the checkpoint and training file placeholders:

```bash
uv run --no-sync automodel 'examples/retrieval/nemotron_3_5/<config>' \
  --nproc-per-node 8 \
  --model.pretrained_model_name_or_path '<path/to/starting-checkpoint>' \
  --tokenizer.pretrained_model_name_or_path '<path/to/starting-checkpoint>' \
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
