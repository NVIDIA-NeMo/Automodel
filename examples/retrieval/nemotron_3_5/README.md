# Train Nemotron 3.5 Embedding and Reranking Models

Use NeMo AutoModel retrieval recipes for multimodal embedding training,
embedding distillation, or reranker training. Each example accepts text queries
paired with documents containing an image, text, or both.

Choose the configuration for your model and training stage:

| Model | Configuration | Training Stage |
| --- | --- | --- |
| [Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | [embed_8b_multimodal.yaml](embed_8b_multimodal.yaml) | Contrastive training of the vision encoder and projector with a frozen language model |
| [Nemotron-3.5-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | [embed_1b_distill.yaml](embed_1b_distill.yaml) | Distillation from Embed 8B with cosine/MSE losses and a frozen student language model |
| [Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | [rerank_8b_multimodal.yaml](rerank_8b_multimodal.yaml) | Multimodal training from a text-trained reranker, updating all parameters with listwise cross-entropy |

These are representative stage examples. The model cards document the released
models' training methods and data sources. The
[Nemotron-3-Embed-8B dataset overview](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16#dataset-overview)
describes the earlier text training.

> [!IMPORTANT]
> The Embed 1B distillation example requires an update to AutoModel's student and
> teacher wrappers to read `text_config.hidden_size` for Mistral3 checkpoints.
> The current wrappers read `config.hidden_size` and fail during model construction.

## Prepare the Starting Checkpoint

Follow the [installation guide](../../../docs/guides/installation.mdx), including
vision dependencies, Transformer Engine, and FlashAttention 2. The launch command
below uses eight NVIDIA GPUs with BF16 support on one node. Memory requirements
depend on the model and document lengths. Distillation loads both the student and
teacher.

Provide a local Hugging Face checkpoint with its configuration, weights, tokenizer,
and image processor. Select the initialization for the stage:

- **Embed 8B:** A prepared Mistral3 multimodal checkpoint with the Nemotron-3-Embed-8B
  language model and a compatible vision encoder and projector, as described in the model card.
- **Embed 1B:** A prepared Mistral3 multimodal student with the Nemotron-3-Embed-1B
  language model. The configuration loads the released Embed 8B teacher and learns
  a projection from the student's 2,048 dimensions to the teacher's 4,096 dimensions.
  Student and teacher must use compatible token IDs and image preprocessing.
- **Rerank 8B:** A complete Mistral3 reranker checkpoint from text training,
  including the trained scoring head, vision encoder, and projector.

The examples begin at these checkpoints. Checkpoint assembly and earlier text
training are separate steps. Using a released Nemotron 3.5 checkpoint as the
starting checkpoint performs continued training.

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
The reranker uses 200 warmup steps, so the run must exceed 200 optimizer steps.
For a smaller dataset, add `--lr_scheduler.lr_warmup_steps 0` to the launch command.

The training configurations resize images with a longest-edge setting of 1,284
pixels and limit document sequences to 4,096 tokens. These training settings differ
from the released processors' 1,120-pixel inference default.

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
Distillation exports the student encoder there and saves its training projection
separately as `projection.pt` in the epoch checkpoint directory.

Adjust the learning rate, schedule, batch sizes, and sequence lengths for your data
and hardware. Keep `global_batch_size` divisible by `local_batch_size` times the
number of GPUs. Use `--checkpoint.checkpoint_dir` to select a new output directory
for each run.
