# Fine-Tune the Llama Nemotron VL 1B Embedding Model

This example shows how to fine-tune an embedding model for visual document retrieval:
[nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2). The model can
embed document pages as image, text, or combined image-text inputs. Documents can be retrieved given a user query in
text form. The model supports page images containing text, tables, charts, and infographics. Review the model's
performance on the
[vision document retrieval and text retrieval benchmarks](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2#evaluation-results).

## Fine-Tune for Domain Adaptation

You can further fine-tune the
[nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) open multimodal
embedding model to adapt it to a specific domain.

The NeMo AutoModel Retrieval recipe expects a training set with the corpus ID-based JSON schema shown below. For
details, refer to the
[Corpus ID-Based JSON documentation](https://docs.nvidia.com/nemo/automodel/latest/datasets/retrieval-dataset#corpus-id-based-json).

```json
{
  "corpus": {
    "path": "/path/to/corpus/colpali_train_set"
  },
  "data": [
    {
      "question_id": "q2",
      "question": "What is the primary purpose of the PTC in lithium batteries?",
      "corpus_id": "colpali_train_set",
      "pos_doc": [
        {"id": "2"}
      ],
      "neg_doc": [
        {"id": "69560"},
        {"id": "112685"},
        {"id": "5132"}, ...
    }, ...
  ]
}
```

The `corpus` path points to a directory that contains the corpus in Parquet format, with a field for the document ID.
The `data` key contains the training samples for contrastive learning: the question, positive samples (`pos_doc`), and
negative samples (`neg_doc`). The positive and negative samples are document IDs from the corpus and can represent
document page images or chunks of text (passages).

### Prepare the ColPali Source Data

Run
[`prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb`](../prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb)
to prepare the ColPali example. The notebook:

1. Downloads a [train set](https://huggingface.co/datasets/Tevatron/colpali) with mined hard negatives and its
   [corpus](https://huggingface.co/datasets/Tevatron/colpali-corpus).
2. Writes the corpus as local Parquet shards and writes `colpali_train.json` in AutoModel's corpus ID-based schema.

These files are the source data for both loading paths below. The notebook does not create normalized Arrow; that is a
separate CPU preparation step for full-scale training.

After the notebook finishes, open [`nemotron_vl_1b_example.yaml`](nemotron_vl_1b_example.yaml) and replace the ColPali
path in `dataset.data_dir_list` with the generated `colpali_train.json` path. The example also includes the
[MIRACL train set](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/viewer/MIRACL) for multilingual
text retrieval. Use `num_samples` to control how many examples to load from each source.

### Choose How Training Loads the Prepared Data

#### Normalize on CPU First for Full-Scale VL Training

The normalization tool accepts local corpus ID-based JSON sources. Before preparing a full-scale VL dataset, make a
copy of the example config for CPU preparation:

1. Keep the generated local `colpali_train.json` source.
2. Remove the `hf://` MIRACL source if the run does not need it, or materialize MIRACL as local corpus-backed data:

   ```bash
   uv run python examples/retrieval/bi_encoder/llama_embed_nemotron_8b/data_preparation.py \
     --download-path ./embed_nemotron_dataset_v1
   ```

   Then replace the MIRACL URI in the preparation config with
   `./embed_nemotron_dataset_v1/MIRACL/MIRACL.json`.

Run the normalization tool with that local-source config:

```bash
uv run python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config /path/to/nemotron_vl_1b_normalized_prep.yaml \
  --output-dir /path/to/normalized_vl_retrieval
```

This command reads every local source in `dataset.data_dir_list` and writes one portable Arrow bundle. For large
datasets on a Slurm cluster, use the CPU array launcher described in the
[retrieval data preparation tools](../../../../tools/retrieval/README.md).

Then replace the `dataset` section in the training config with the normalized loader:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig
  data_path: /path/to/normalized_vl_retrieval
  model_type: bi_encoder
  data_type: train
  n_passages: 5
```

Starting GPU training directly from a large image corpus can leave every allocated GPU waiting while the corpus is
loaded and its dataset cache is built. Normalizing on CPU moves that work before the GPU allocation.

#### Load the Source Data Directly for Small Runs

For a small verification run or a small dataset, skip normalization and keep the original dataset config. It reads the
notebook's JSON and corpus files directly:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig
  model_type: bi_encoder
  data_dir_list:
    - path: /path/to/trainset/colpali_train.json
      num_samples: 5000
    - path: hf://nvidia/embed-nemotron-dataset-v1/MIRACL
      num_samples: 5000
  data_type: train
  n_passages: 5
```

The direct path is usually sufficient for text-only retrieval. If its initial startup is slow, the retrieval data tools
also provide a CPU cache-warming script that keeps this dataset configuration unchanged.

If you have a Weights & Biases (W&B) account, configure the YAML file to log training metrics during training:

```yaml
wandb:
  project: YOUR_WANDB_PROJECT
  entity: YOUR_WANDB_ENTITY
  name: nemotron_vl_1b_embedding_example
```

### Fine-Tune with AutoModel

The following example fine-tunes the
[nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) embedding model
across eight A100 GPUs in a single instance. If you have fewer GPUs available, set `--nproc-per-node` accordingly. For
multi-node training on a Slurm cluster, use `sbatch`.

```bash
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_example.yaml
```

### Use the Optimized Configuration

For better training performance, use [nemotron_vl_1b_optimized.yaml](nemotron_vl_1b_optimized.yaml). It trains the
same model and uses the same data format as the base example, while enabling:

- An optimized Llama backend with Transformer Engine fused QKV and MLP projections.
- Optimized Transformer Engine SigLIP encoder layers, with the unused SigLIP pooling head disabled.
- Bidirectional attention masks prepared by the data loader instead of during every training step.

On the reference workload, the optimized configuration reduced step time by about 15% and increased per-GPU
throughput by about 18% compared with `nemotron_vl_1b_example.yaml`, while producing comparable loss values. The
model-specific switches are grouped under `model.optimization_config`, while bidirectional mask preparation is under
`bi_encoder_optimization`. Use the base example when you prefer the simplest configuration.

The configuration uses `global_batch_size: 64` and `local_batch_size: 2`. On one 8-GPU node, this means four gradient
accumulation steps per optimizer step, so the file sets `distributed.static_graph: false`. Set the allocator option
before launching on 80 GB GPUs:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_optimized.yaml
```

With 4 nodes and 8 GPUs per node, the same batch settings require no gradient accumulation. For that launch, enable
the DDP static-graph optimization:

```yaml
distributed:
  strategy: ddp
  static_graph: true
```

Keep `static_graph: false` for any setup that uses more than one gradient accumulation step.

Note that `torchrun --nproc-per-node` launches a single node. For multi-node runs, submit through your cluster scheduler (e.g., `sbatch` on Slurm), as described in the base example above.

### Choose a GPU Budget

The optimized config keeps `global_batch_size=64` fixed. Adjust it to your available GPUs with these options:

1. **Gradient accumulation**: Whenever `local_batch_size × GPUs` is smaller than `global_batch_size`, the recipe uses
   `global_batch_size / (local_batch_size × GPUs)` accumulation steps automatically.
2. **Activation checkpointing**: If you want to use fewer GPUs with a larger `local_batch_size`, enable
   `distributed.activation_checkpointing: true`. Use `activation_checkpointing_scope: vision` to checkpoint only the
   vision tower, or `activation_checkpointing_scope: all` to checkpoint the full model. This uses less memory at the
   cost of additional computation. See [Use Gradient (Activation) Checkpointing](https://docs.nvidia.com/nemo/automodel/latest/guides/gradient-checkpointing.html).

Measured reference points on 80GB GPUs (262k-sample VL retrieval workload, no gradient accumulation in any row):

| GPUs | local/global batch | Activation ckpt. | Samples/s per GPU | Peak mem/GPU | Approx. epoch |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 (8 nodes) | 1/64 | no | ~1.95 | ~50GiB | ~35m |
| 32 (4 nodes) | 2/64 | no | ~2.20 | ~70GiB | ~62m |
| 8 (1 node) | 8/64 | vision tower only | ~2.11 | ~80GiB | ~4.3h |
| 4 | 16/64 | full model | ~2.05 | ~61GiB | ~8.9h |

Per-GPU efficiency peaks at the 32-GPU `local_batch_size=2` configuration, and the single-node setups stay close to it (within ~5-7%) even with activation-checkpointing recompute. All three are meaningfully more GPU-hour-efficient than the 64-GPU `local_batch_size=1` run. Using fewer GPUs costs wall-clock time but almost no total GPU-hours.

For maximum throughput on one 8-GPU node, override the optimized config with `local_batch_size=8` and checkpoint only the vision tower:

```yaml
step_scheduler:
  global_batch_size: 64
  local_batch_size: 8

distributed:
  strategy: ddp
  static_graph: true
  activation_checkpointing: true
  activation_checkpointing_scope: vision
```

For example, to run on 4 GPUs, checkpoint the full model and raise `local_batch_size` to 16:

```yaml
step_scheduler:
  global_batch_size: 64
  local_batch_size: 16

distributed:
  strategy: ddp
  static_graph: true
  activation_checkpointing: true
  activation_checkpointing_scope: all
```
