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
[Corpus ID-Based JSON documentation](https://docs.nvidia.com/nemo/automodel/latest/datasets/retrieval-dataset#corpus-id-based-json-merlin-or-nemo-retriever-style).

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

For a full-scale VL run, use the updated example config as the input to the normalized dataset preparation script:

```bash
uv run python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_example.yaml \
  --output-dir /path/to/normalized_vl_retrieval
```

This command reads every source in the original `dataset.data_dir_list`, including the files produced by the notebook,
and writes one portable Arrow bundle. For large datasets on a Slurm cluster, use the CPU array launcher described in the
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

For a smoke test or a small dataset, skip normalization and keep the original dataset config. It reads the notebook's
JSON and corpus files directly:

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
