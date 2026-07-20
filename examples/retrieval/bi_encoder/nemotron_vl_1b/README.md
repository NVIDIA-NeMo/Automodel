# Finetuning Llama Nemotron VL 1B embedding model

This example shows how to finetune an embedding model for visual document retrieval: [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2).  The model can embed document pages in the form of image, text, or combined image–text inputs. Documents can be retrieved given a user query in text form. The model supports page images containing text, tables, charts, and infographics. You can check the performance of the model on vision document retrieval and text retrieval benchmarks [here](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2#evaluation-results).


## Finetuning for domain adaptation
You might want to further finetune the [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) open multimodal embedding model for a specific domain adaptation. 

The Nemo Automodel Retrieval recipe expects a train set with a Corpus ID-Based JSON schema as presented below, and also documented [here](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/retrieval-dataset.html#corpus-id-based-json-merlin-nemo-retriever-style).

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

The "corpus" path points to a directory with the corpus in parquet format containing a field with the document id.  
The "data" contains the list of training samples for contrastive learning, in particular the question, positive samples (pos_doc) and negative samples (neg_doc), which are document ids from the corpus. Positive and negative samples can be document page images or chunks of text (passages). 

### Prepare the ColPali source data

Run
[prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb](../prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb)
to prepare the ColPali example. The notebook:

1. Downloads a [train set](https://huggingface.co/datasets/Tevatron/colpali) with mined hard negatives and its
   [corpus](https://huggingface.co/datasets/Tevatron/colpali-corpus).
2. Writes the corpus as local Parquet shards and writes `colpali_train.json` in AutoModel's corpus ID-based schema.

These files are the source data for both loading paths below. The notebook does not create normalized Arrow; that is a
separate CPU preparation step for full-scale training.

After the notebook finishes, open [nemotron_vl_1b_example.yaml](nemotron_vl_1b_example.yaml) and replace the ColPali
path in `dataset.data_dir_list` with the generated `colpali_train.json` path. The example also includes the
[MIRACL train set](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/viewer/MIRACL) for multilingual
text retrieval. Use `num_samples` to control how many examples to load from each source.

### Choose how training loads the prepared data

#### Full-scale VL training: normalize on CPU first

For a full-scale VL run, use the updated example config as the input to the normalized dataset preparation script:

```bash
uv run python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_example.yaml \
  --output-dir /path/to/normalized_vl_retrieval
```

This command reads every source in the original `dataset.data_dir_list`, including the files produced by the notebook,
and writes one portable Arrow bundle. For large datasets on a Slurm cluster, use the CPU array launcher described in the
[retrieval data preparation tools](../../../../tools/retrieval/README.md).

Then replace the dataset section in the training config with the normalized loader:

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

#### Small runs: load the source data directly

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

Also, if you have a Weights & Biases (W&B) account, you can configure the YAML file to log training metrics during training.
```
wandb:
  project: YOUR_WANDB_PROJECT
  entity: YOUR_WANDB_ENTITY
  name: nemotron_vl_1b_embedding_example
```

### Fine-tuning with Automodel

Here is an example on how to fine-tune the [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) embedding model, distributed to 8x A100 GPUs in a single instance. If you have less GPUs available, you can set `--nproc-per-node` accordingly. For multi-node training, you can use `sbatch` command for Slurm clusters.

```bash
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_example.yaml
```
