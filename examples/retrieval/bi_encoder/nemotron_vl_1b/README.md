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

**Before a full-scale VL training run, prepare the dataset on CPU nodes.** Starting GPU training directly from a raw
image corpus can leave every allocated GPU waiting while the corpus is loaded and its dataset cache is built. Use the
normalized dataset preparation script in the
[retrieval data preparation tools](../../../../tools/retrieval/README.md) to build an Arrow bundle on CPU nodes first.
For text-only retrieval, the original dataset path is usually sufficient. If its initial startup is slow, use the CPU
cache-warming script instead.

### Preparing the vision retrieval train set
This example includes a notebook that demonstrates how to prepare a vision retrieval training dataset using [ColPali](https://huggingface.co/datasets/vidore/colpali_train_set): [prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb](../prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb). It performs the following steps:
1. Downloads a [train set](https://huggingface.co/datasets/Tevatron/colpali) that includes mined hard-negatives (for faster contrastive learning) and the corresponding [corpus](https://huggingface.co/datasets/Tevatron/colpali-corpus)
2. Converts and saves that data into the Corpus ID-Based JSON schema expected for training biencoder models with Nemo Automodel.

After running the notebook, open [nemotron_vl_1b_example.yaml](nemotron_vl_1b_example.yaml) and modify inside `data_dir_list` the path of ColPali dataset (`colpali_train.json`) with the one you set when running the notebook above, as in below example. 
Notice that in this example there is a second dataset in `data_dir_list`, where the positive/negative samples are text passages (from [MIRACL train set](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/viewer/MIRACL), focused on multi-lingual text retrieval). You can also set the number of samples per dataset as in the following example.

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - path: /path/to/trainset/colpali_train.json 
        num_samples: 5000
      - path: hf://nvidia/embed-nemotron-dataset-v1/MIRACL
        num_samples: 5000
```

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
