# Fine-Tuning Llama Nemotron VL 1B Embedding Model

This example shows how to fine-tune an embedding model for visual document retrieval: [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2). The model can embed document pages in the form of image, text, or combined image-text inputs. Documents can be retrieved given a user query in text form. The model supports page images containing text, tables, charts, and infographics. You can check the performance of the model on vision document retrieval and text retrieval benchmarks [here](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2#evaluation-results).


## Fine-Tune for Domain Adaptation
You can fine-tune the [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) open multimodal embedding model for specific domain adaptation.

The NeMo AutoModel Retrieval recipe expects a training set with a Corpus ID-Based JSON schema, which is presented below and documented [here](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/retrieval-dataset.html#corpus-id-based-json-merlin-nemo-retriever-style).

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

The "corpus" path points to a directory with the corpus in Parquet format containing a field with the document ID.  
The "data" field contains the list of training samples for contrastive learning, in particular the question, positive samples (`pos_doc`), and negative samples (`neg_doc`), which are document IDs from the corpus. Positive and negative samples can be document page images or chunks of text (passages). 

### Prepare the Vision Retrieval Training Set
This example includes a notebook that demonstrates how to prepare a vision retrieval training dataset using [ColPali](https://huggingface.co/datasets/vidore/colpali_train_set): [prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb](prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb). It performs the following steps:
1. Downloads a [train set](https://huggingface.co/datasets/Tevatron/colpali) that includes mined hard-negatives (for faster contrastive learning) and the corresponding [corpus](https://huggingface.co/datasets/Tevatron/colpali-corpus)
2. Converts and saves that data into the Corpus ID-Based JSON schema expected for training biencoder models with Nemo Automodel.

After running the notebook, open [nemotron_vl_1b_example.yaml](nemotron_vl_1b_example.yaml) and modify the path of the ColPali dataset (`colpali_train.json`) inside `data_dir_list` to match the path set when running the notebook, as shown in the example below. 
Notice that a second dataset is present in `data_dir_list`, where the positive and negative samples are text passages (from the [MIRACL training set](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1/viewer/MIRACL), which focuses on multilingual text retrieval). You can also set the number of samples per dataset, as shown in the following example.

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig
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

### Fine-Tune with NeMo AutoModel

Here is an example of how to fine-tune the [nvidia/llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) embedding model, distributed across 8 A100 GPUs in a single instance. If you have fewer GPUs available, you can set `--nproc-per-node` accordingly. For multi-node training, use the `sbatch` command for Slurm clusters.

```bash
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_example.yaml
```

### Use the Higher-Throughput Configuration

For higher throughput, use [nemotron_vl_1b_optimized.yaml](nemotron_vl_1b_optimized.yaml). It trains the same model and
uses the same data format as the base example, with these performance options enabled:

- A custom Llama backend with Transformer Engine fused QKV and MLP projections.
- Transformer Engine SigLIP encoder layers, with the unused SigLIP pooling head disabled.
- Bidirectional attention masks prepared by the data loader instead of during every training step.
- DDP reducer settings tuned for this workload.

On the reference workload, this trains the same model about 15% faster per step (~18% more samples per second per GPU) than the tuned DDP baseline, with matching loss curves. The optimizations are opt-in flags in the `model:` and `bi_encoder_optimization:` sections, so the base example remains the simplest starting point.

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
