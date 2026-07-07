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

### Preparing the vision retrieval train set
This example includes a notebook that demonstrates how to prepare a vision retrieval training dataset using [ColPali](https://huggingface.co/datasets/vidore/colpali_train_set): [prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb]([prepare_dataset_for_vdr/convert_colpali_dataset_for_training.ipynb]). It performs the following steps:
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

### Optimized configuration for higher throughput

For higher throughput on the VL bi-encoder path, use the optimized configs. Compared with the base example above, they enable:

- the optimized custom LLaMA backend with Transformer Engine fused RMSNorm+QKV and RMSNorm+MLP kernels,
- Transformer Engine fused SigLIP vision encoder layers, with the unused SigLIP pooling head frozen,
- bidirectional attention masks precomputed in the dataloader (removing per-step host-device syncs),
- tuned DDP settings (bucket size, static graph, gradient-as-bucket-view).

On the reference workload this trains the same model about 15% faster per step (~18% more samples/s per GPU) than the tuned DDP baseline, with matching loss curves. All optimizations are opt-in config flags — see the `model:` and `bi_encoder_optimization:` sections of the configs.

Two ready-to-use configs are provided. Both train with `global_batch_size=64`, so they are numerically equivalent — they only differ in how the batch is laid out across GPUs:

| Config | Intended setup | `local_batch_size` | Activation ckpt. |
| --- | --- | ---: | --- |
| [nemotron_vl_1b_optimized_ddp_lbs2.yaml](nemotron_vl_1b_optimized_ddp_lbs2.yaml) | multi-node, e.g. 4 nodes × 8 GPUs | 2 | off |
| [nemotron_vl_1b_optimized_ddp_1node.yaml](nemotron_vl_1b_optimized_ddp_1node.yaml) | single 8-GPU node | 8 | vision tower |

On a single node, launch with `torchrun` (both runs are memory-tight on 80GB GPUs, so set the allocator option first):

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/nemotron_vl_1b/nemotron_vl_1b_optimized_ddp_1node.yaml
```

Note that `torchrun --nproc-per-node` launches a single node. For multi-node runs, submit through your cluster scheduler (e.g. `sbatch` on Slurm), as with the base example above.

### Choosing a GPU budget

None of these GPU counts is a requirement. The configs keep `global_batch_size=64` fixed, and you can reproduce the same training on almost any GPU budget by combining three levers:

1. **GPU count** — more GPUs shortens wall-clock time; it does not change the result.
2. **Gradient accumulation** — happens automatically: whenever `local_batch_size × GPUs` is smaller than `global_batch_size`, the recipe accumulates over `global / (local × GPUs)` micro-batches per optimizer step. For example, running the `lbs2` config unchanged on a single 8-GPU node simply accumulates over 4 micro-batches (~70GiB peak, no other changes needed).
3. **Activation checkpointing** — trades recompute for memory so a larger `local_batch_size` fits, avoiding accumulation entirely. Enable it under `distributed:` and optionally scope it (`activation_checkpointing_scope: vision` checkpoints only the vision tower). See [Use Gradient (Activation) Checkpointing](https://docs.nvidia.com/nemo/automodel/latest/guides/gradient-checkpointing.html).

Measured reference points on 80GB GPUs (262k-sample VL retrieval workload, no gradient accumulation in any row):

| GPUs | local/global batch | Activation ckpt. | Samples/s/GPU | Peak mem/GPU | Approx. epoch |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 (8 nodes) | 1/64 | no | ~1.95 | ~50GiB | ~35m |
| 32 (4 nodes) | 2/64 | no | ~2.20 | ~70GiB | ~62m |
| 8 (1 node) | 8/64 | vision tower only | ~2.11 | ~80GiB | ~4.3h |
| 4 | 16/64 | full model | ~2.05 | ~61GiB | ~8.9h |

Per-GPU efficiency peaks at the 32-GPU `local_batch_size=2` configuration, and the single-node setups stay close to it (within ~5-7%) even with activation-checkpointing recompute — all three are meaningfully more GPU-hour-efficient than the 64-GPU `local_batch_size=1` run. In other words, using fewer GPUs costs wall-clock time but almost no total GPU-hours.

For example, to run on 4 GPUs, checkpoint the full model and raise `local_batch_size` to 16:

```yaml
step_scheduler:
  global_batch_size: 64
  local_batch_size: 16

distributed:
  strategy: ddp
  activation_checkpointing: true
  activation_checkpointing_scope: all
```
