# Llama-Embed-Nemotron-8B Training Pipeline

## Overview

[llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) is a versatile text embedding model trained by NVIDIA and optimized for retrieval, reranking, semantic similarity, and classification use cases. This model has robust capabilities for multilingual and cross-lingual text retrieval and is designed to serve as a foundational component in text-based Retrieval-Augmented Generation (RAG) systems. This model achieves state-of-the-art performance on the multilingual [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard as of October 21, 2025.

This guide provides step-by-step instructions to reproduce the training pipeline for the `llama-embed-nemotron-8b` model using [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) framework.

## Reproduction Steps

### 1. Download and Prepare the Dataset

Download and prepare the `nvidia/embed-nemotron-dataset-v1` dataset from [Hugging Face](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1). This dataset is a selected subset of the fine-tuning data used for training the `llama-embed-nemotron-8b` model:

```python
python examples/biencoder/llama_embed_nemotron_8b/data_preparation.py \
    --download-path ./embed_nemotron_dataset_v1
```

This script will download the dataset and prepare it for training. 

### 2. Run Model Finetuning

Run the model finetuning with the specified configuration using 8 GPUs:

```bash
torchrun --nproc-per-node=8 examples/biencoder/finetune.py \
    --config examples/biencoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml
```

The final model checkpoint in Hugging Face format will be stored in `output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated`

## Citation

If you use this model or training pipeline in your research, please cite:

```bibtex
@misc{babakhin2025llamaembednemotron8buniversaltextembedding,
      title={Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks}, 
      author={Yauhen Babakhin and Radek Osmulski and Ronay Ak and Gabriel Moreira and Mengyao Xu and Benedikt Schifferer and Bo Liu and Even Oldridge},
      year={2025},
      eprint={2511.07025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07025}, 
}
```
