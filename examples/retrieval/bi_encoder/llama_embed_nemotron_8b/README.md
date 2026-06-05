# Llama-Embed-Nemotron-8B Training Pipeline

## Overview

[llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b) is a versatile text embedding model trained by NVIDIA and optimized for retrieval, reranking, semantic similarity, and classification use cases. This model has robust capabilities for multilingual and cross-lingual text retrieval and is designed to serve as a foundational component in text-based Retrieval-Augmented Generation (RAG) systems. This model achieves state-of-the-art performance on the multilingual [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard as of October 21, 2025.

This guide provides step-by-step instructions to reproduce the training pipeline for the `llama-embed-nemotron-8b` model using the [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) framework.

## Run the Training Pipeline

### Download and Prepare the Dataset

Download and prepare the `nvidia/embed-nemotron-dataset-v1` dataset from [Hugging Face](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1). This dataset is a selected subset of the fine-tuning data used for training the `llama-embed-nemotron-8b` model:

```bash
uv run python examples/retrieval/bi_encoder/llama_embed_nemotron_8b/data_preparation.py \
    --download-path ./embed_nemotron_dataset_v1
```

Run this command from the repository root, or update the relative paths in the YAML. This script downloads the dataset and prepares it for training.


### Fine-Tune the Model

Fine-tune the model on 8 GPUs using the example YAML configuration:

```bash
uv run automodel examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml --nproc-per-node 8
```

The final model checkpoint in Hugging Face format is stored in `output/llama_embed_nemotron_8b/epoch_0_step_28614/model/consolidated`.

## Citation

If you use this model or training pipeline in your research, cite:

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
