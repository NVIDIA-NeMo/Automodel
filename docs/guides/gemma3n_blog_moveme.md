# 🚀 NeMo Automodel Now Supports Google Gemma 3N: Efficient Multimodal Fine-tuning Made Simple

## Introduction

Large Language Models are evolving rapidly, and Google's Gemma 3N is a prime example. As a next-generation multimodal model, Gemma 3N brings both architectural optimizations and image understanding capabilities to the table.

Today, we are excited to announce that NeMo Automodel now supports Gemma 3N, making it easier than ever to load, train, and inference with Gemma 3N models.

---

## 🌟 What is Gemma 3N?

Gemma 3N is part of Google's Gemma family, optimized for both **performance and multimodality**. Some key highlights:

- **Optimized architecture** featuring MatFormer’s nested transformers, per-layer embeddings, and KV cache sharing, enabling sub-model extraction, reduced GPU memory usage, and fast prefill speed
- **Multimodal capabilities**, incorporating image encoder and audio encoder alongside its language model, enabling various multimodal tasks.  
- Pretrained checkpoints are available under the [Gemma 3N releases on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4).


---

## ⚡ Fine-tuning Gemma 3N with NeMo Automodel

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) offers a unified interface to load and finetune diverse models across modalities, abstracting away backend complexity. With Gemma 3N support:

- Load models with a single `from_pretrained` call  
- Finetune models with PEFT techniques (LoRA, DoRA) or full parameter training with predefined recipes
- Accelerate training with kernel optimizations
- Leverage FSDP2/nvFSDP for efficient distributed training


Check out our notebook for SFT and PEFT Gemma3n model!

## 🔍 Observations

### Performance Limitations
We observed **lower overall performance on vision and audio tasks** with the pre-trained HuggingFace checkpoints. The model appears to underperform compared to expectations for general multimodal understanding tasks.

### Training Dynamics
We noticed **large gradients during initial training steps** that quickly stabilize within the first hundred steps, but overall training convergence remain worse than Gemma 3.

This appears to be due to:

- **Shared KV Cache Issues**: Gemma 3N checkpoints contain **empty (all-zero) weights** for KV cache shared layers, causing instability during early training phases when cache is not enabled during training
- **Vision Component Weakness**: As our benchmarking suggests, the vision component in Gemma 3N could limit the overall model performance
- **Architecture Misalignment**: The optimization-focused design may sacrifice some capability for efficiency, resulting in reduced performance compared to Gemma 3



## ✨ Conclusion
Gemma 3N brings impressive efficiency and opens up new possibilities for multimodal tasks. With NeMo Automodel, getting started requires only a few commands!

We look forward to seeing what you build with Gemma 3N and NeMo Automodel. Check out the documentation guide for a full walkthrough, and reach out on GitHub Discussions if you have questions.

## 🔗 References
[Gemma 3N on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)

NeMo Automodel Gemma 3 Guide