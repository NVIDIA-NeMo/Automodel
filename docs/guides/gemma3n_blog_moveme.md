# 🚀 NeMo Automodel Now Supports Google Gemma 3n: Efficient Multimodal Fine-tuning Made Simple

## Introduction

These are **multimodal models** capable of understanding images and audio, and they are optimized to run efficiently on everyday devices. They introduce innovations such as Per-Layer Embedding parameter caching and the MatFormer architecture, which help reduce compute and memory demands.

Gemma 3n is designed for efficient resource usage and fast inference, and offers full multimodal support with both vision and audio inputs. Some key highlights:

- **Optimized architecture** featuring MatFormer's nested transformers, per-layer embeddings, and KV cache sharing. These enable sub-model extraction, reduced GPU memory usage, and faster prefill speeds.
- **Multimodal capabilities** with integrated image and audio encoders alongside the language model, enabling diverse tasks across modalities.  
- Pretrained checkpoints are available under the [Gemma 3N releases on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4).

Today, we are excited to announce that NeMo Automodel now supports Gemma 3n, making it easier than ever to load, train, and inference with Gemma 3n models.


---

## ⚡ Fine-tuning Gemma 3n with NeMo Automodel

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) offers a unified interface to load and finetune diverse models across modalities, abstracting away backend complexity. With Gemma 3N support:

- Load models with a single `from_pretrained` call  
- Finetune models using full parameter training or PEFT(LoRA) with predefined recipes
- Accelerate training with kernel optimizations
- Leverage FSDP2/nvFSDP for efficient distributed training

Check out our tutorial for SFT/PEFT for Gemma3 and Gemma3n model!

## 🔍 Observations

### Accuracy
We observed **lower overall accuracy on vision and audio tasks** with the pre-trained HuggingFace checkpoints. The model appears to underperform compared to expectations for general multimodal understanding tasks.

### Training Dynamics
We noticed **large gradients during initial training steps** that quickly stabilize within the first hundred steps, but overall training convergence remain worse than Gemma 3.

This appears to be due to:

- **Shared KV Cache Issues**: Gemma 3n checkpoints contain **empty (all-zero) weights** for KV cache shared layers, causing instability during early training phases when cache is not enabled during training
- **Vision Component Weakness**: As our benchmarking suggests, the vision component in Gemma 3n could limit the overall model performance
- **Architecture Misalignment**: The optimization-focused design may sacrifice some capability for efficiency, resulting in reduced performance compared to Gemma 3



## ✨ Conclusion
Gemma 3n brings impressive efficiency and opens up new possibilities for multimodal tasks on devices. With NeMo Automodel, getting started requires only a few commands!

We look forward to seeing what you build with Gemma 3n and NeMo Automodel. Check out the documentation guide for a full walkthrough, and reach out on GitHub Discussions if you have questions.

## 🔗 References
[Gemma 3n on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)

NeMo Automodel Gemma 3 Guide