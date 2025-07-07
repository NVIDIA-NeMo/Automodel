# 🚀 NeMo Framework Now Supports Google Gemma 3n: Efficient Multimodal Fine-tuning Made Simple

## Introduction

Gemma 3n is a generative AI model that takes inputs from a variety modalities, including images and audio, and is optimized for efficient resource usage and fast inference on everyday devices. It introduces innovations such as Per-Layer Embedding parameter caching and the MatFormer architecture, which help reduce compute and memory demands. Some key highlights:

- **Optimized architecture** featuring MatFormer's nested transformers, per-layer embeddings, and KV cache sharing. These enable sub-model extraction, reduced GPU memory usage, and faster prefill speeds.
- **Multimodal capabilities** with integrated image and audio encoders alongside the language model, enabling diverse tasks across modalities.  
- Pretrained checkpoints are available under the [Gemma 3n releases on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4).

Today, we are excited to announce that NeMo Automodel now supports Gemma 3n, making it easier than ever to load, train, and inference with Gemma 3n models.


---

## ⚡ Fine-tuning Gemma 3n with NeMo Automodel

[NeMo Framework's Automodel path ("Nemo AutoModel")](https://github.com/NVIDIA-NeMo/Automodel) offers day-0 support for :hugs:Hugging Face models via a unified interface to load and finetune models across modalities, abstracting away backend complexity. With Gemma 3n support:

- Load models with a single `from_pretrained` call  
- Finetune models using full parameter training or PEFT(LoRA) with predefined recipes
- Accelerate training with kernel optimizations
- Leverage FSDP2/nvFSDP for efficient distributed training

Check out our [tutorial]() on SFT and PEFT for both Gemma 3 and Gemma 3n models!

## 🔍 Observations

### Accuracy
We observed **lower overall accuracy on vision and audio tasks** with the pre-trained HuggingFace checkpoints. The model appears to underperform compared to expectations for general multimodal understanding tasks.

### Training Dynamics
We noticed **large gradients during initial training steps** that quickly stabilize within the first hundred steps, but overall training convergence remain worse than Gemma 3.

<img src="gemma3_3n_trainloss.png" alt="Training Loss Curve" width="400">

This appears to be due to:

- **Shared KV Cache Issues**: Gemma 3n checkpoints contain **empty (all-zero) weights** for KV cache shared layers, causing instability during early training phases when cache is not enabled in training.
- **Vision Component Weakness**: As our benchmarking suggests, the vision component in Gemma 3n could limit the overall model performance.
- **Architecture**: The focus on fast and resource-efficient inference may sacrifice some accuracies, making it worse compared to Gemma 3.



## ✨ Conclusion
Gemma 3n brings impressive efficiency and opens up new possibilities for multimodal tasks on devices. With NeMo Automodel, getting started requires only a few commands!

We look forward to seeing what you build with Gemma 3n and NeMo Automodel. Check out the documentation guide for a full walkthrough, and reach out on GitHub Discussions if you have questions.

## 🔗 References
[Gemma 3n on Hugging Face](https://huggingface.co/collections/google/gemma-3n-685065323f5984ef315c93f4)

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel)

[NeMo Automodel Gemma 3 Guide]()
