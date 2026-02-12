# Automatic Speech Recognition (ASR) Models

## Introduction

Automatic Speech Recognition (ASR) models convert spoken language into written text. NeMo AutoModel provides a simple interface for loading and fine-tuning ASR models hosted on the Hugging Face Hub, supporting both encoder-decoder (Seq2Seq) and encoder-only (CTC) architectures.

## Run ASR Models with NeMo AutoModel

To run ASR models with NeMo AutoModel, use NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you want to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest NeMo AutoModel using:

```bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/AutoModel.git
```

For other installation options (e.g., uv) please see our [Installation Guide](../guides/installation.md).

### System Dependencies

ASR requires FFmpeg libraries for audio decoding. Install them based on your OS:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

**macOS:**
```bash
brew install ffmpeg
```

**Note**: If using the Docker container, these dependencies are already included.

## Supported Models

NeMo AutoModel supports two Auto classes for ASR:
- **`AutoModelForCTC`** - CTC-based encoder-only models (Parakeet)
- **`AutoModelForSpeechSeq2Seq`** - Encoder-decoder models with CrossEntropy loss (Whisper)

### Parakeet CTC Models (NVIDIA)

Parakeet models use CTC (Connectionist Temporal Classification) loss with encoder-only Conformer architecture for efficient speech recognition.

| Model | Parameters | Architecture | Example YAML |
|-------|-----------|--------------|--------------|
| nvidia/parakeet-ctc-0.6b | 600M | Encoder-only CTC | [parakeet_ctc_0.6b_librispeech.yaml](../../examples/asr_finetune/parakeet/parakeet_ctc_0.6b_librispeech.yaml), [parakeet_ctc_0.6b_librispeech_peft.yaml](../../examples/asr_finetune/parakeet/parakeet_ctc_0.6b_librispeech_peft.yaml) |
| nvidia/parakeet-ctc-1.1b | 1.1B | Encoder-only CTC | [parakeet_ctc_1.1b_librispeech.yaml](../../examples/asr_finetune/parakeet/parakeet_ctc_1.1b_librispeech.yaml), [parakeet_ctc_1.1b_librispeech_peft.yaml](../../examples/asr_finetune/parakeet/parakeet_ctc_1.1b_librispeech_peft.yaml) |

### Whisper Models (OpenAI)

Whisper models use encoder-decoder architecture with CrossEntropy loss and support multilingual transcription and translation across 99 languages.

| Model | Parameters | Languages | Architecture | Example YAML |
|-------|-----------|-----------|--------------|--------------|
| openai/whisper-tiny | 39M | 99 | Encoder-Decoder Seq2Seq | - |
| openai/whisper-base | 74M | 99 | Encoder-Decoder Seq2Seq | - |
| openai/whisper-small | 244M | 99 | Encoder-Decoder Seq2Seq | [whisper_small_librispeech.yaml](../../examples/asr_finetune/whisper/whisper_small_librispeech.yaml), [whisper_small_librispeech_peft.yaml](../../examples/asr_finetune/whisper/whisper_small_librispeech_peft.yaml) |
| openai/whisper-medium | 769M | 99 | Encoder-Decoder Seq2Seq | [whisper_medium_librispeech.yaml](../../examples/asr_finetune/whisper/whisper_medium_librispeech.yaml), [whisper_medium_librispeech_peft.yaml](../../examples/asr_finetune/whisper/whisper_medium_librispeech_peft.yaml) |
| openai/whisper-large-v3 | 1.55B | 99 | Encoder-Decoder Seq2Seq | - |

## Fine-Tuning ASR Models with NeMo AutoModel

The models listed above can be fine-tuned using NeMo AutoModel to adapt them to specific domains or acoustic conditions. We support two primary fine-tuning approaches:

1. **Supervised Fine-Tuning (SFT)**: Updates all model parameters for deeper adaptation to your audio domain and vocabulary. Suitable for high-precision applications where you have sufficient training data.

2. **Parameter-Efficient Fine-Tuning (PEFT)**: Updates only a small subset of parameters (typically <1%) using techniques like Low-Rank Adaptation (LoRA). This provides 40-60% memory reduction compared to full fine-tuning, making it ideal for resource-constrained environments. PEFT typically uses 5-10x higher learning rates and produces 10-30x smaller checkpoints (15-50MB vs 500MB-1.5GB).

For detailed instructions and examples, see `examples/asr_finetune/` with comprehensive configs for all models.

:::{tip}
In these guides, we use the `LibriSpeech` dataset for demonstration purposes, but you can use your own audio data.

To do so, update the recipe YAML `dataset` / `validation_dataset` sections (for example `dataset._target_`, `path_or_dataset`, and `split`). See [ASR datasets](../guides/asr/dataset.md) and [dataset overview](../guides/dataset-overview.md).
:::
