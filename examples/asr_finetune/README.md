# ASR Fine-tuning Examples

Examples for fine-tuning ASR (Automatic Speech Recognition) models with NeMo AutoModel.

## Supported Models

### Whisper (OpenAI)
- **openai/whisper-tiny** (39M params) - Fast, lower accuracy
- **openai/whisper-base** (74M params) - Balanced speed/accuracy
- **openai/whisper-small** (244M params) - Good accuracy
- **openai/whisper-medium** (769M params) - High accuracy
- **openai/whisper-large-v3** (1.55B params) - Best accuracy, 99 languages

Whisper models use encoder-decoder architecture with CrossEntropy loss and support multilingual transcription and translation.

## Supported Datasets

### LibriSpeech (Recommended)
- **librispeech_asr** - 1000 hours of English audiobooks
- High-quality recordings with accurate transcriptions
- Splits: train.100, train.clean.360, train.other.500, test, test.other
- Readily available on HuggingFace

### Common Voice (Relocated)
- **mozilla-foundation/common_voice_17_0** - 100+ languages, community-contributed recordings
- ⚠️ **Note**: As of October 2025, Mozilla Common Voice datasets are exclusively available through [Mozilla Data Collective](https://datacollective.mozillafoundation.org)
- The HuggingFace repo is now empty - use Mozilla Data Collective to download
- Good for multilingual fine-tuning once downloaded

### Custom Datasets
Use `make_custom_asr_dataset` to load any HuggingFace audio dataset with audio and text fields.

## Installation

```bash
# Install ASR dependencies
uv sync --extra asr

# Or install all extras including ASR
uv sync --all-extras
```

## Quick Start

### Single GPU Training

```bash
# Whisper Small on LibriSpeech (100h clean English)
uv run examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_small_librispeech.yaml

# Whisper Medium on LibriSpeech (full dataset)
uv run examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_medium_librispeech.yaml
```

### Multi-GPU Training (Data Parallel)

```bash
# 8 GPUs with data parallelism
uv run torchrun --nproc-per-node=8 examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_small_librispeech.yaml
```

### Multi-GPU with Tensor Parallelism

```bash
# Whisper Medium with TP=2, DP=4 (requires 8 GPUs)
uv run torchrun --nproc-per-node=8 examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_medium_librispeech.yaml \
  --distributed.tp_size 2
```

### Using the automodel CLI

```bash
# Single node, 8 GPUs
uv run automodel finetune asr --nproc-per-node=8 \
  -c examples/asr_finetune/whisper/whisper_small_librispeech.yaml

# Multi-node SLURM (see CLAUDE.md for SLURM configuration)
uv run automodel finetune asr \
  -c examples/asr_finetune/whisper/whisper_medium_librispeech.yaml \
  --slurm.nodes 4 \
  --slurm.gpus_per_node 8
```

## Attention Implementations

The example configs use **SDPA** (Scaled Dot Product Attention) by default, which is PyTorch-native and requires no extra dependencies.

### Using Flash Attention 2 (Optional)

For better memory efficiency and speed, you can use Flash Attention 2:

```bash
# Install flash attention
uv sync --extra fa --extra asr

# Use flash attention (override config)
uv run examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_small_librispeech.yaml \
  --model.attn_implementation flash_attention_2
```

**Performance comparison:**
- **SDPA**: Good performance, no installation required, works everywhere
- **Flash Attention 2**: Best performance, requires compilation, GPU-specific

For most use cases, SDPA provides excellent performance without the installation complexity.

## Configuration

### Override Config Values via CLI

```bash
uv run examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/whisper/whisper_small_librispeech.yaml \
  --model.pretrained_model_name_or_path openai/whisper-base \
  --step_scheduler.max_steps 2000 \
  --optimizer.lr 5e-6 \
  --dataset.split train.clean.360
```

### Key Configuration Sections

#### Model
```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForSpeechSeq2Seq.from_pretrained
  pretrained_model_name_or_path: openai/whisper-small
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
```

#### Dataset
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: train.100  # Options: train.100, train.clean.360, train.other.500
  streaming: false
  limit_dataset_samples: 10000  # For quick testing
```

#### Distributed Training
```yaml
distributed:
  dp_size: null  # Auto-calculated from available GPUs
  tp_size: 1     # Tensor parallelism
  cp_size: 1     # Context parallelism

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config
  sequence_parallel: false  # Enable for large models
```

## Advanced Features

### PEFT (Parameter-Efficient Fine-Tuning)

Train only adapter layers instead of full model:

```yaml
peft:
  _target_: nemo_automodel.components._peft.PeftConfig
  method: lora
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
```

### FP8 Quantization

Enable FP8 training for memory efficiency:

```yaml
quantization:
  enable_fp8: true
```

### Pipeline Parallelism

For very large models across multiple GPUs:

```yaml
distributed:
  pp_size: 2

autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.config.PipelineConfig
  pp_microbatch_size: 1
  schedule: "1f1b"
```

## SPMD Principle

The same training script scales from 1 GPU to 1000+ GPUs by changing only the configuration:

```bash
# 1 GPU
python finetune.py --config config.yaml

# 8 GPUs (data parallel)
torchrun --nproc-per-node=8 finetune.py --config config.yaml

# 8 GPUs (tensor parallel)
torchrun --nproc-per-node=8 finetune.py --config config.yaml --distributed.tp_size 2

# 32 GPUs across 4 nodes (SLURM)
automodel finetune asr -c config.yaml --slurm.nodes 4 --slurm.gpus_per_node 8
```

No code changes required!

## Checkpointing

Checkpoints are saved in SafeTensors format:

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: ./asr_checkpoints/whisper_small
  model_save_format: safetensors
  save_consolidated: true
```

Resume training:

```bash
uv run examples/asr_finetune/finetune.py \
  --config config.yaml \
  --checkpoint.restore_from ./asr_checkpoints/whisper_small/step-500
```

## Logging

### Weights & Biases

```yaml
wandb:
  project_name: asr-finetuning
  run_name: whisper-small-cv-en
  entity: your-team
```

### Local JSONL Logs

Training and validation metrics are logged to:
- `training.jsonl` - Training loss, learning rate, tokens/sec
- `validation.jsonl` - Validation loss per checkpoint

## Troubleshooting

### Out of Memory
- Reduce `local_batch_size`
- Enable `sequence_parallel: true` in MegatronFSDPConfig
- Use smaller model (whisper-small instead of whisper-medium)
- Enable gradient checkpointing (added in model config)

### Slow Training
- Increase `num_workers` in dataloader
- Use `streaming: true` for very large datasets
- Try Flash Attention 2 (optional, requires `uv sync --extra fa`):
  ```bash
  --model.attn_implementation flash_attention_2
  ```
  Note: Examples use SDPA by default which provides good performance without extra dependencies

### Flash Attention Issues
- If you get "flash_attn not installed" error, either:
  - Install it: `uv sync --extra fa --extra asr`
  - Or use SDPA (default): `--model.attn_implementation sdpa`
- Flash Attention requires CUDA-compatible GPU and compilation time

### Dataset Issues
- **Common Voice**: As of October 2025, Mozilla Common Voice is no longer available on HuggingFace. Download from [Mozilla Data Collective](https://datacollective.mozillafoundation.org) instead.
- **LibriSpeech**: Readily available on HuggingFace, no special authentication required
- Use `limit_dataset_samples` for quick debugging
- Check audio sampling rate is 16kHz (Whisper requirement)

## Examples Overview

| Config | Model | Dataset | GPUs | Batch Size | Steps | Notes |
|--------|-------|---------|------|------------|-------|-------|
| whisper_small_librispeech.yaml | Whisper Small (244M) | LibriSpeech 100h | 1-8 | 32 | 1000 | Quick start, clean English |
| whisper_medium_librispeech.yaml | Whisper Medium (769M) | LibriSpeech Full | 8+ | 64 | 5000 | TP=2, production quality |

## Next Steps

- **More Data**: Use `train.clean.360` or `train.other.500` splits for more training data
- **Multilingual Training**: Download Common Voice from Mozilla Data Collective or use other multilingual datasets
- **Translation**: Whisper supports translation tasks (use appropriate prompts)
- **Custom Data**: Use `make_custom_asr_dataset` for your own audio datasets
- **Evaluation**: Add WER (Word Error Rate) calculation in validation loop
- **Production Deployment**: Export to ONNX or use HuggingFace inference

## Resources

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Common Voice Dataset](https://commonvoice.mozilla.org/)
- [LibriSpeech Dataset](https://www.openslr.org/12)
- [NeMo AutoModel Documentation](https://docs.nvidia.com/deeplearning/nemo/)
