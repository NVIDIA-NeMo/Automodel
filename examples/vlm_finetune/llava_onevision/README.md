# LLaVA-OneVision-1.5 in NeMo AutoModel

LLaVA-OneVision-1.5 is a vision-language model combining a **Rice ViT** vision encoder with a **Qwen3** language model, capable of handling both image and video understanding tasks. This implementation integrates LLaVA-OneVision-1.5 into NVIDIA NeMo AutoModel's training pipeline with FSDP2/HSDP support, LoRA fine-tuning, and distributed training.

## Architecture

```
LLaVA-OneVision-1.5
├── Vision Tower: Rice Transformer (ViT)
│   ├── Patch Embed: 14x14 patches, 2D RoPE
│   ├── Transformer Blocks: LayerNorm + Attention + MLP
│   └── Patch Merger: 2x2 spatial merge → text hidden size
├── Projector: Integrated in PatchMerger (MLP with GELU)
└── Language Model: Qwen3 (4B or 8B variants)
```

### Key Specifications

| Component | 4B Model | 8B Model |
|-----------|----------|----------|
| Vision Encoder Depth | 24 layers | 24 layers |
| Vision Hidden Size | 1024 | 1024 |
| Text Hidden Size | 2560 | 4096 |
| Text Layers | 36 | 48 |
| Patch Size | 14x14 | 14x14 |
| Spatial Merge Size | 2x2 | 2x2 |
| Attention | GQA | GQA |

## Quick Start

### Fine-tune 4B Model (Full)

```bash
automodel examples/vlm_finetune/llava_onevision/llava_ov_1_5_4b_finetune.yaml --nproc-per-node 8
```

### Fine-tune 8B Model (LoRA)

```bash
automodel examples/vlm_finetune/llava_onevision/llava_ov_1_5_8b_lora.yaml --nproc-per-node 8
```

### Override Model Path

```bash
automodel examples/vlm_finetune/llava_onevision/llava_ov_1_5_4b_finetune.yaml \
  --model.pretrained_model_name_or_path /path/to/local/weights \
  --nproc-per-node 8
```

## Dataset Preparation

The default dataset is `liuhaotian/LLaVA-Instruct-150K`. To use a custom dataset:

1. Edit the YAML config:
   ```yaml
   dataset:
     _target_: nemo_automodel.components.datasets.vlm.datasets.make_llava_onevision_dataset
     path_or_dataset: your/dataset-name
     split: train
   ```

2. Ensure your dataset follows the conversation format:
   ```json
   {
     "conversations": [
       {"from": "human", "value": "<image>\nDescribe this image."},
       {"from": "gpt", "value": "This is a description..."}
     ],
     "image": <PIL.Image or path>
   }
   ```

## Configuration Options

### Key YAML Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.pretrained_model_name_or_path` | Model checkpoint path | `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct` |
| `processor.pretrained_model_name_or_path` | Processor path | Same as model |
| `dataset.path_or_dataset` | Dataset name/path | `liuhaotian/LLaVA-Instruct-150K` |
| `dataset.split` | Dataset split | `train[:1000]` |
| `optimizer.lr` | Learning rate | `2e-5` (4B), `1e-4` (LoRA) |
| `freeze_config.freeze_vision_tower` | Freeze vision encoder | `true` |
| `freeze_config.freeze_language_model` | Freeze LLM | `false` (4B), `true` (LoRA) |

### LoRA Configuration

The 8B config uses LoRA with these defaults:
```yaml
peft:
  enabled: true
  method: lora
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

## Hardware Requirements

| Configuration | Minimum VRAM | Recommended |
|---------------|--------------|-------------|
| 4B Full Fine-tune | 80GB (A100/H100) | 8× A100 80GB |
| 8B LoRA | 40GB (A100) | 8× A100 40GB |

**Note:** These are estimates. Actual memory usage depends on batch size, sequence length, and gradient checkpointing settings.

## Implementation Details

### Files Added

```
nemo_automodel/
├── components/models/llava_onevision/
│   ├── __init__.py
│   ├── model.py                    # Main model class
│   ├── rice_vit.py                 # Rice ViT implementation
│   └── state_dict_adapter.py       # HF checkpoint adapter
├── components/datasets/vlm/
│   ├── collate_fns.py              # Added llava_onevision_collate_fn
│   └── datasets.py                 # Added make_llava_onevision_dataset
├── _transformers/
│   └── registry.py                 # Added LlavaOneVisionForConditionalGeneration
examples/
└── vlm_finetune/llava_onevision/
    ├── llava_ov_1_5_4b_finetune.yaml
    └── llava_ov_1_5_8b_lora.yaml
```

### Token Protocol

LLaVA-OneVision-1.5 uses special tokens for multimodal content:

| Token | ID | Purpose |
|-------|-----|---------|
| `<|image_pad|>` | 151655 | Image placeholder |
| `<|image_pad|>` | 151656 | Video placeholder |
| `<think>` | 151652 | Vision start |
| `</think>` | 151653 | Vision end |

The processor handles token placement automatically via the chat template.

## Troubleshooting

### Out of Memory

- Reduce `local_batch_size` in `step_scheduler`
- Enable gradient checkpointing (if not already)
- For 8B model, use LoRA config instead of full fine-tuning

### Dataset Loading Errors

- Verify dataset format matches conversation structure
- Check that image paths or PIL Images are correctly provided
- For HuggingFace datasets, ensure you have internet access or use local path

### Processor Not Found

- Ensure `trust_remote_code: true` is set in processor config
- Verify model path is correct and accessible

## References

- [LLaVA-OneVision Original Repo](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2)
- [NeMo AutoModel Documentation](https://github.com/NVIDIA-NeMo/Automodel)
- [Qwen3 Documentation](https://qwenlm.github.io/blog/qwen3/)

## License

This implementation is licensed under the Apache License 2.0. See the main repository LICENSE file for details.
