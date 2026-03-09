(diffusion-dataset)=

# Diffusion Dataset Preparation

## Introduction

Diffusion model training in NeMo AutoModel requires pre-encoded `.meta` files rather than raw images or videos. During preprocessing, a VAE encodes visual data into latent representations and a text encoder produces text embeddings. These are saved as `.meta` files so that training operates entirely in latent space, avoiding the need to load heavy encoder models during training.

## Data Format

### Directory Structure

Organize your raw data as a folder of videos (or images) with a `meta.json` manifest:

```
<your_video_folder>/
├── video1.mp4
├── video2.mp4
└── meta.json
```

### meta.json Schema

The `meta.json` file is a JSON array describing each sample:

```json
[
  {
    "file_name": "video1.mp4",
    "width": 1280,
    "height": 720,
    "start_frame": 0,
    "end_frame": 121,
    "vila_caption": "A detailed description of the video content..."
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `file_name` | string | Filename of the video or image |
| `width` | int | Original width in pixels |
| `height` | int | Original height in pixels |
| `start_frame` | int | First frame index (video only) |
| `end_frame` | int | Last frame index (video only) |
| `vila_caption` | string | Text caption or prompt describing the content |

## Preprocessing

:::{important}
Data preprocessing (`prepare_dataset_wan.py`) lives in the DFM repository and must be run separately to produce `.meta` files before training.
:::

### Mode 1: Full Video (Recommended for Training)

Encodes the complete video into latent representations:

```bash
python prepare_dataset_wan.py \
  --video_folder <your_video_folder> \
  --output_folder ./processed_meta \
  --output_format automodel \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode video \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop
```

### Mode 2: Frame Extraction (for Frame-Based Training)

Extracts evenly-spaced frames from each video:

```bash
python prepare_dataset_wan.py \
  --video_folder <your_video_folder> \
  --output_folder ./processed_meta \
  --output_format automodel \
  --model "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --mode frames \
  --num-frames 40 \
  --height 480 \
  --width 832 \
  --resize_mode bilinear \
  --center-crop
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--mode` | `video` (full video) or `frames` (extract evenly-spaced frames) |
| `--num-frames` | Number of frames to extract (only for `frames` mode) |
| `--height` / `--width` | Target resolution for the encoded latents |
| `--center-crop` | Crop to exact size after aspect-preserving resize |
| `--output_format` | Use `automodel` for NeMo AutoModel compatibility |
| `--model` | Hugging Face model ID, used to select the correct VAE and text encoder |

## .meta File Contents

Each `.meta` file produced by preprocessing contains:

- **Encoded video latents** — Normalized VAE latent representations
- **Text embeddings** — Pre-computed from UMT5 (or model-appropriate text encoder)
- **First frame as JPEG** — Reference image (video mode only)
- **Metadata** — Resolution, frame count, and other sample properties

## Multiresolution Bucketing

NeMo AutoModel supports multiresolution training through bucketed sampling. This groups samples by their spatial resolution so that each batch contains samples of the same size, avoiding padding waste.

Key configuration parameters:

- `base_resolution`: The target resolution used for bucketing (e.g., `[512, 512]`)
- The `SequentialBucketSampler` groups samples by resolution bucket
- `dynamic_batch_size`: When `true`, adjusts batch size per resolution bucket to maintain constant memory usage

## YAML Configuration

### Video Dataloader (Wan 2.1 / HunyuanVideo)

Used for text-to-video models. Set `model_type` to match your model (`wan` or `hunyuan`):

```yaml
data:
  dataloader:
    _target_: nemo_automodel.components.datasets.diffusion.build_video_multiresolution_dataloader
    cache_dir: /path/to/processed_meta
    model_type: wan          # or "hunyuan"
    base_resolution: [512, 512]
    dynamic_batch_size: false
    shuffle: true
    drop_last: false
    num_workers: 0
```

### Image Dataloader (FLUX)

Used for text-to-image models:

```yaml
data:
  dataloader:
    _target_: nemo_automodel.components.datasets.diffusion.build_text_to_image_multiresolution_dataloader
    cache_dir: /path/to/processed_meta
    train_text_encoder: false
    num_workers: 0
    base_resolution: [512, 512]
    dynamic_batch_size: false
    shuffle: true
    drop_last: false
```

:::{tip}
Supported image resolutions for FLUX include `[256, 256]`, `[512, 512]`, and `[1024, 1024]`. While a 1:1 aspect ratio is currently used as a proxy for the closest image size, the implementation is designed to support multiple aspect ratios.
:::
