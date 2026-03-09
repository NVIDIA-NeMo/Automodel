(diffusion-dataset)=

# Diffusion Dataset Preparation

## Introduction

Diffusion model training in NeMo AutoModel requires pre-encoded `.meta` files rather than raw images or videos. During preprocessing, a VAE encodes visual data into latent representations and a text encoder produces text embeddings. These are saved as `.meta` files so that training operates entirely in latent space, avoiding the need to load heavy encoder models during training.

## Input Data Format

### Images

Place your images in a directory. Supported formats: `jpg`, `jpeg`, `png`, `webp`, `bmp`.

Captions can be provided in several formats:
- **Sidecar JSON** (default for images): A `.json` file alongside each image with a caption field
- **JSONL**: A `.jsonl` file with `internvl` or `usr` caption fields

### Videos

Place your videos in a directory. Supported formats: `mp4`, `avi`, `mov`, `mkv`, `webm`.

Captions can be provided in several formats:
- **Sidecar JSON** (`--caption_format sidecar`): A `.json` file alongside each video
- **meta.json** (`--caption_format meta_json`): A single `meta.json` manifest in the video directory
- **JSONL** (`--caption_format jsonl`): A `.jsonl` file with captions

If no caption is found for a sample, the filename (with underscores replaced by spaces) is used as a fallback.

## Preprocessing

NeMo AutoModel includes a unified preprocessing tool at [`tools/diffusion/preprocessing_multiprocess.py`](../../../tools/diffusion/preprocessing_multiprocess.py) that encodes raw images and videos into cache files compatible with the multiresolution dataloader. It uses model-specific processors from [`tools/diffusion/processors/`](../../../tools/diffusion/processors/) to handle VAE encoding, text embedding, and cache data formatting for each supported model.

The tool automatically distributes work across all available GPUs using multiprocessing, with one worker per GPU.

### Available Processors

| Processor | Media Type | Model |
|-----------|-----------|-------|
| `flux` | Image | FLUX.1-dev |
| `wan` | Video | Wan 2.1 |
| `hunyuan` | Video | HunyuanVideo 1.5 |

You can list all registered processors with:

```bash
python -m tools.diffusion.preprocessing_multiprocess --list_processors
```

### Image Preprocessing (FLUX)

```bash
python -m tools.diffusion.preprocessing_multiprocess image \
  --image_dir /path/to/images \
  --output_dir /path/to/cache \
  --processor flux \
  --resolution_preset 512p
```

### Video Preprocessing (Wan 2.1)

**Video mode** (encodes the full video as a single sample, recommended for training):

```bash
python -m tools.diffusion.preprocessing_multiprocess video \
  --video_dir /path/to/videos \
  --output_dir /path/to/cache \
  --processor wan \
  --resolution_preset 512p \
  --caption_format sidecar
```

**Frames mode** (extracts evenly-spaced frames, each becomes a separate sample):

```bash
python -m tools.diffusion.preprocessing_multiprocess video \
  --video_dir /path/to/videos \
  --output_dir /path/to/cache \
  --processor wan \
  --mode frames \
  --num_frames 40 \
  --resolution_preset 512p
```

### Video Preprocessing (HunyuanVideo)

```bash
python -m tools.diffusion.preprocessing_multiprocess video \
  --video_dir /path/to/videos \
  --output_dir /path/to/cache \
  --processor hunyuan \
  --target_frames 121 \
  --caption_format meta_json
```

### Key Arguments

**Common arguments:**

| Argument | Description |
|----------|-------------|
| `--processor` | Processor name (`flux`, `wan`, `hunyuan`) |
| `--model_name` | HuggingFace model name (uses processor default if omitted) |
| `--output_dir` | Output directory for cached data |
| `--shard_size` | Number of samples per metadata shard (default: 10000) |

**Image-specific arguments:**

| Argument | Description |
|----------|-------------|
| `--image_dir` | Input image directory |
| `--resolution_preset` | Resolution preset: `256p`, `512p`, `768p`, `1024p`, `1536p` |
| `--max_pixels` | Custom pixel budget (alternative to preset) |
| `--caption_field` | Caption field in JSONL files (`internvl` or `usr`) |
| `--verify` | Verify latents can be decoded back |

**Video-specific arguments:**

| Argument | Description |
|----------|-------------|
| `--video_dir` | Input video directory |
| `--mode` | `video` (full video) or `frames` (extract evenly-spaced frames) |
| `--num_frames` | Number of frames to extract in `frames` mode |
| `--target_frames` | Target frame count (e.g., 121 for HunyuanVideo 4n+1 constraint) |
| `--resolution_preset` | Resolution preset for bucketing |
| `--height` / `--width` | Explicit target size (disables bucketing) |
| `--resize_mode` | Interpolation: `bilinear`, `bicubic`, `nearest`, `area`, `lanczos` |
| `--center_crop` / `--no_center_crop` | Enable/disable center cropping (default: enabled) |
| `--caption_format` | Caption source: `sidecar`, `meta_json`, `jsonl` |
| `--caption_field` | Field name for captions (default: `caption`) |
| `--output_format` | Output format: `meta` (pickle) or `pt` (torch.save) |

## Output Format

The preprocessing tool produces a cache directory organized by resolution bucket:

```
/path/to/cache/
├── 512x512/
│   ├── <hash1>.meta
│   ├── <hash2>.meta
│   └── ...
├── 832x480/
│   └── ...
├── metadata.json          # Global config (processor, model, total items)
└── metadata_shard_0000.json  # Per-sample metadata (paths, resolutions, captions)
```

Each cache file (`.meta` or `.pt`) contains:

- **Encoded latents** — VAE latent representations of the image or video
- **Text embeddings** — Pre-computed from the model's text encoder
- **First frame** — Reference image for image-to-video conditioning (video mode only)
- **Image embeddings** — For models that support i2v conditioning (video mode only)
- **Metadata** — Original and bucket resolutions, caption, source path

## Multiresolution Bucketing

NeMo AutoModel supports multiresolution training through bucketed sampling. This groups samples by their spatial resolution so that each batch contains samples of the same size, avoiding padding waste.

During preprocessing, the `--resolution_preset` argument controls the pixel budget used for bucketing. Available presets: `256p`, `512p`, `768p`, `1024p`, `1536p`. Alternatively, use `--max_pixels` for a custom pixel budget, or `--height`/`--width` to disable bucketing and use a fixed resolution.

During training, the dataloader uses these key configuration parameters:

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
