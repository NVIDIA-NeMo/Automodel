# Integrate Your Own ASR Dataset

This guide shows how to integrate audio datasets into NeMo Automodel for ASR (Automatic Speech Recognition) training. You'll learn about audio preprocessing, architecture-specific collate functions, and YAML configuration.

## Quick Reference

| Dataset | Use Case | Factory Function | Collate Function |
|---------|----------|-----------------|------------------|
| LibriSpeech | English audiobooks (1000h) | `make_librispeech_dataset` | `whisper_collate_fn` or `parakeet_collate_fn` |
| Common Voice | Multilingual speech (100+ langs) | `make_common_voice_dataset` | `whisper_collate_fn` or `parakeet_collate_fn` |
| Custom Audio | Your own data | `make_custom_asr_dataset` | `whisper_collate_fn` or `parakeet_collate_fn` |

## ASR Dataset Structure

ASR datasets pair audio with text transcriptions. Each dataset example contains:
- **audio**: Raw audio waveform array with sampling rate (typically 16kHz)
- **text** or **sentence**: Ground truth transcription

The audio is processed into mel spectrograms by collate functions during training, and transcriptions are tokenized according to the model's vocabulary.

## LibriSpeech Dataset

LibriSpeech is the recommended dataset for English ASR, containing 1000 hours of audiobook recordings with high-quality transcriptions.

### Using LibriSpeech

```python
from nemo_automodel.components.datasets.asr.datasets import make_librispeech_dataset

# Load the clean 100-hour subset
dataset = make_librispeech_dataset(
    path_or_dataset="librispeech_asr",
    split="train.100",
    streaming=False,
    limit_dataset_samples=None  # or specify a limit for debugging
)
```

### Available Splits

- `train.100` - 100 hours of clean training data (recommended for quick experiments)
- `train.clean.360` - 360 hours of clean training data
- `train.other.500` - 500 hours of other training data
- `validation` - Validation split
- `test.clean` - Clean test split
- `test.other` - Other test split

### YAML Configuration

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: train.100
  streaming: false
  limit_dataset_samples: 10000  # Optional: limit for faster iteration

validation_dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: validation
  streaming: false
```

## Common Voice Dataset

Common Voice is a multilingual speech corpus with over 100 languages, contributed by volunteers worldwide.

### Using Common Voice

```python
from nemo_automodel.components.datasets.asr.datasets import make_common_voice_dataset

# Load English Common Voice 17.0
dataset = make_common_voice_dataset(
    path_or_dataset="mozilla-foundation/common_voice_17_0",
    language_code="en",
    split="train",
    streaming=False
)
```

:::{note}
**Availability Note**: As of October 2025, Mozilla Common Voice datasets are no longer hosted on HuggingFace Hub. They must be downloaded from the Mozilla Data Collective and loaded from local paths. For readily available English ASR, use LibriSpeech instead.
:::

### YAML Configuration

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_common_voice_dataset
  path_or_dataset: /path/to/local/common_voice  # Local path
  language_code: en
  split: train
  streaming: false
```

## Custom ASR Dataset

The custom ASR dataset loader allows you to use any HuggingFace audio dataset with configurable column names.

### Using Custom Dataset

```python
from nemo_automodel.components.datasets.asr.datasets import make_custom_asr_dataset

# Load your custom dataset
dataset = make_custom_asr_dataset(
    path_or_dataset="your-username/your-asr-dataset",
    audio_column="audio",  # Column containing audio arrays
    text_column="transcription",  # Column containing text
    split="train",
    streaming=False
)
```

### Column Mapping

The loader automatically renames your columns to the standard `audio` and `text` fields expected by ASR training:
- Your `audio_column` → `audio`
- Your `text_column` → `text`

This allows seamless integration with existing ASR collate functions.

### YAML Configuration

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_custom_asr_dataset
  path_or_dataset: your-username/your-asr-dataset
  audio_column: audio
  text_column: transcription
  split: train
  streaming: false
  limit_dataset_samples: null
```

### Supported Formats

The custom loader supports any format that HuggingFace `datasets` can load:
- Parquet files
- JSON/JSONL files
- CSV files with audio paths
- Arrow datasets
- HuggingFace Hub datasets

## Audio Requirements

### Sampling Rate

Most ASR models expect **16kHz audio**. The audio processing pipeline will automatically resample if needed, but for best performance, ensure your audio is already at 16kHz.

### Supported Formats

Audio decoding uses `torchcodec` with FFmpeg backends, supporting:
- WAV, MP3, FLAC, OGG, M4A
- Any format supported by FFmpeg

### System Dependencies

**Required**: FFmpeg libraries for audio decoding

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
```

**macOS:**
```bash
brew install ffmpeg
```

**Docker**: Pre-installed in NeMo AutoModel containers.

### Duration Recommendations

- **Training**: 1-30 seconds per audio clip (optimal: 5-15 seconds)
- **Validation**: Similar to training distribution
- **Very long audio** (>30s): May require increased memory or sequence length limits

## Collate Functions

ASR models require architecture-specific collate functions that process audio into mel spectrograms and prepare labels.

### Whisper Collate Function

For Whisper encoder-decoder (Seq2Seq) models.

**Features**:
- Converts audio to 80-channel mel spectrograms
- Tokenizes transcriptions with padding
- Creates `decoder_input_ids` via right-shifted labels (teacher forcing)
- Prepends decoder start token

**Usage in YAML**:
```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 4
  num_workers: 4
  pin_memory: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.asr.collate_fns.whisper_collate_fn
    max_length: 448  # Max tokens for transcription
```

**Parameters**:
- `max_length` (int, default=448): Maximum length for tokenized transcriptions

**Returns**:
- `input_features`: Mel spectrograms (batch_size, 80, 3000)
- `decoder_input_ids`: Right-shifted labels for teacher forcing
- `labels`: Target transcriptions with -100 for padding

### Parakeet Collate Function

For Parakeet CTC encoder-only models.

**Features**:
- Converts audio to mel spectrograms for CTC
- Tokenizes transcriptions for CTC loss
- Generates attention masks for variable-length sequences
- No decoder setup required (encoder-only)

**Usage in YAML**:
```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 8
  num_workers: 4
  pin_memory: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.asr.collate_fns.parakeet_collate_fn
    max_length: null  # Optional: limit sequence length
```

**Parameters**:
- `max_length` (int, optional): Maximum length for padded sequences

**Returns**:
- `input_features`: Mel spectrograms with shape (batch_size, seq_len, feature_dim)
- `attention_mask`: Masks for variable-length audio
- `labels`: Target transcriptions with -100 for padding

## Complete YAML Example

### Whisper Example

```yaml
model:
  _target_: nemo_automodel._transformers.auto_model.NeMoAutoModelForSpeechSeq2Seq.from_pretrained
  pretrained_model_name_or_path: openai/whisper-small

dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: train.100
  streaming: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: validation
  streaming: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 4
  num_workers: 4
  pin_memory: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.asr.collate_fns.whisper_collate_fn
    max_length: 448
```

### Parakeet Example

```yaml
model:
  _target_: nemo_automodel._transformers.auto_model.NeMoAutoModelForCTC.from_pretrained
  pretrained_model_name_or_path: nvidia/parakeet-ctc-0.6b

dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: train.100
  streaming: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.asr.datasets.make_librispeech_dataset
  path_or_dataset: librispeech_asr
  split: validation
  streaming: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 8
  num_workers: 4
  pin_memory: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.asr.collate_fns.parakeet_collate_fn
```

## Troubleshooting

### Audio Format Issues

**Problem**: `RuntimeError: Failed to load audio`

**Solution**: Install FFmpeg system libraries:
```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# macOS
brew install ffmpeg
```

### Memory Problems with Long Audio

**Problem**: `CUDA out of memory` with long audio files

**Solutions**:
1. Reduce batch size in dataloader config
2. Limit audio duration during dataset loading
3. Use gradient accumulation to maintain effective batch size
4. Use PEFT/LoRA to reduce memory footprint

### Sampling Rate Mismatches

**Problem**: Audio quality degradation or errors

**Solution**: Ensure audio is at 16kHz. The pipeline will resample automatically, but pre-resampled audio is more efficient:
```python
# If your audio is not 16kHz, it will be resampled automatically
# For best performance, resample your dataset beforehand
```

### Text Encoding Issues

**Problem**: Special characters or non-ASCII text causing errors

**Solution**:
- Whisper models handle multilingual UTF-8 text natively
- For Parakeet, ensure transcriptions match the model's vocabulary
- Clean transcriptions: remove timestamps, speaker labels, etc.

### Dataset Loading Errors

**Problem**: `DatasetNotFoundError` or permission errors

**Solutions**:
1. Verify dataset name/path is correct
2. For private HuggingFace datasets, authenticate: `huggingface-cli login`
3. For local datasets, use absolute paths
4. Check HuggingFace Hub status if dataset won't load

## See Also

- [ASR Model Coverage](../../model-coverage/asr.md) - Supported ASR models
- [Dataset Overview](../dataset-overview.md) - Overview of all dataset types
- `examples/asr_finetune/` - Complete training examples with configs for Whisper and Parakeet models
