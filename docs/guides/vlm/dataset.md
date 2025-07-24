# Integrate Your Own Dataset

This guide shows you how to integrate your own dataset into NeMo Automodel for training.
You'll learn about **multi-modal datasets** that combine text with images or other modalities. We'll cover how to create custom datasets by implementing the required methods and preprocessing functions, and finally show you how to specify your own data logic using YAML configuration with file paths—allowing you to define custom dataset processing without modifying the main codebase.

## Quick Start Summary
| **Type**        |  **Use Case**    | **Example** | **Preprocessor**               | **Section**              |
| --------------- | ------------------ | -------------- | --------------------------------- | --------------------------- |
| 🖼️ Multi-modal  | Vision + Language  | MedPix-VQA     | `apply_chat_template`, collate fn | [Jump](#multi-modal-datasets) |
| 🎤 Audio        | Speech + Language  | Common Voice 17| `apply_chat_template`, collate fn | [Jump](#audio-datasets) |


## Multi-modal Datasets

Multi-modal datasets combine text with other input types (e.g., images, audio, or video) and are essential for training Vision-Language Models (VLMs). These datasets introduce specific challenges such as aligning modalities, batching diverse data types, and formatting prompts for multi-turn, multi-modal dialogue.

NeMo Automodel supports multi-modal dataset integration through flexible preprocessing, custom formatting, and YAML-based configuration.

### Typical Types in Multi-modal Datasets
A multi-modal dataset typically contains:
- **Images, videos, audios** or other non-text modalities.
- **Textual inputs** such as questions, instructions, or captions.
- **Answers** or expected outputs from the model.

These are formatted into structured conversations or instruction-response pairs for use with VLMs like BLIP, Llava, or Flamingo.

#### Example: MedPix-VQA Dataset

The [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) dataset is a comprehensive medical Visual Question Answering dataset designed for training and evaluating VQA models in the medical domain. It contains radiological images (from MedPix; well-known medial image dataset) and associated QA pairs used for medical image interpretation.

**Structure**:
- 20,500 total examples
- Columns: `image_id`, `mode`, `case_id`, `question`, `answer`

```json
{
  "image_id": "medpix_0143.jpg",
  "mode": "CT",
  "case_id": "case_101",
  "question": "What abnormality is visible in the left hemisphere?",
  "answer": "Subdural hematoma"
}
```

The example dataset preprocessing performs the following steps:

1. Loads the dataset using Hugging Face's `datasets` library.
2. Extracts the `question` and `answer`.
3. Transforms the data into a chat-like format that is compatible with Hugging Face's Autoprocessor `apply_chat_template` function. For example:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": example["image_id"]},
            {"type": "text", "text": example["question"]},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": example["answer"]}]
    },
]
```

For more detailed examples of how to process multi-modal datasets for VLMs, see the examples in [`datasets.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/vlm/datasets.py).

## Audio Datasets

Audio datasets combine speech input with text transcriptions and are essential for training models capable of speech recognition and transcription tasks. NeMo Automodel supports audio dataset integration through specialized preprocessing functions and custom collate functions for multimodal models like Phi-4.

### Example: Common Voice 17 Dataset

The [Common Voice 17](https://huggingface.co/datasets/ysdede/commonvoice_17_tr_fixed) dataset is a comprehensive speech recognition dataset containing audio clips and corresponding transcriptions. This particular version focuses on Turkish speech data and has been preprocessed and fixed for compatibility with modern training frameworks.

**Structure**:
- **Audio**: Speech recordings in various formats
- **Transcription**: Text transcriptions of the spoken content
- **Use case**: Speech-to-text transcription for multimodal models

```json
{
  "audio": {
    "path": "common_voice_tr_17528071.mp3",
    "array": [-0.1600779, -0.13843077],
    "sampling_rate": 16000
  },
  "transcription": "Kosova başkentinswki yolcu sayısı arttı."
}
```

The example dataset preprocessing performs the following steps:

1. Loads the dataset using Hugging Face's `datasets` library.
2. Extracts the `audio` and `transcription` fields.

For more detailed examples of how to process multi-modal datasets, see the examples in [`datasets.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/vlm/datasets.py).


### Collate Functions

NeMo Automodel provides specialized collate functions for different VLM processors. The collate function is responsible for batching examples and preparing them for model input.

Multi-modal models require custom collate functions to batch and process each sample correctly. If your model uses a Hugging Face `AutoProcessor`, you can use it directly. Otherwise, you can define your own collate logic and point to it in your YAML config. We provide [example custom collate functions](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/vlm/collate_fns.py) that you can use as references for your implementation. After you implement your own collate function, you can specify it in your YAML config.


## YAML-based Custom Dataset Configuration

NeMo Automodel supports YAML-based dataset specification using the _target_ key. This lets you reference dataset-building classes or functions using either:

- 1. Python Dotted Path

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
```

- 2. File Path + Function Name

```
<file-path>:<function-name>
```

Where:
- `<file-path>`: The absolute path to a Python file containing your dataset function
- `<function-name>`: The name of the function to call from that file

```yaml
dataset:
  _target_: /path/to/your/custom_dataset.py:build_my_dataset
  num_blocks: 111
```
This will call `build_my_dataset()` from the specified file with the other keys (e.g., num_blocks) as arguments. This approach allows you to integrate custom datasets via config alone—no need to alter the codebase or package structure.


## Troubleshooting Tips

- **Tokenization Mismatch?** Ensure your tokenizer aligns with the model's expected inputs.
- **Dataset too large?** Use `limit_dataset_samples` in your YAML config to load a subset, useful for quick debugging.
- **Loss not decreasing?** Verify that your loss mask correctly ignores prompt tokens.
