# Integrate Your Own Dataset

This guide shows you how to integrate your own dataset into NeMo Automodel for training. You'll learn about three main dataset types: **completion datasets** for language modeling (like [HellaSwag](https://huggingface.co/datasets/rowan/hellaswag)), **instruction datasets** for question-answering tasks (like [SQuAD](https://huggingface.co/datasets/rajpurkar/squad)), and **multi-modal datasets** that combine text with images or other modalities. We'll cover how to create custom datasets by implementing the required methods and preprocessing functions, and finally show you how to specify your own data logic using YAML configuration with file paths—allowing you to define custom dataset processing without modifying the main codebase.

## Quick Start Summary
| **Type**        |  **Use Case**    | **Example** | **Preprocessor**               | **Section**              |
| --------------- | ------------------ | -------------- | --------------------------------- | --------------------------- |
| ✍️ Completion   | Language modeling  | HellaSwag      | `SFTSingleTurnPreprocessor`       | [Jump](#completion-datasets)  |
| 🗣️ Instruction  | Question answering | SQuAD          | `make_*` function                 | [Jump](#instruction-datasets) |
| 🖼️ Multi-modal  | Vision + Language  | MedPix-VQA     | `apply_chat_template`, collate fn | [Jump](#multi-modal-datasets) |

## Types of Supported Datasets

NeMo Automodel supports a variety of datasets, depending on the task.
### Completion Datasets

**Completion datasets** are single text sequences designed for language modeling where the model learns to predict the next token given a context. These datasets typically contain a context (prompt) and a target (completion) that the model should learn to generate.

#### Example: HellaSwag

The [HellaSwag](https://huggingface.co/datasets/rowan/hellaswag) dataset is a popular completion dataset used for commonsense reasoning. It contains situations with multiple-choice endings where the model must choose the most plausible continuation.

**HellaSwag dataset structure:**
- **Context (`ctx`)**: A situation or scenario description
- **Endings**: Multiple possible completions (4 options)
- **Label**: Index of the correct ending

**Example:**
```
Context: "A man is sitting at a piano in a large room."
Endings: [
  "He starts playing a beautiful melody.",
  "He eats a sandwich while sitting there.",
  "He suddenly becomes invisible.",
  "He transforms into a robot."
]
Label: 0  # First ending is correct
```

#### Preprocessing with SFTSingleTurnPreprocessor

NeMo Automodel provides the `SFTSingleTurnPreprocessor` class to handle completion datasets. This processor:

1. **Extracts context and target** using `get_context()` and `get_target()`.
2. **Tokenizes and cleans** context and target separately.
3. **Concatenates** them into one sequence.
4. **Creates loss mask**: `-100` for context, target IDs for target.
5. **Pads** sequences to equal length.


#### Create Your Own Completion Dataset

To adapt your dataset into this format, define a class like this:

```python
from datasets import load_dataset
from nemo_automodel.components.datasets.utils import SFTSingleTurnPreprocessor

class MyCompletionDataset:
    def __init__(self, path_or_dataset, tokenizer, split="train"):
        raw_datasets = load_dataset(path_or_dataset, split=split)
        processor = SFTSingleTurnPreprocessor(tokenizer)
        self.dataset = processor.process(raw_datasets, self)

    def get_context(self, examples):
        """Extract context/prompt from your dataset"""
        return examples["context_field"]  # Replace with your context field

    def get_target(self, examples):
        """Extract target/completion from your dataset"""
        return examples["target_field"]   # Replace with your target field

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
```


### Instruction Datasets

**Instruction datasets** are question-answer pairs where the model learns to respond to specific instructions or questions. These datasets are structured as context-question pairs with corresponding answers, making them ideal for teaching models to follow instructions and provide accurate responses.

#### Example: SQuAD

The [SQuAD (Stanford Question Answering Dataset)](https://huggingface.co/datasets/rajpurkar/squad) is a popular instruction dataset for reading comprehension. It contains questions based on Wikipedia articles along with their answers.

**SQuAD dataset structure:**
- **Context**: A paragraph of text from Wikipedia
- **Question**: A question about the context
- **Answers**: The correct answer with its position in the context

#### Create Your Own Instruction Dataset

The [`squad.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/squad.py) file contains the implementation for processing the SQuAD dataset into a format suitable for instruction tuning. It defines a dataset class and preprocessing functions that extract the context, question, and answer fields, concatenate them into a prompt-completion format, and apply tokenization, padding, and loss masking. This serves as a template for building custom instruction datasets by following a similar structure and adapting the extraction logic to your dataset's schema.

Based on the SQuAD implementation in `squad.py`, you can create your own instruction dataset using the `make_squad_dataset` pattern:

```python
from datasets import load_dataset

def make_my_instruction_dataset(
    tokenizer,
    seq_length=None,
    limit_dataset_samples=None,
    split="train",
    dataset_name="your-dataset-name",
):
    if limit_dataset_samples:
        split = f"{split}[:{limit_dataset_samples}]"

    dataset = load_dataset(dataset_name, split=split)

    return dataset.map(
        your_own_fmt_fn,  # Your formatting function
        batched=False,
        remove_columns=dataset.column_names,
    )
```

### Multi-modal Datasets

Multi-modal datasets combine text with other input types (e.g., images, audio or video) and are essential for training Vision-Language Models (VLMs). These datasets introduce specific challenges such as aligning modalities, batching diverse data types, and formatting prompts for multi-turn, multi-modal dialogue.

NeMo Automodel supports multi-modal dataset integration through flexible preprocessing, custom formatting, and YAML-based configuration.

#### Typical types in Multi-modal Datasets
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

#### Collate Functions

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


## Packed Sequence Support in NeMo AutoModel
NeMo AutoModel supports **packed sequences**, a technique to optimize training with variable-length sequences (e.g., text) by minimizing padding.

### What is a Packed Sequence?
Instead of padding each sequence to a fixed length (wasting computation on `[PAD]` tokens), packed sequences:
- Concatenate short sequences into a single continuous sequence.
- Separate sequences with special tokens (e.g., `[EOS]`).
- Track lengths via a "attention mask" to prevent cross-sequence information leakage.

### Benefits
- Reduces redundant computation on padding tokens leading to faster training.
- Enables larger effective batch sizes leading to better GPU utilization.
- Especially useful for language modeling and text datasets.


### Enable Packed Sequences in NeMo Automodel

To enable packed sequences, add these keys to your recipe's YAML config:
```
packed_sequence:
   # Set packed_sequence_size > 0 to run with packed sequences
   packed_sequence_size: 1024
   split_across_pack: False
```

The `packed_sequence` has two options:
- **packed_sequence_size**: Defines the total token length of each packed sequence, higher values require higher GPU memory usage.
- **split_across_pack**: If two will split a sequence across different packed sequences.


## Troubleshooting Tips

- **Tokenization Mismatch?** Ensure your tokenizer aligns with the model's expected inputs.
- **Dataset too large?** Use `limit_dataset_samples` in your YAML config to load a subset, useful for quick debugging.
- **Loss not decreasing?** Verify that your loss mask correctly ignores prompt tokens.
