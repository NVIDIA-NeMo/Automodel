# Bringing Your Own Dataset

This guide shows you how to bring your own dataset into NeMo Automodel for training. You'll learn about three main dataset types: **completion datasets** for language modeling (like HellaSwag), **instruction datasets** for question-answering tasks (like SQuAD), and **multi-modal datasets** that combine text with images or other modalities. We'll cover how to create custom datasets by implementing the required methods and preprocessing functions, and finally show you how to specify your own data logic using YAML configuration with file paths—allowing you to define custom dataset processing without modifying the main codebase.

## Types of Supported Datasets

NeMo Automodel supports several types of datasets for different training scenarios.

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

#### SFTSingleTurnPreprocessor

NeMo Automodel provides the `SFTSingleTurnPreprocessor` class to handle completion datasets. This processor:

1. **Extracts context and target** using `get_context()` and `get_target()`
2. **Tokenizes and cleans** context and target separately
3. **Concatenates** them into one sequence
4. **Creates loss mask**: `-100` for context, target IDs for target
5. **Pads** sequences to equal length


#### Creating Your Own Completion Dataset

To create a completion dataset like HellaSwag, you need to implement a class with `get_context()` and `get_target()` methods:

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

#### Creating Your Own Instruction Dataset

The `squad.py` file contains the implementation for processing the SQuAD dataset into a format suitable for instruction tuning. It defines a dataset class and preprocessing functions that extract the context, question, and answer fields, concatenate them into a prompt-completion format, and apply tokenization, padding, and loss masking. This serves as a template for building custom instruction datasets by following a similar structure and adapting the extraction logic to your dataset's schema.

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
    """
    Load and preprocess your instruction dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        seq_length: Optional sequence length for padding
        limit_dataset_samples: Limit number of samples
        split: Dataset split to use
        dataset_name: Your dataset identifier
    
    Returns:
        Processed dataset with input_ids, labels, and loss_mask
    """
    # Load dataset
    if limit_dataset_samples:
        split = f"{split}[:{limit_dataset_samples}]"
    
    dataset = load_dataset(dataset_name, split=split)

    # Apply formatting
    return dataset.map(
        your_own_fmt_fn,
        batched=False,
        remove_columns=dataset.column_names,
    )
```

### Multi-modal Datasets

Multi-modal datasets combine text with other modalities like images, audio, or video. These datasets are essential for training Vision-Language Models (VLMs) and other multi-modal AI systems.

#### Example: MedPix-VQA Dataset

The [MedPix-VQA](https://huggingface.co/datasets/mmoukouba/MedPix-VQA) dataset is a comprehensive medical Visual Question Answering dataset designed for training and evaluating VQA models in the medical domain. It contains medical images from MedPix, a well-known medical image database, paired with questions and answers that focus on medical image interpretation.

The dataset consists of 20,500 examples with the following structure:
- **Training Set**: 17,420 examples (85%)
- **Validation Set**: 3,080 examples (15%)
- **Columns**: `image_id`, `mode`, `case_id`, `question`, `answer`

The dataset preprocessing performs the following steps:

1. **Load the dataset** using HuggingFace's `datasets` library
2. **Extract question-answer pairs** - Process the `question` and `answer` fields from the dataset
3. **Convert to Huggingface message list format** - Transform the data into a chat-like format suitable for Huggingface Autoprocessor's `apply_chat_template` function:

```python
# Example of the conversation format created
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

For more detailed examples of how to process multi-modal datasets for VLMs, see the [examples in `datasets.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/vlm/datasets.py). These examples demonstrate how to load, preprocess, and format multi-modal data.

#### Collate Functions

NeMo Automodel provides specialized collate functions for different VLM processors. The collate function is responsible for batching examples and preparing them for model input.

If your model provides a HuggingFace `AutoProcessor`, you can use it directly for preprocessing and collation. Otherwise, you will need to implement your own preprocessing and collate logic tailored to your model and dataset. We provide [example custom collate functions](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/vlm/collate_fns.py) that you can use as references for your implementation. After you implement your own collate function, you can specify it in your YAML config.


## Customizing Data Processing with YAML Configuration

NeMo Automodel supports specifying the `_target_` parameter using Python dotted module paths. This allows you to reference a function or class directly from an installed Python module.

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
```

NeMo Automodel also supports using file paths directly in the `_target_` parameter, which enables you to specify custom dataset functions from a file. This is particularly useful when you want to define your own dataset processing logic and use it directly in your YAML configuration.

### Syntax

The `_target_` parameter supports the following file path format:

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
In the above example, it will call the `build_my_dataset` with the rest of the parameters (i.e., num_blocks) that are under the dataset section. This feature makes it very convenient to define custom datasets and use them directly through YAML configuration without the need to modify the main codebase or create formal Python packages.
