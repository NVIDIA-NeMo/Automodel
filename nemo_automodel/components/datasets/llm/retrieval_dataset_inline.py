# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from typing import Any, Dict, List, Union

from datasets import Dataset, concatenate_datasets

INLINE_CORPUS_ID = "__inline__"


def _load_json_or_jsonl(path: str) -> Union[dict, list]:
    """Load a JSON file, falling back to JSONL (one JSON object per line)."""
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # Fall back to JSONL
            f.seek(0)
            records: list[dict] = []
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e
            if not records:
                raise ValueError(f"No records found in JSONL file: {path}")
            return records


def _coerce_to_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_inline_doc(doc: Any) -> Dict[str, Any]:
    """Normalize an inline doc (text/image provided) into a canonical dict shape."""
    if isinstance(doc, dict):
        if "text" not in doc:
            raise ValueError(f"Inline doc dict must include 'text'. Got keys: {sorted(list(doc.keys()))}")
        text = doc.get("text", "")
        image = doc.get("image", "")
        nr_ocr = doc.get("nr_ocr", "")
    else:
        text = doc if isinstance(doc, str) else str(doc)
        image = ""
        nr_ocr = ""
    return {
        "id": "",
        "text": "" if text is None else str(text),
        "image": "" if image is None else image,
        "nr_ocr": "" if nr_ocr is None else str(nr_ocr),
    }


def _resolve_doc_to_example(doc: Any) -> dict:
    """
    Resolve a doc reference into an example dict with keys: text, image, nr_ocr.

    Supported doc forms:
    - `str`: interpreted as inline document text
    - `dict`: must include `text` (optionally `image`, `nr_ocr`)
    """
    example = {"text": "", "image": "", "nr_ocr": ""}
    if isinstance(doc, dict):
        if "text" not in doc:
            raise ValueError(f"Inline doc dict must include 'text'. Got keys: {sorted(list(doc.keys()))}")

        if "text" in doc and doc["text"] is not None:
            example["text"] = str(doc["text"])
        if "image" in doc and doc["image"] is not None:
            example["image"] = doc["image"]
        if "nr_ocr" in doc and doc["nr_ocr"] is not None:
            example["nr_ocr"] = str(doc["nr_ocr"])
        return example

    if isinstance(doc, str):
        example["text"] = doc
        return example

    # Fallback: coerce to string text
    example["text"] = str(doc)
    return example


def load_datasets(data_dir_list: Union[List[str], str], concatenate: bool = True):
    """
    Load retrieval datasets from JSON/JSONL files.

    Copied from nemo-retriever-research/src/data/datasets.py

    Returns:
        Tuple of (dataset, corpus_dict)
    """
    if not isinstance(data_dir_list, list):
        data_dir_list = [data_dir_list]
    datasets = []
    for data_dir in data_dir_list:
        train_data = _load_json_or_jsonl(data_dir)

        # Corpus-id based format is intentionally not supported in this "inline" loader.
        # Use `nemo_automodel.components.datasets.llm.retrieval_dataset.load_datasets` instead.
        is_corpus_id_format = isinstance(train_data, dict) and "corpus" in train_data and "data" in train_data
        if is_corpus_id_format:
            raise ValueError(
                "Corpus-id retrieval format (top-level 'corpus' + 'data') is not supported by "
                "retrieval_dataset_inline. Use retrieval_dataset.py (corpus-id) or convert the dataset "
                "to inline JSONL with inline `pos_doc`/`neg_doc` texts."
            )

        # Inline-text format (JSONL or JSON list/dict). Example record:
        # {"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}
        if isinstance(train_data, dict) and "data" in train_data and "corpus" not in train_data:
            records = train_data["data"]
        else:
            records = train_data

        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ValueError(f"Unsupported inline retrieval dataset container type: {type(records)} in {data_dir}")

        normalized_data = []
        file_prefix = os.path.basename(data_dir)
        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValueError(f"Inline retrieval record must be a dict. Got: {type(item)} ({item})")

            question = item.get("query", item.get("question", None))
            if question is None:
                raise ValueError(f"Inline retrieval record must include 'query' or 'question'. Got: {item}")

            if "pos_doc" not in item:
                raise ValueError(f"Inline retrieval record must include 'pos_doc'. Got: {item}")
            if "neg_doc" not in item:
                raise ValueError(f"Inline retrieval record must include 'neg_doc'. Got: {item}")

            question_id = item.get("question_id", item.get("id", f"{file_prefix}:{idx}"))
            corpus_id = item.get("corpus_id", INLINE_CORPUS_ID)

            pos_docs_raw = _coerce_to_list(item.get("pos_doc"))
            if len(pos_docs_raw) == 0:
                raise ValueError(f"Inline retrieval record pos_doc cannot be empty. Got: {item}")

            normalized_item = {
                "question_id": question_id,
                "question": question,
                "corpus_id": corpus_id,
                "pos_doc": [_normalize_inline_doc(d) for d in pos_docs_raw],
                "neg_doc": [_normalize_inline_doc(d) for d in _coerce_to_list(item.get("neg_doc"))],
            }
            normalized_data.append(normalized_item)

        datasets.append(Dataset.from_list(normalized_data))

    if concatenate:
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets
    return (dataset, {})


def _transform_func(examples, num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """
    Transform function to convert from raw format to training format.
    Same as _format_process_data in RetrievalMultiModalDatasetLoader.

    Args:
        examples: Batch of examples with question, corpus_id, pos_doc, neg_doc
        num_neg_docs: Number of negative documents to use
        corpus_dict: Dictionary mapping corpus_id to corpus objects
        use_dataset_instruction: Whether to use instruction from dataset's metadata
    """
    # Handle both batched and single examples
    is_batched = isinstance(examples["question"], list)

    if not is_batched:
        # Convert single example to batch for uniform processing
        examples = {k: [v] for k, v in examples.items()}

    questions = examples["question"]
    corpus_ids = examples["corpus_id"]
    batch_positives = examples["pos_doc"]
    batch_negatives = examples["neg_doc"]

    cur_pos_neg_doc_batch = []

    for i_example in range(len(questions)):
        cur_pos_neg_doc = []

        # Get one positive doc (take first one)
        positives = batch_positives[i_example]
        if isinstance(positives, list) and len(positives) > 0:
            cur_pos_neg_doc.append(positives[0])
        elif isinstance(positives, list) and len(positives) == 0:
            raise ValueError(f"pos_doc cannot be empty for question='{questions[i_example]}'")
        else:
            cur_pos_neg_doc.append(positives)

        # Get negatives (limit to num_neg_docs)
        negatives = batch_negatives[i_example]
        if not isinstance(negatives, list):
            negatives = _coerce_to_list(negatives)
        if num_neg_docs > 0 and len(negatives) == 0:
            raise ValueError(
                f"neg_doc must contain at least 1 document to sample {num_neg_docs} negatives "
                f"for question='{questions[i_example]}'"
            )
        if num_neg_docs > 0:
            neg_ids = [i for i in range(len(negatives))]
            cur_neg_ids = [neg_ids[idx % len(neg_ids)] for idx in range(num_neg_docs)]
            cur_pos_neg_doc += [negatives[n_id] for n_id in cur_neg_ids]

        cur_pos_neg_doc_batch.append(cur_pos_neg_doc)

    # Extract text and images from corpus
    cur_pos_neg_text_batch = []
    cur_pos_neg_image_batch = []
    query_instruction_batch = []
    passage_instruction_batch = []

    for idx_doc, docs in enumerate(cur_pos_neg_doc_batch):
        cur_pos_neg_text = []
        cur_pos_neg_image = []
        cur_corpus_id = corpus_ids[idx_doc]

        for doc in docs:
            cur_doc = _resolve_doc_to_example(doc)

            # Extract text
            if cur_doc["text"] != "" and not cur_doc["image"]:
                text = cur_doc["text"]
            elif cur_doc["image"]:
                text = " " + cur_doc["text"] if cur_doc["text"] else ""
                text = text.strip()
            else:
                text = ""

            cur_pos_neg_text.append(text)

            # Extract image
            if cur_doc["image"] != "":
                cur_doc["image"] = cur_doc["image"].convert("RGB")
            cur_pos_neg_image.append(cur_doc["image"])

        cur_pos_neg_text_batch.append(cur_pos_neg_text)
        cur_pos_neg_image_batch.append(cur_pos_neg_image)

        if use_dataset_instruction and cur_corpus_id in corpus_dict:
            query_instruction_batch.append(corpus_dict[cur_corpus_id].query_instruction)
            passage_instruction_batch.append(corpus_dict[cur_corpus_id].passage_instruction)
        else:
            query_instruction_batch.append("")
            passage_instruction_batch.append("")

    result = {
        "question": questions,
        "doc_text": cur_pos_neg_text_batch,
        "doc_image": cur_pos_neg_image_batch,
        "query_instruction": query_instruction_batch,
        "passage_instruction": passage_instruction_batch,
    }

    # If input was not batched, return single example
    if not is_batched:
        result = {k: v[0] for k, v in result.items()}

    return result


def _create_transform_func(num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """Create transform function with specified number of negative documents."""

    def transform(examples):
        return _transform_func(
            examples,
            num_neg_docs=num_neg_docs,
            corpus_dict=corpus_dict,
            use_dataset_instruction=use_dataset_instruction,
        )

    return transform


def make_retrieval_dataset(
    data_dir_list: Union[List[str], str],
    data_type: str = "train",
    train_n_passages: int = 5,
    eval_negative_size: int = 10,
    seed: int = 42,
    do_shuffle: bool = False,
    max_train_samples: int = None,
    train_data_select_offset: int = 0,
    use_dataset_instruction: bool = False,
):
    """
    Load and return dataset in retrieval format for biencoder training.

    This function loads data from JSON files using the same method as
    RetrievalMultiModalDatasetLoader and returns it ready for training.
    Uses set_transform() for lazy evaluation - tokenization is handled by collator.

    Args:
        data_dir_list: Path(s) to JSON file(s) containing training data
        data_type: Type of data ("train" or "eval")
        train_n_passages: Number of passages for training (1 positive + n-1 negatives)
        eval_negative_size: Number of negative documents for evaluation
        seed: Random seed for reproducibility (for shuffling if needed)
        do_shuffle: Whether to shuffle the dataset
        max_train_samples: Maximum number of training samples to use
        train_data_select_offset: Offset for selecting training samples

    Returns:
        A HuggingFace Dataset where each example is a dict with keys:
        - 'question': Query text
        - 'doc_text': List of document texts [positive, negatives...]
        - 'doc_image': List of images or empty strings

    Note:
        Tokenization should be handled by a collator (e.g., RetrievalBiencoderCollator)
        which is more efficient for batch padding and supports dynamic processing.
    """

    logging.info(f"Loading data from {data_dir_list if isinstance(data_dir_list, str) else len(data_dir_list)} file(s)")

    # Load datasets using the same method as RetrievalMultiModalDatasetLoader
    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True)

    logging.info(f"Loaded dataset with {len(dataset)} examples")

    # Apply same processing as _get_processed_dataset
    if data_type == "train":
        if do_shuffle:
            dataset = dataset.shuffle(seed=seed)
        if max_train_samples is not None:
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

        # Set transform for training (train_n_passages - 1 negatives)
        negative_size = train_n_passages - 1
        dataset.set_transform(_create_transform_func(negative_size, corpus_dict, use_dataset_instruction))

    elif data_type == "eval":
        # Set transform for evaluation
        dataset.set_transform(_create_transform_func(eval_negative_size, corpus_dict, use_dataset_instruction))

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")

    return dataset
