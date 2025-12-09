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
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Union

from datasets import Dataset, concatenate_datasets, load_dataset

EXAMPLE_TEMPLATE = {"text": "", "image": "", "nr_ocr": ""}


class AbstractDataset(ABC):
    @abstractmethod
    def get_document_by_id(self, id):
        pass

    @abstractmethod
    def get_all_ids(self):
        pass


class TextQADataset(AbstractDataset):
    def __init__(self, path):
        self.path = path
        self.data = load_dataset(path)["train"]
        docid2idx = {}
        for idx, docid in enumerate(self.data["id"]):
            docid2idx[str(docid)] = idx
        self.docid2idx = docid2idx

    def get_document_by_id(self, id):
        example = deepcopy(EXAMPLE_TEMPLATE)
        example["text"] = self.data[self.docid2idx[id]]["text"]
        return example

    def get_all_ids(self):
        return sorted(list(self.docid2idx.keys()))


DATASETS = {
    "TextQADataset": TextQADataset,
}


def load_corpus_metadata(path: str):
    path_metadata = os.path.join(path, "merlin_metadata.json")
    if not os.path.isfile(path_metadata):
        raise ValueError("Metadata File for Corpus does not exist: " + path_metadata)

    metadata = json.load(open(path_metadata, "r"))
    return metadata


def load_corpus(path, metadata: Optional[dict] = None):
    if metadata is None:
        metadata = load_corpus_metadata(path)
    if metadata["class"] not in DATASETS:
        raise ValueError("DatasetClass is not implemented: " + metadata["class"])
    corpus = DATASETS[metadata["class"]](path)
    corpus_id = metadata["corpus_id"]
    return (corpus_id, corpus)


def add_corpus(qa_corpus_paths: Union[dict, list], corpus_dict: dict):
    if corpus_dict is None:
        raise ValueError("Corpus dictionary is not provided")
    if not isinstance(qa_corpus_paths, list):
        qa_corpus_paths = [qa_corpus_paths]

    for corpus_info in qa_corpus_paths:
        corpus_metadata = load_corpus_metadata(corpus_info["path"])
        if corpus_metadata["corpus_id"] in corpus_dict:
            if corpus_dict[corpus_metadata["corpus_id"]].path != corpus_info["path"]:
                raise ValueError(
                    "Two Different Datasets have the same corpus id but different paths: "
                    + "1. "
                    + corpus_dict[corpus_metadata["corpus_id"]].path
                    + "2. "
                    + corpus_info["path"]
                )
        else:
            corpus_id, corpus = load_corpus(corpus_info["path"], corpus_metadata)
            corpus_dict[corpus_id] = corpus


def load_datasets(data_dir_list: Union[List[str], str], concatenate: bool = True):
    """
    Load datasets from JSON files.

    Copied from nemo-retriever-research/src/data/datasets.py

    Returns:
        Tuple of (dataset, corpus_dict)
    """
    REQUIRED_FIELDS = ["question_id", "question", "corpus_id", "pos_doc", "neg_doc"]
    if not isinstance(data_dir_list, list):
        data_dir_list = [data_dir_list]
    corpus_dict = {}
    datasets = []
    for data_dir in data_dir_list:
        with open(data_dir, "r") as f:
            train_data = json.load(f)
        qa_corpus_paths = train_data["corpus"]
        add_corpus(qa_corpus_paths, corpus_dict)

        # Extract only the required fields for training, ignoring extra fields
        normalized_data = []
        for item in train_data["data"]:
            # Extract only the essential fields we need
            missing = [f for f in REQUIRED_FIELDS if f not in item]
            if missing:
                raise ValueError(f"Missing required fields: {missing} in train_data item: {item}")
            normalized_item = {
                "question_id": item["question_id"],
                "question": item["question"],
                "corpus_id": item["corpus_id"],
            }
            # Extract pos_doc with only id field
            normalized_item["pos_doc"] = []
            for doc in item["pos_doc"]:
                if isinstance(doc, dict) and "id" in doc:
                    normalized_item["pos_doc"].append({"id": doc["id"]})
                else:
                    # Handle case where doc might be just a string ID
                    doc_id = doc if isinstance(doc, str) else str(doc)
                    normalized_item["pos_doc"].append({"id": doc_id})
            # Extract neg_doc with only id field
            normalized_item["neg_doc"] = []
            for doc in item["neg_doc"]:
                if isinstance(doc, dict) and "id" in doc:
                    normalized_item["neg_doc"].append({"id": doc["id"]})
                else:
                    # Handle case where doc might be just a string ID
                    doc_id = doc if isinstance(doc, str) else str(doc)
                    normalized_item["neg_doc"].append({"id": doc_id})
            normalized_data.append(normalized_item)

        datasets.append(Dataset.from_list(normalized_data))

    if concatenate:
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets
    return (dataset, corpus_dict)


def _slice_with_mod(elements: List, offset: int, cnt: int) -> List:
    """Select elements using modulo for cycling through the list.

    This function allows cycling through elements based on an offset (e.g., epoch number).
    When offset exceeds the length of elements, it wraps around using modulo.

    Args:
        elements: List of elements to select from
        offset: Starting offset (e.g., current epoch)
        cnt: Number of elements to select

    Returns:
        List of selected elements

    Example:
        elements = [0, 1, 2], offset = 0, cnt = 1 -> [0]
        elements = [0, 1, 2], offset = 1, cnt = 1 -> [1]
        elements = [0, 1, 2], offset = 3, cnt = 1 -> [0]  # wraps around
    """
    return [elements[(offset + idx) % len(elements)] for idx in range(cnt)]


def _transform_func(examples, num_neg_docs, corpus_dict, current_epoch=0):
    """
    Transform function to convert from raw format to training format.
    Same as _format_process_data in RetrievalMultiModalDatasetLoader.

    Args:
        examples: Batch of examples with question, corpus_id, pos_doc, neg_doc
        num_neg_docs: Number of negative documents to use
        corpus_dict: Dictionary mapping corpus_id to corpus objects
        current_epoch: Current training epoch for cycling through positives
    """
    # Handle both batched and single examples
    is_batched = isinstance(examples["question"], list)

    if not is_batched:
        # Convert single example to batch for uniform processing
        examples = {k: [v] for k, v in examples.items()}

    questions = examples["question"]
    question_ids = examples.get("question_id", [None] * len(questions))
    corpus_ids = examples["corpus_id"]
    batch_positives = examples["pos_doc"]
    batch_negatives = examples["neg_doc"]

    # Check if we have enough negatives
    if num_neg_docs > len(batch_negatives[0]):
        raise Exception(f"num_neg_docs {num_neg_docs} is bigger than 'neg_docs': {len(batch_negatives[0])}")

    cur_pos_neg_doc_batch = []

    for i_example in range(len(questions)):
        cur_pos_neg_doc = []

        # Get one positive doc using epoch-based cycling
        positives = batch_positives[i_example]
        if isinstance(positives, list) and len(positives) > 0:
            pos_ids = list(range(len(positives)))
            # Use modulo to cycle through positives based on epoch
            cur_pos_id = _slice_with_mod(pos_ids, offset=current_epoch, cnt=1)
            selected_pos_idx = cur_pos_id[0]
            cur_pos_neg_doc.append(positives[selected_pos_idx])

            # Log the positive selection for debugging (only log first example per batch to avoid spam)
            if i_example == 0 and len(positives) > 1:
                logging.debug(
                    f"Epoch {current_epoch}: Selected positive index {selected_pos_idx} "
                    f"out of {len(positives)} for question_id {question_ids[i_example]}"
                )
        else:
            cur_pos_neg_doc.append(positives)

        # Get negatives (limit to num_neg_docs)
        negatives = batch_negatives[i_example]
        neg_ids = [i for i in range(len(negatives))]
        cur_neg_ids = neg_ids[:num_neg_docs]
        cur_pos_neg_doc += [negatives[n_id] for n_id in cur_neg_ids]

        cur_pos_neg_doc_batch.append(cur_pos_neg_doc)

    # Extract text and images from corpus
    cur_pos_neg_text_batch = []
    cur_pos_neg_image_batch = []

    for idx_doc, docs in enumerate(cur_pos_neg_doc_batch):
        cur_pos_neg_text = []
        cur_pos_neg_image = []
        cur_corpus_id = corpus_ids[idx_doc]

        for doc in docs:
            cur_id = doc["id"]
            cur_doc = corpus_dict[cur_corpus_id].get_document_by_id(cur_id)

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

    result = {"question": questions, "doc_text": cur_pos_neg_text_batch, "doc_image": cur_pos_neg_image_batch}

    # If input was not batched, return single example
    if not is_batched:
        result = {k: v[0] for k, v in result.items()}

    return result


class RetrievalDatasetWrapper:
    """
    Wrapper for HuggingFace Dataset that supports epoch-based positive document cycling.

    This wrapper tracks the current epoch and uses it to select different positive
    documents for each epoch, cycling through all available positives using modulo.

    Attributes:
        dataset: The underlying HuggingFace Dataset
        corpus_dict: Dictionary mapping corpus_id to corpus objects
        num_neg_docs: Number of negative documents to use
        current_epoch: Current training epoch (used for positive selection)
    """

    def __init__(self, dataset: Dataset, corpus_dict: dict, num_neg_docs: int):
        """
        Initialize the wrapper.

        Args:
            dataset: HuggingFace Dataset with raw examples
            corpus_dict: Dictionary mapping corpus_id to corpus objects
            num_neg_docs: Number of negative documents to use per example
        """
        self._dataset = dataset
        self._corpus_dict = corpus_dict
        self._num_neg_docs = num_neg_docs
        self._current_epoch = 0

        # Set the transform function
        self._dataset.set_transform(self._transform)

    def _transform(self, examples):
        """Transform function that uses current epoch for positive selection."""
        return _transform_func(
            examples,
            num_neg_docs=self._num_neg_docs,
            corpus_dict=self._corpus_dict,
            current_epoch=self._current_epoch,
        )

    def set_epoch(self, epoch: int):
        """
        Set the current epoch for positive document cycling.

        This should be called at the start of each epoch to cycle through
        different positive documents.

        Args:
            epoch: The current epoch number (0-indexed)
        """
        self._current_epoch = epoch
        logging.info(
            f"RetrievalDatasetWrapper: Setting epoch to {epoch} - "
            f"positive documents will be selected using index (epoch % num_positives)"
        )

    @property
    def current_epoch(self) -> int:
        """Get the current epoch."""
        return self._current_epoch

    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self._dataset)

    def __getitem__(self, idx):
        """Get an item from the underlying dataset."""
        return self._dataset[idx]

    def __iter__(self):
        """Iterate over the underlying dataset."""
        return iter(self._dataset)

    @property
    def dataset(self) -> Dataset:
        """Access the underlying HuggingFace Dataset."""
        return self._dataset


def _create_transform_func(num_neg_docs, corpus_dict):
    """Create transform function with specified number of negative documents.

    Note: This function creates a transform that always uses epoch=0.
    For epoch-based cycling, use RetrievalDatasetWrapper instead.
    """

    def transform(examples):
        return _transform_func(examples, num_neg_docs=num_neg_docs, corpus_dict=corpus_dict, current_epoch=0)

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
) -> RetrievalDatasetWrapper:
    """
    Load and return dataset in retrieval format for biencoder training.

    This function loads data from JSON files using the same method as
    RetrievalMultiModalDatasetLoader and returns it ready for training.
    Uses set_transform() for lazy evaluation - tokenization is handled by collator.

    For training datasets, returns a RetrievalDatasetWrapper that supports
    epoch-based positive document cycling. Call wrapper.set_epoch(epoch) at the
    start of each epoch to cycle through different positive documents.

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
        For training: RetrievalDatasetWrapper with epoch-based positive cycling
        For eval: RetrievalDatasetWrapper (epoch is always 0)

        Each example is a dict with keys:
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
        if max_train_samples is not None:
            if do_shuffle:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

        # Wrap dataset for training with epoch-based positive cycling
        negative_size = train_n_passages - 1
        wrapper = RetrievalDatasetWrapper(dataset, corpus_dict, negative_size)
        logging.info(
            f"Created training dataset wrapper with {len(dataset)} examples. "
            f"Call set_epoch(epoch) to cycle through positive documents."
        )

    elif data_type == "eval":
        # Wrap dataset for evaluation (epoch is always 0)
        wrapper = RetrievalDatasetWrapper(dataset, corpus_dict, eval_negative_size)
        logging.info(f"Created eval dataset wrapper with {len(dataset)} examples.")

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    return wrapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and transform dataset to retrieval format")
    parser.add_argument(
        "--data_dir_list", type=str, nargs="+", required=True, help="Path(s) to JSON file(s) containing training data"
    )
    parser.add_argument(
        "--data_type", type=str, default="train", choices=["train", "eval"], help="Type of data (train or eval)"
    )
    parser.add_argument(
        "--train_n_passages", type=int, default=5, help="Number of passages for training (1 positive + n-1 negatives)"
    )
    parser.add_argument(
        "--eval_negative_size", type=int, default=10, help="Number of negative documents for evaluation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--do_shuffle", action="store_true", help="Whether to shuffle the dataset")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset = make_retrieval_dataset(
        data_dir_list=args.data_dir_list,
        data_type=args.data_type,
        train_n_passages=args.train_n_passages,
        eval_negative_size=args.eval_negative_size,
        seed=args.seed,
        do_shuffle=args.do_shuffle,
        max_train_samples=args.max_train_samples,
    )

    print(f"\n{'=' * 60}")
    print(f"Dataset loading completed successfully! (mode: {args.data_type})")
    print(f"{'=' * 60}")
    print(f"Dataset size: {len(dataset)}")
    print("\nSample example:")
    example = dataset[0]
    print(f"Question: {example['question'][:100]}...")
    print(f"Num documents: {len(example['doc_text'])}")
    print(f"Positive doc: {example['doc_text'][0][:100] if example['doc_text'][0] else '(empty)'}...")
    if len(example["doc_text"]) > 1:
        print(f"First negative: {example['doc_text'][1][:100] if example['doc_text'][1] else '(empty)'}...")
    print(f"{'=' * 60}\n")

    # Demo epoch cycling
    if args.data_type == "train":
        print("\nDemo: Epoch-based positive cycling")
        print("-" * 40)
        for epoch in range(3):
            dataset.set_epoch(epoch)
            example = dataset[0]
            pos_doc_preview = example['doc_text'][0][:80] if example['doc_text'][0] else '(empty)'
            print(f"Epoch {epoch}: Positive doc = {pos_doc_preview}...")
        print(f"{'=' * 60}\n")
