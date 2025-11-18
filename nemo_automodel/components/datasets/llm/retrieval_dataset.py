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


def _transform_func(examples, num_neg_docs, corpus_dict):
    """
    Transform function to convert from raw format to training format.
    Same as _format_process_data in RetrievalMultiModalDatasetLoader.

    Args:
        examples: Batch of examples with question, corpus_id, pos_doc, neg_doc
        num_neg_docs: Number of negative documents to use
        corpus_dict: Dictionary mapping corpus_id to corpus objects
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

    # Check if we have enough negatives
    if num_neg_docs > len(batch_negatives[0]):
        raise Exception(f"num_neg_docs {num_neg_docs} is bigger than 'neg_docs': {len(batch_negatives[0])}")

    cur_pos_neg_doc_batch = []

    for i_example in range(len(questions)):
        cur_pos_neg_doc = []

        # Get one positive doc (take first one)
        positives = batch_positives[i_example]
        if isinstance(positives, list) and len(positives) > 0:
            cur_pos_neg_doc.append(positives[0])
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


def _create_transform_func(num_neg_docs, corpus_dict):
    """Create transform function with specified number of negative documents."""

    def transform(examples):
        return _transform_func(examples, num_neg_docs=num_neg_docs, corpus_dict=corpus_dict)

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
        if max_train_samples is not None:
            if do_shuffle:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

        # Set transform for training (train_n_passages - 1 negatives)
        negative_size = train_n_passages - 1
        dataset.set_transform(_create_transform_func(negative_size, corpus_dict))

    elif data_type == "eval":
        # Set transform for evaluation
        dataset.set_transform(_create_transform_func(eval_negative_size, corpus_dict))

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")

    return dataset


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
