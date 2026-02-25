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
from dataclasses import dataclass
from typing import List, Optional, Union

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download

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


class HFCorpusDataset(AbstractDataset):
    """Wraps an already-loaded HuggingFace Dataset as a corpus (in-memory, no local Parquet)."""

    def __init__(self, hf_dataset: Dataset, path: str = ""):
        self.path = path
        self._data = hf_dataset
        self._docid2idx = {str(doc_id): idx for idx, doc_id in enumerate(self._data["id"])}

    def get_document_by_id(self, id):
        example = deepcopy(EXAMPLE_TEMPLATE)
        example["text"] = self._data[self._docid2idx[id]]["text"]
        return example

    def get_all_ids(self):
        return sorted(self._docid2idx.keys())


DATASETS = {
    "TextQADataset": TextQADataset,
}


@dataclass
class CorpusInfo:
    """
    Data structure to hold corpus metadata and dataset object together.
    Provides easy access to both components with descriptive attribute names.
    """

    metadata: dict
    corpus: AbstractDataset

    @property
    def corpus_id(self) -> str:
        """Get corpus ID from metadata"""
        return self.metadata["corpus_id"]

    @property
    def query_instruction(self) -> str:
        """Get query instruction from metadata"""
        if "query_instruction" in self.metadata:
            return self.metadata["query_instruction"]
        else:
            return ""

    @property
    def passage_instruction(self) -> str:
        """Get passage instruction from metadata"""
        if "passage_instruction" in self.metadata:
            return self.metadata["passage_instruction"]
        else:
            return ""

    @property
    def task_type(self) -> str:
        """Get task type from metadata"""
        if "task_type" in self.metadata:
            return self.metadata["task_type"]
        else:
            return ""

    @property
    def path(self) -> str:
        """Get corpus path from the corpus object"""
        return self.corpus.path

    def get_document_by_id(self, doc_id: str):
        """Delegate to corpus for convenience"""
        return self.corpus.get_document_by_id(doc_id)

    def get_all_ids(self):
        """Delegate to corpus for convenience"""
        return self.corpus.get_all_ids()


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
            corpus_dict[corpus_id] = CorpusInfo(corpus_metadata, corpus)


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

        # Resolve relative corpus paths relative to the JSON file's directory
        # This makes the data portable across machines/containers
        json_dir = os.path.dirname(os.path.abspath(data_dir))
        if isinstance(qa_corpus_paths, dict):
            qa_corpus_paths = [qa_corpus_paths]
        for corpus_info in qa_corpus_paths:
            corpus_path = corpus_info.get("path", "")
            if corpus_path and not os.path.isabs(corpus_path):
                corpus_info["path"] = os.path.normpath(os.path.join(json_dir, corpus_path))

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


_HF_PREFIX = "hf://"


def _parse_hf_uri(uri: str):
    """Parse an ``hf://`` URI into ``(repo_id, subset_or_none)``.

    Examples::

        "hf://nvidia/embed-nemotron-dataset-v1/FEVER"  -> ("nvidia/embed-nemotron-dataset-v1", "FEVER")
        "hf://nvidia/embed-nemotron-dataset-v1"         -> ("nvidia/embed-nemotron-dataset-v1", None)
    """
    if not uri.startswith(_HF_PREFIX):
        raise ValueError(f"Not an HF URI (must start with {_HF_PREFIX!r}): {uri}")
    path = uri[len(_HF_PREFIX) :].strip("/")
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"HF URI must contain at least org/repo: {uri}")
    repo_id = f"{parts[0]}/{parts[1]}"
    subset = "/".join(parts[2:]) if len(parts) > 2 else None
    return repo_id, subset


def _list_hf_subsets(repo_id: str) -> List[str]:
    """Discover all subset names in *repo_id* by finding ``dataset_metadata.json`` files."""
    api = HfApi()
    tree = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
    subsets = set()
    for item in tree:
        if item.path.endswith("/dataset_metadata.json"):
            subset_name = os.path.dirname(item.path)
            if subset_name and subset_name != ".":
                subsets.add(subset_name)
    return sorted(subsets)


# ---------------------------------------------------------------------------
# Core HF subset loader
# ---------------------------------------------------------------------------


def _load_hf_subset(repo_id: str, subset: str):
    """Load a single HF subset and return ``(normalized_data_list, CorpusInfo)``."""

    # 1. Download dataset_metadata.json
    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subset}/dataset_metadata.json",
        repo_type="dataset",
    )
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    corpus_id = metadata["corpus_id"]

    # 2. Load corpus
    corpus_hf = load_dataset(repo_id, f"{subset}_corpus", split="train")

    # 3. Build HFCorpusDataset + CorpusInfo
    hf_corpus = HFCorpusDataset(corpus_hf, path=f"hf://{repo_id}/{subset}")
    corpus_info = CorpusInfo(metadata, hf_corpus)

    # 4. Load queries
    queries_hf = load_dataset(repo_id, subset, split="train")

    # 5. Normalize to the standard {question_id, question, corpus_id, pos_doc, neg_doc} shape
    _HF_REQUIRED_FIELDS = ["question", "pos_doc"]
    normalized_data = []
    for idx, item in enumerate(queries_hf):
        missing = [f for f in _HF_REQUIRED_FIELDS if f not in item]
        if missing:
            raise ValueError(f"HF subset {repo_id}/{subset} record {idx} missing required fields: {missing}")
        normalized_item = {
            "question_id": str(item.get("question_id", f"{subset}:{idx}")),
            "question": item["question"],
            "corpus_id": corpus_id,
        }
        # pos_doc
        pos_docs = item["pos_doc"]
        if not isinstance(pos_docs, list):
            pos_docs = [pos_docs]
        if not pos_docs:
            raise ValueError(f"HF subset {repo_id}/{subset} record {idx} has empty pos_doc")
        normalized_item["pos_doc"] = []
        for doc in pos_docs:
            if isinstance(doc, dict) and "id" in doc:
                normalized_item["pos_doc"].append({"id": str(doc["id"])})
            else:
                normalized_item["pos_doc"].append({"id": str(doc)})
        # neg_doc (may be absent or empty — validated later at transform time)
        neg_docs = item.get("neg_doc", [])
        if not isinstance(neg_docs, list):
            neg_docs = [neg_docs]
        normalized_item["neg_doc"] = []
        for doc in neg_docs:
            if isinstance(doc, dict) and "id" in doc:
                normalized_item["neg_doc"].append({"id": str(doc["id"])})
            else:
                normalized_item["neg_doc"].append({"id": str(doc)})
        normalized_data.append(normalized_item)

    return normalized_data, corpus_info


# ---------------------------------------------------------------------------
# Public helpers for data_sources
# ---------------------------------------------------------------------------


def _normalize_data_source(entry):
    """Normalize a data source entry to a dict with at minimum a ``source`` key."""
    if isinstance(entry, str):
        return {"source": entry}
    elif isinstance(entry, dict):
        if "source" not in entry:
            raise ValueError(f"Data source dict must include a 'source' key, got keys: {list(entry.keys())}")
        return entry
    else:
        raise TypeError(f"Data source entry must be a str or dict, got {type(entry).__name__}")


def load_datasets_from_sources(data_sources: list, concatenate: bool = True):
    """Load datasets from a list of data sources (``hf://`` URIs and/or local JSON paths).

    Returns the same ``(dataset, corpus_dict)`` tuple as :func:`load_datasets`.
    """
    hf_data: List[dict] = []
    hf_corpus_dict: dict = {}
    local_paths: List[str] = []

    for entry in data_sources:
        entry = _normalize_data_source(entry)
        source = entry["source"]

        if source.startswith(_HF_PREFIX):
            repo_id, subset = _parse_hf_uri(source)
            subsets = [subset] if subset is not None else _list_hf_subsets(repo_id)

            for sub in subsets:
                logging.info(f"Loading HF subset: {repo_id}/{sub}")
                data_list, corpus_info = _load_hf_subset(repo_id, sub)
                hf_data.extend(data_list)
                if corpus_info.corpus_id in hf_corpus_dict:
                    existing = hf_corpus_dict[corpus_info.corpus_id]
                    if existing.path != corpus_info.path:
                        raise ValueError(
                            f"Duplicate corpus_id '{corpus_info.corpus_id}' with different paths: "
                            f"{existing.path} vs {corpus_info.path}"
                        )
                else:
                    hf_corpus_dict[corpus_info.corpus_id] = corpus_info
        else:
            local_paths.append(source)

    datasets_list = []
    corpus_dict = dict(hf_corpus_dict)

    if hf_data:
        datasets_list.append(Dataset.from_list(hf_data))

    if local_paths:
        local_dataset, local_corpus = load_datasets(local_paths, concatenate=True)
        datasets_list.append(local_dataset)
        for cid, cinfo in local_corpus.items():
            if cid in corpus_dict and corpus_dict[cid].path != cinfo.path:
                raise ValueError(
                    f"Duplicate corpus_id '{cid}' with different paths: {corpus_dict[cid].path} vs {cinfo.path}"
                )
            corpus_dict[cid] = cinfo

    if not datasets_list:
        raise ValueError("No datasets loaded from data_sources")

    if concatenate:
        dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
    else:
        dataset = datasets_list

    return dataset, corpus_dict


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
        else:
            cur_pos_neg_doc.append(positives)

        # Get negatives (limit to num_neg_docs)
        negatives = batch_negatives[i_example]
        if num_neg_docs > 0 and len(negatives) == 0:
            raise ValueError(
                f"neg_doc is empty for example {i_example} but {num_neg_docs} negative(s) requested "
                f"(train_n_passages > 1). Provide negatives or set train_n_passages=1."
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

        if use_dataset_instruction:
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
    data_dir_list: Union[List[str], str] = None,
    data_sources: Optional[List[Union[str, dict]]] = None,
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

    This function loads data from JSON files or HuggingFace ``hf://`` URIs and
    returns it ready for training.  Uses ``set_transform()`` for lazy evaluation —
    tokenization is handled by the collator.

    Args:
        data_dir_list: Path(s) to JSON file(s) containing training data (legacy).
        data_sources: List of ``hf://`` URIs or local paths (preferred).
            Exactly one of *data_dir_list* or *data_sources* must be provided.
        data_type: Type of data ("train" or "eval")
        train_n_passages: Number of passages for training (1 positive + n-1 negatives)
        eval_negative_size: Number of negative documents for evaluation
        seed: Random seed for reproducibility (for shuffling if needed)
        do_shuffle: Whether to shuffle the dataset
        max_train_samples: Maximum number of training samples to use
        train_data_select_offset: Offset for selecting training samples
        use_dataset_instruction: Whether to use instruction from dataset's metadata

    Returns:
        A HuggingFace Dataset where each example is a dict with keys:
        - 'question': Query text
        - 'doc_text': List of document texts [positive, negatives...]
        - 'doc_image': List of images or empty strings

    Note:
        Tokenization should be handled by a collator (e.g., RetrievalBiencoderCollator)
        which is more efficient for batch padding and supports dynamic processing.
    """

    if data_sources is not None and data_dir_list is not None:
        raise ValueError("data_dir_list and data_sources are mutually exclusive; provide one, not both")
    if data_sources is not None:
        logging.info(f"Loading data from {len(data_sources)} source(s)")
        dataset, corpus_dict = load_datasets_from_sources(data_sources)
    elif data_dir_list is not None:
        logging.info(
            f"Loading data from {data_dir_list if isinstance(data_dir_list, str) else len(data_dir_list)} file(s)"
        )
        dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True)
    else:
        raise ValueError("Either data_dir_list or data_sources must be provided")

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
        dataset.set_transform(_create_transform_func(negative_size, corpus_dict, use_dataset_instruction))

    elif data_type == "eval":
        # Set transform for evaluation
        dataset.set_transform(_create_transform_func(eval_negative_size, corpus_dict, use_dataset_instruction))

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and transform dataset to retrieval format")
    parser.add_argument(
        "--data_dir_list", type=str, nargs="+", default=None, help="Path(s) to JSON file(s) containing training data"
    )
    parser.add_argument(
        "--data_sources", type=str, nargs="+", default=None, help="Data source(s): hf:// URIs or local JSON paths"
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
        data_sources=args.data_sources,
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
