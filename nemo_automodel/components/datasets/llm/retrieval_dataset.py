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
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download

EXAMPLE_TEMPLATE = {"text": "", "image": "", "nr_ocr": ""}

_OVERSAMPLING_WARNED_CORPORA: set[str] = set()

_VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")


class AbstractDataset(ABC):
    """Interface for corpus datasets addressable by document id."""

    @abstractmethod
    def get_document_by_id(self, id):
        pass

    @abstractmethod
    def get_all_ids(self):
        pass


class TextQADataset(AbstractDataset):
    """Load TextQA corpus documents from a HuggingFace dataset path."""

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


class ColPaliDataset(AbstractDataset):
    """Load ColPali corpus documents from a dataset path."""

    def __init__(self, path):
        self.path = path
        self.data = load_dataset(path)["train"]
        docid2idx = {}
        for idx, docid in enumerate(self.data["image_filename"]):
            docid2idx[str(docid)] = idx
        self.docid2idx = docid2idx

    def get_document_by_id(self, id):
        example = deepcopy(EXAMPLE_TEMPLATE)
        doc = self.data[self.docid2idx[id]]
        example["image"] = doc["image"]
        if "nr_ocr" in doc:
            example["nr_ocr"] = doc["nr_ocr"]
        if "complex_ocr" in doc:
            example["complex_ocr"] = doc["complex_ocr"]
        return example

    def get_all_ids(self):
        return sorted(list(self.docid2idx.keys()))


class WikiSSNQDataset(AbstractDataset):
    """Load Wiki-SS corpus documents from a dataset path."""

    def __init__(self, path):
        self.path = path
        self.data = load_dataset(path)["train"]
        docid2idx = {}
        for idx, docid in enumerate(self.data["docid"]):
            docid2idx[str(docid)] = idx
        self.docid2idx = docid2idx

    def get_document_by_id(self, id):
        example = deepcopy(EXAMPLE_TEMPLATE)
        doc = self.data[self.docid2idx[id]]
        example["text"] = doc["text"]
        example["image"] = doc["image"]
        if "nr_ocr" in doc:
            example["nr_ocr"] = doc["nr_ocr"]
        if "complex_ocr" in doc:
            example["complex_ocr"] = doc["complex_ocr"]
        return example

    def get_all_ids(self):
        return sorted(list(self.docid2idx.keys()))


class DocMatixDataset(AbstractDataset):
    """Load DocMatix corpus documents from a dataset path."""

    def __init__(self, path):
        self.path = path
        self.data = load_dataset(path, "images")["train"]

    def get_document_by_id(self, id):
        example = deepcopy(EXAMPLE_TEMPLATE)
        example_idx, image_idx = id.split("_")
        example["image"] = self.data[int(example_idx)]["images"][int(image_idx)]
        return example

    def get_all_ids(self):
        return [str(x) + "_" + str(0) for x in list(range(len(self.data)))]


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
    "ColPaliDataset": ColPaliDataset,
    "WikiSSNQDataset": WikiSSNQDataset,
    "DocMatixDataset": DocMatixDataset,
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
    """Load Merlin corpus metadata from a corpus directory."""
    path_metadata = os.path.join(path, "merlin_metadata.json")
    if not os.path.isfile(path_metadata):
        raise ValueError("Metadata File for Corpus does not exist: " + path_metadata)

    with open(path_metadata, "r") as f:
        metadata = json.load(f)
    return metadata


def load_corpus(path, metadata: Optional[dict] = None):
    """Instantiate a corpus dataset from a path and optional metadata."""
    if metadata is None:
        metadata = load_corpus_metadata(path)
    if metadata["class"] not in DATASETS:
        raise ValueError("DatasetClass is not implemented: " + metadata["class"])
    corpus = DATASETS[metadata["class"]](path)
    corpus_id = metadata["corpus_id"]
    return (corpus_id, corpus)


def add_corpus(qa_corpus_paths: Union[dict, list], corpus_dict: dict):
    """Add one or more corpus paths to a corpus dictionary."""
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


DataEntry = Union[str, dict[str, Any]]


def _parse_data_entry(entry: DataEntry) -> Tuple[Optional[int], str]:
    """
    Parse a data entry.

    Supported forms:
    - "path_or_hf_uri": use all samples
    - {"path": "path_or_hf_uri", "num_samples": N}: sample N examples once from that source
    """
    if isinstance(entry, str):
        return None, entry

    if isinstance(entry, dict):
        allowed_keys = {"path", "num_samples"}
        unknown_keys = set(entry) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Unsupported data entry field(s): {sorted(unknown_keys)}")
        if "path" not in entry:
            raise ValueError("data entry dictionary must contain a 'path' field")

        path = entry["path"]
        num_samples = entry.get("num_samples")
        if num_samples is not None:
            if isinstance(num_samples, bool) or not isinstance(num_samples, int):
                raise ValueError(f"num_samples must be an integer or None, got {type(num_samples)}")
            if num_samples < 0:
                raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        if not isinstance(path, str):
            raise ValueError(f"path must be a string, got {type(path)}")
        return num_samples, path

    raise ValueError(f"Invalid data entry format: {entry}. Expected a string path or a dictionary with 'path'")


def _normalize_data_entries(data_dir_list: Union[List[DataEntry], DataEntry]) -> List[Tuple[Optional[int], str]]:
    """Normalize a single source or list of sources into parsed entries."""
    if isinstance(data_dir_list, (str, dict)):
        entries = [data_dir_list]
    elif isinstance(data_dir_list, list):
        entries = data_dir_list
    else:
        raise ValueError(
            f"Invalid data_dir_list format: {data_dir_list}. Expected a string path, a dictionary entry, "
            "or a list of those entries."
        )

    return [entry if isinstance(entry, tuple) else _parse_data_entry(entry) for entry in entries]


def _sample_data_items(data_items: List[dict], num_samples: Optional[int], source: str, seed: int) -> List[dict]:
    if num_samples is None:
        return data_items
    if num_samples >= len(data_items):
        logging.warning(
            f"Requested {num_samples} samples but source {source} only has {len(data_items)} examples. Using all."
        )
        return data_items

    rng = random.Random(seed)
    sampled_items = rng.sample(data_items, num_samples)
    logging.info(f"Randomly sampled {num_samples} examples from {source} (total: {len(data_items)})")
    return sampled_items


def load_datasets(
    data_dir_list: Union[List[DataEntry], DataEntry],
    concatenate: bool = True,
    seed: int = 42,
):
    """
    Load datasets from JSON files.

    Entries can be strings (use all samples) or dictionaries with path and optional
    num_samples fields (sample a fixed subset once while loading).

    Returns:
        Tuple of (dataset, corpus_dict)
    """
    REQUIRED_FIELDS = ["question_id", "question", "corpus_id", "pos_doc", "neg_doc"]
    data_entries = _normalize_data_entries(data_dir_list)
    if not data_entries:
        raise ValueError("data_dir_list must contain at least one source")
    corpus_dict = {}
    datasets = []
    for num_samples, data_dir in data_entries:
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

        data_items = _sample_data_items(train_data["data"], num_samples, data_dir, seed)

        # Extract only the required fields for training, ignoring extra fields
        normalized_data = []
        for item in data_items:
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
            if not item["pos_doc"]:
                raise ValueError(f"pos_doc cannot be empty in train_data item: {item}")
            normalized_item["pos_doc"] = []
            for doc in item["pos_doc"]:
                if isinstance(doc, dict) and "id" in doc:
                    normalized_item["pos_doc"].append({"id": str(doc["id"])})
                else:
                    # Handle case where doc might be just a string ID
                    doc_id = doc if isinstance(doc, str) else str(doc)
                    normalized_item["pos_doc"].append({"id": doc_id})
            # Extract neg_doc with only id field
            normalized_item["neg_doc"] = []
            for doc in item["neg_doc"]:
                if isinstance(doc, dict) and "id" in doc:
                    normalized_item["neg_doc"].append({"id": str(doc["id"])})
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
    """Load a single HF subset and return ``(normalized_data_list, CorpusInfo)``.

    Note:
        The direct ``hf://`` path currently expects the Automodel retrieval schema:
        - ``{subset}/dataset_metadata.json`` with ``corpus_id`` metadata
        - ``{subset}_corpus`` split with corpus columns like ``id`` and ``text``
        - ``{subset}`` split with query columns like ``question`` and ``pos_doc``

        FEVER and SyntheticClassificationData from
        ``nvidia/embed-nemotron-dataset-v1`` are examples that follow this layout.
        Datasets with different structures should use a custom adapter/preprocessor
        before calling this loader.
    """

    # 1. Download dataset_metadata.json
    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{subset}/dataset_metadata.json",
        repo_type="dataset",
    )
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    corpus_id = metadata["corpus_id"]

    if metadata.get("ids_only", False):
        raise ValueError(
            f"HF subset '{repo_id}/{subset}' has ids_only=true in its metadata, meaning "
            f"document and query text must be resolved from an external source before use. "
            f"This is not supported for direct HF loading. Either use a subset that contains "
            f"inline text, or pre-process the dataset with data_preparation.py and load the "
            f"resulting local JSON files via a file path instead."
        )

    # 2. Load corpus
    _CORPUS_REQUIRED_COLS = {"id", "text"}
    corpus_hf = load_dataset(repo_id, f"{subset}_corpus", split="train")

    missing_cols = _CORPUS_REQUIRED_COLS - set(corpus_hf.column_names)
    if missing_cols:
        raise ValueError(
            f"HF corpus dataset '{repo_id}/{subset}_corpus' does not match the expected schema. "
            f"Required columns: {sorted(_CORPUS_REQUIRED_COLS)}, "
            f"found columns: {sorted(corpus_hf.column_names)}. "
            f"Missing: {sorted(missing_cols)}. "
            f"If your dataset uses a different format, implement a custom "
            f"adapter/preprocessor before using direct hf:// loading."
        )

    # 3. Build HFCorpusDataset + CorpusInfo
    hf_corpus = HFCorpusDataset(corpus_hf, path=f"hf://{repo_id}/{subset}")
    corpus_info = CorpusInfo(metadata, hf_corpus)

    # 4. Load queries
    _QUERY_REQUIRED_COLS = {"question", "pos_doc"}
    queries_hf = load_dataset(repo_id, subset, split="train")

    missing_query_cols = _QUERY_REQUIRED_COLS - set(queries_hf.column_names)
    if missing_query_cols:
        raise ValueError(
            f"HF query dataset '{repo_id}/{subset}' does not match the expected schema. "
            f"Required columns: {sorted(_QUERY_REQUIRED_COLS)}, "
            f"found columns: {sorted(queries_hf.column_names)}. "
            f"Missing: {sorted(missing_query_cols)}. "
            f"If your dataset uses a different format, implement a custom "
            f"adapter/preprocessor before using direct hf:// loading."
        )

    # 5. Normalize to the standard {question_id, question, corpus_id, pos_doc, neg_doc} shape
    normalized_data = []
    for idx, item in enumerate(queries_hf):
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


def _load_hf_sources(hf_entries: List[Tuple[Optional[int], str]], seed: int = 42):
    """Load one or more ``hf://`` URIs and return ``(Dataset, corpus_dict)``."""
    hf_data: List[dict] = []
    corpus_dict: dict = {}

    for num_samples, uri in hf_entries:
        repo_id, subset = _parse_hf_uri(uri)
        subsets = [subset] if subset is not None else _list_hf_subsets(repo_id)
        source_data: List[dict] = []

        for sub in subsets:
            logging.info(f"Loading HF subset: {repo_id}/{sub}")
            data_list, corpus_info = _load_hf_subset(repo_id, sub)
            source_data.extend(data_list)
            if corpus_info.corpus_id in corpus_dict:
                existing = corpus_dict[corpus_info.corpus_id]
                if existing.path != corpus_info.path:
                    raise ValueError(
                        f"Duplicate corpus_id '{corpus_info.corpus_id}' with different paths: "
                        f"{existing.path} vs {corpus_info.path}"
                    )
            else:
                corpus_dict[corpus_info.corpus_id] = corpus_info

        hf_data.extend(_sample_data_items(source_data, num_samples, uri, seed))

    return Dataset.from_list(hf_data), corpus_dict


def _transform_func(
    examples,
    num_neg_docs,
    corpus_dict,
    use_dataset_instruction: bool = False,
    epoch: int = 0,
    use_text_in_document: bool = False,
):
    """
    Transform function to convert from raw format to training format.

    Args:
        examples: Batch of examples with question, corpus_id, pos_doc, neg_doc
        num_neg_docs: Number of negative documents to use
        corpus_dict: Dictionary mapping corpus_id to corpus objects
        use_dataset_instruction: Whether to use instruction from dataset's metadata
        epoch: Current epoch for cycling through positive documents
        use_text_in_document: Whether image documents should also include their text
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
    cur_pos_neg_doc_id_batch = []

    for i_example in range(len(questions)):
        cur_pos_neg_doc = []

        # Get one positive doc (cycle through positives based on epoch)
        positives = batch_positives[i_example]
        if isinstance(positives, list) and len(positives) > 0:
            cur_pos_neg_doc.append(positives[epoch % len(positives)])
        else:
            cur_pos_neg_doc.append(positives)

        # Get negatives (limit to num_neg_docs)
        negatives = batch_negatives[i_example]
        if num_neg_docs > 0 and len(negatives) == 0:
            raise ValueError(
                f"neg_doc is empty for example {i_example} but {num_neg_docs} negative(s) requested "
                f"(n_passages > 1). Provide negatives."
            )
        cur_corpus_id = corpus_ids[i_example]
        if num_neg_docs > 0 and len(negatives) < num_neg_docs and cur_corpus_id not in _OVERSAMPLING_WARNED_CORPORA:
            _OVERSAMPLING_WARNED_CORPORA.add(cur_corpus_id)
            logging.warning(
                f"corpus_id={cur_corpus_id}: a sample has only {len(negatives)} negatives "
                f"(< num_neg_docs={num_neg_docs}). Oversampling will repeat negatives. "
                "This warning is logged once per corpus."
            )
        if num_neg_docs > 0:
            neg_ids = [i for i in range(len(negatives))]
            cur_neg_ids = [neg_ids[idx % len(neg_ids)] for idx in range(num_neg_docs)]
            cur_pos_neg_doc += [negatives[n_id] for n_id in cur_neg_ids]

        cur_pos_neg_doc_batch.append(cur_pos_neg_doc)
        cur_pos_neg_doc_id_batch.append(
            [d["id"] if isinstance(d, dict) and "id" in d else str(d) for d in cur_pos_neg_doc]
        )

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
            if isinstance(doc, dict) and "text" in doc:
                # Inline-text records (.jsonl/.tsv/.csv) keep document content
                # directly in pos_doc / neg_doc.
                cur_doc = {
                    "text": "" if doc.get("text") is None else str(doc.get("text", "")),
                    "image": doc.get("image", ""),
                    "nr_ocr": "" if doc.get("nr_ocr") is None else str(doc.get("nr_ocr", "")),
                }
            else:
                if cur_corpus_id not in corpus_dict:
                    raise ValueError(
                        f"Unknown corpus_id '{cur_corpus_id}' in retrieval example. "
                        f"Available corpus ids: {sorted(corpus_dict.keys())}"
                    )
                cur_id = doc["id"] if isinstance(doc, dict) else str(doc)
                cur_doc = corpus_dict[cur_corpus_id].get_document_by_id(cur_id)

            # Extract text
            if cur_doc["text"] != "" and not cur_doc["image"]:
                text = cur_doc["text"]
            elif use_text_in_document and cur_doc["image"]:
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
        "doc_id": cur_pos_neg_doc_id_batch,
        "query_instruction": query_instruction_batch,
        "passage_instruction": passage_instruction_batch,
    }

    # If input was not batched, return single example
    if not is_batched:
        result = {k: v[0] for k, v in result.items()}

    return result


def _cross_encoder_transform_func(
    examples,
    num_neg_docs,
    corpus_dict,
    use_dataset_instruction: bool = False,
    epoch: int = 0,
    use_text_in_document: bool = False,
):
    """
    Transform function to convert from raw format to cross-encoder training format.
    """
    from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import flatten_bi_encoder_to_cross_encoder

    data = _transform_func(
        examples,
        num_neg_docs,
        corpus_dict,
        use_dataset_instruction,
        epoch=epoch,
        use_text_in_document=use_text_in_document,
    )
    return flatten_bi_encoder_to_cross_encoder(data)


class RetrievalTransform:
    """Stateful transform for retrieval datasets with epoch-based positive cycling.

    This class encapsulates the transform state (epoch, corpus_dict, etc.) and
    provides a clean interface for updating the epoch without recreating the transform.
    """

    def __init__(
        self,
        num_neg_docs: int,
        corpus_dict: dict,
        use_dataset_instruction: bool = False,
        model_type: str = "bi_encoder",
        cycle_positive_docs: bool = False,
        use_text_in_document: bool = False,
    ):
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
        self.num_neg_docs = num_neg_docs
        self.corpus_dict = corpus_dict
        self.use_dataset_instruction = use_dataset_instruction
        self.model_type = model_type
        self.cycle_positive_docs = cycle_positive_docs
        self.use_text_in_document = use_text_in_document
        self.epoch = 0

    def __call__(self, examples):
        epoch = self.epoch if self.cycle_positive_docs else 0
        if self.model_type == "cross_encoder":
            return _cross_encoder_transform_func(
                examples,
                num_neg_docs=self.num_neg_docs,
                corpus_dict=self.corpus_dict,
                use_dataset_instruction=self.use_dataset_instruction,
                epoch=epoch,
                use_text_in_document=self.use_text_in_document,
            )
        return _transform_func(
            examples,
            num_neg_docs=self.num_neg_docs,
            corpus_dict=self.corpus_dict,
            use_dataset_instruction=self.use_dataset_instruction,
            epoch=epoch,
            use_text_in_document=self.use_text_in_document,
        )

    def set_epoch(self, epoch: int):
        """Update the epoch for positive document cycling."""
        self.epoch = epoch


def make_retrieval_dataset(
    data_dir_list: Union[List[DataEntry], DataEntry] = None,
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int = 5,
    eval_negative_size: int = None,
    seed: int = 42,
    do_shuffle: bool = False,
    max_train_samples: int = None,
    train_data_select_offset: int = 0,
    use_dataset_instruction: bool = False,
    cycle_positive_docs: bool = False,
    use_text_in_document: bool = False,
):
    """
    Load and return dataset in retrieval format for encoder training.

    Entries in *data_dir_list* can be local JSON file paths **or** ``hf://`` URIs
    pointing to a HuggingFace dataset repository (e.g.
    ``hf://nvidia/embed-nemotron-dataset-v1/SciFact``). A source can also be
    provided as ``{"path": path_or_uri, "num_samples": N}`` to sample a fixed
    subset once while loading. Uses ``set_transform()`` for lazy evaluation —
    tokenization is handled by the collator.

    Args:
        data_dir_list: Path(s) to JSON file(s), ``hf://`` URIs, or dictionary entries with path and
            num_samples.
        model_type: "bi_encoder" (default) or "cross_encoder"
        data_type: Type of data ("train" or "eval")
        n_passages: Number of passages (1 positive + n-1 negatives)
        eval_negative_size: Number of negative documents for evaluation
        seed: Random seed for reproducibility (for shuffling if needed)
        do_shuffle: Shuffle dataset rows before subset selection. Only applied when
            ``max_train_samples`` is set; otherwise iteration order is controlled by
            the dataloader's sampler (e.g. ``StatefulDistributedSampler``).
        max_train_samples: Maximum number of training samples to use
        train_data_select_offset: Offset for selecting training samples
        use_dataset_instruction: Whether to use instruction from dataset's metadata
        cycle_positive_docs: Whether training should cycle through positive documents across epochs.
            Defaults to ``False`` (always use the first positive document). Set to ``True`` only
            when a query has multiple positive documents and you want to rotate through them by epoch.
        use_text_in_document: Whether image documents should also include their text.

    Returns:
        A HuggingFace Dataset where each example is a dict with keys:
        - 'question': Query text
        - 'doc_text': List of document texts [positive, negatives...]
        - 'doc_image': List of images or empty strings

    Note:
        Direct ``hf://`` loading currently supports HF datasets that already follow
        the Automodel retrieval schema (corpus-id based layout used by
        ``nvidia/embed-nemotron-dataset-v1`` subsets such as FEVER and
        SyntheticClassificationData). For other HF dataset formats, implement a
        custom adapter/preprocessor before calling this loader.

        Tokenization should be handled by a collator (e.g., BiEncoderCollator)
        which is more efficient for batch padding and supports dynamic processing.
    """

    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")

    if data_dir_list is None:
        raise ValueError("data_dir_list is required")
    data_entries = _normalize_data_entries(data_dir_list)
    if not data_entries:
        raise ValueError("data_dir_list must contain at least one source")

    hf_entries = [(num_samples, path) for num_samples, path in data_entries if path.startswith(_HF_PREFIX)]
    local_entries = [(num_samples, path) for num_samples, path in data_entries if not path.startswith(_HF_PREFIX)]

    logging.info(f"Loading data from {len(data_entries)} source(s) ({len(hf_entries)} HF, {len(local_entries)} local)")

    datasets_list = []
    corpus_dict: dict = {}

    if hf_entries:
        hf_dataset, hf_corpus = _load_hf_sources(hf_entries, seed=seed)
        datasets_list.append(hf_dataset)
        corpus_dict.update(hf_corpus)

    if local_entries:
        corpus_local_entries: list[tuple[Optional[int], str]] = []
        inline_local_entries: list[tuple[Optional[int], str]] = []
        for num_samples, local_path in local_entries:
            suffix = Path(local_path).suffix.lower()
            if suffix in {".jsonl", ".tsv", ".csv"}:
                inline_local_entries.append((num_samples, local_path))
            else:
                corpus_local_entries.append((num_samples, local_path))

        if corpus_local_entries:
            local_dataset, local_corpus = load_datasets(corpus_local_entries, concatenate=True, seed=seed)
            datasets_list.append(local_dataset)
            for cid, cinfo in local_corpus.items():
                if cid in corpus_dict and corpus_dict[cid].path != cinfo.path:
                    raise ValueError(
                        f"Duplicate corpus_id '{cid}' with different paths: {corpus_dict[cid].path} vs {cinfo.path}"
                    )
                corpus_dict[cid] = cinfo

        if inline_local_entries:
            from nemo_automodel.components.datasets.llm import retrieval_dataset_inline

            inline_rows: list[dict] = []
            for num_samples, local_path in inline_local_entries:
                inline_dataset, _ = retrieval_dataset_inline.load_datasets(local_path, concatenate=True)
                sampled_rows = _sample_data_items(inline_dataset.to_list(), num_samples, local_path, seed)
                inline_rows.extend(sampled_rows)

            if inline_rows:
                datasets_list.append(Dataset.from_list(inline_rows))

    dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]

    logging.info(f"Loaded dataset with {len(dataset)} examples")

    if data_type == "train":
        if max_train_samples is not None:
            if do_shuffle:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

        negative_size = n_passages - 1
        transform = RetrievalTransform(
            negative_size,
            corpus_dict,
            use_dataset_instruction,
            model_type,
            cycle_positive_docs,
            use_text_in_document,
        )
        dataset.set_transform(transform)
        if cycle_positive_docs:
            # NOTE: set_epoch is monkey-patched onto this Dataset instance. It is only
            # reachable as long as the recipe uses this exact object. If the dataset is
            # later wrapped/copied (e.g. IterableDataset.from_generator), re-expose
            # set_epoch on the wrapper or move the epoch into a dataset wrapper class.
            dataset.set_epoch = transform.set_epoch

    elif data_type == "eval":
        if eval_negative_size is None:
            eval_negative_size = n_passages - 1
        transform = RetrievalTransform(
            eval_negative_size,
            corpus_dict,
            use_dataset_instruction,
            model_type,
            cycle_positive_docs,
            use_text_in_document,
        )
        dataset.set_transform(transform)

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")

    return dataset


@dataclass
class RetrievalDatasetConfig:
    """Construction-time configuration for the retrieval dataset."""

    data_dir_list: list[DataEntry] | DataEntry | None = None
    """Path(s) to JSON file(s), ``hf://`` URIs, or dict entries with path and num_samples."""
    model_type: str = "bi_encoder"
    """``bi_encoder`` or ``cross_encoder``."""
    data_type: str = "train"
    """Type of data (``train`` or ``eval``)."""
    n_passages: int = 5
    """Number of passages (1 positive + n-1 negatives)."""
    eval_negative_size: int | None = None
    """Number of negative documents for evaluation."""
    seed: int = 42
    """Random seed for shuffling / sampling."""
    do_shuffle: bool = False
    """Shuffle dataset rows before subset selection (only when ``max_train_samples`` is set)."""
    max_train_samples: int | None = None
    """Maximum number of training samples to use."""
    train_data_select_offset: int = 0
    """Offset for selecting training samples."""
    use_dataset_instruction: bool = False
    """Whether to use the instruction from the dataset's metadata."""
    cycle_positive_docs: bool = False
    """Whether to rotate through multiple positive documents by epoch during training."""
    use_text_in_document: bool = False
    """Whether image documents should also include their text."""

    def build(self) -> Dataset:
        """Build the retrieval :class:`~datasets.Dataset` from this :class:`RetrievalDatasetConfig`."""
        return make_retrieval_dataset(
            data_dir_list=self.data_dir_list,
            model_type=self.model_type,
            data_type=self.data_type,
            n_passages=self.n_passages,
            eval_negative_size=self.eval_negative_size,
            seed=self.seed,
            do_shuffle=self.do_shuffle,
            max_train_samples=self.max_train_samples,
            train_data_select_offset=self.train_data_select_offset,
            use_dataset_instruction=self.use_dataset_instruction,
            cycle_positive_docs=self.cycle_positive_docs,
            use_text_in_document=self.use_text_in_document,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and transform dataset to retrieval format")
    parser.add_argument(
        "--data_dir_list",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to JSON file(s) or hf:// URIs",
    )
    parser.add_argument(
        "--data_type", type=str, default="train", choices=["train", "eval"], help="Type of data (train or eval)"
    )
    parser.add_argument("--n_passages", type=int, default=5, help="Number of passages (1 positive + n-1 negatives)")
    parser.add_argument(
        "--eval_negative_size",
        type=int,
        default=None,
        help="Number of negative documents for evaluation (default: n_passages - 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--do_shuffle", action="store_true", help="Whether to shuffle the dataset")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset = make_retrieval_dataset(
        data_dir_list=args.data_dir_list,
        data_type=args.data_type,
        n_passages=args.n_passages,
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
