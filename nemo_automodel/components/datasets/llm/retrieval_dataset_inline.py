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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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


def load_datasets(data_dir_list: Union[List[str], str], concatenate: bool = True,
                  extra_columns: Optional[tuple[str, ...]] = None):
    """
    Load retrieval datasets from JSON/JSONL files.

    Copied from nemo-retriever-research/src/data/datasets.py

    Returns:
        Tuple of (dataset, corpus_dict)

    Columns named in *extra_columns* are carried through verbatim; every other key
    outside the normalized set is dropped. Absent values become None so the column
    stays present on every row.
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
            for column in extra_columns or ():
                normalized_item[column] = item.get(column)
            normalized_data.append(normalized_item)

        datasets.append(Dataset.from_list(normalized_data))

    if concatenate:
        dataset = concatenate_datasets(datasets)
    else:
        dataset = datasets
    return (dataset, {})


def _retrieval_transform_func(examples, num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """
    Transform function to convert from raw format to training format.
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


def flatten_bi_encoder_to_cross_encoder(data: dict) -> dict:
    """Flatten grouped bi-encoder output into cross-encoder format.

    Takes bi-encoder-style data (queries with grouped doc lists) and flattens it
    so each query-doc pair becomes a separate entry. Used by cross-encoder transforms
    in both retrieval_dataset.py and retrieval_dataset_inline.py.
    """
    cur_pos_neg_image_batch = data["doc_image"]
    cur_pos_neg_text_batch = data["doc_text"]
    questions = data["question"]

    # Flattening query-grouped docs images and text and repeating queries
    cur_pos_neg_image_batch_flatten = [y for x in cur_pos_neg_image_batch for y in x]
    cur_pos_neg_text_batch_flatten = [y for x in cur_pos_neg_text_batch for y in x]
    questions_repeated = [[q] * len(i) for q, i in zip(questions, cur_pos_neg_image_batch)]
    questions_repeated_flatten = [y for x in questions_repeated for y in x]
    num_labels = len(questions)

    assert (
        len(cur_pos_neg_image_batch_flatten) == len(cur_pos_neg_text_batch_flatten) == len(questions_repeated_flatten)
    )
    return {
        "doc_image": cur_pos_neg_image_batch_flatten,
        "doc_text": cur_pos_neg_text_batch_flatten,
        "question": questions_repeated_flatten,
        # Only necessary for training. Collator might use it to create the labels with the right shape
        "num_labels": [num_labels] * len(questions_repeated_flatten),
    }


def _cross_encoder_transform_func(examples, num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """
    Transform function to convert from raw format to cross-encoder training format.
    """
    data = _retrieval_transform_func(examples, num_neg_docs, corpus_dict, use_dataset_instruction)
    return flatten_bi_encoder_to_cross_encoder(data)


def _create_retrieval_transform_func(num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """Create transform function with specified number of negative documents."""

    def transform(examples):
        return _retrieval_transform_func(
            examples,
            num_neg_docs=num_neg_docs,
            corpus_dict=corpus_dict,
            use_dataset_instruction=use_dataset_instruction,
        )

    return transform


def _create_cross_encoder_transform_func(num_neg_docs, corpus_dict, use_dataset_instruction: bool = False):
    """Create transform function with specified number of negative documents."""

    def transform(examples):
        return _cross_encoder_transform_func(
            examples,
            num_neg_docs=num_neg_docs,
            corpus_dict=corpus_dict,
            use_dataset_instruction=use_dataset_instruction,
        )

    return transform


def make_retrieval_dataset(
    data_dir_list: Union[List[str], str],
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int = 5,
    eval_negative_size: int = None,
    seed: int = 42,
    do_shuffle: bool = False,
    max_train_samples: int = None,
    train_data_select_offset: int = 0,
    use_dataset_instruction: bool = False,
):
    """
    Load and return dataset in retrieval format for encoder training.

    This function loads data from JSON files and returns it ready for training.
    Uses set_transform() for lazy evaluation - tokenization is handled by collator.

    Args:
        data_dir_list: Path(s) to JSON file(s) containing training data
        model_type: "bi_encoder" (default) or "cross_encoder"
        data_type: Type of data ("train" or "eval")
        n_passages: Number of passages (1 positive + n-1 negatives)
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
        Tokenization should be handled by a collator (e.g., BiEncoderCollator)
        which is more efficient for batch padding and supports dynamic processing.
    """

    _VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")

    logging.info(f"Loading data from {data_dir_list if isinstance(data_dir_list, str) else len(data_dir_list)} file(s)")

    # Load datasets from JSON files
    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True)

    logging.info(f"Loaded dataset with {len(dataset)} examples")

    if model_type == "cross_encoder":
        transform_factory = _create_cross_encoder_transform_func
    else:
        transform_factory = _create_retrieval_transform_func

    if data_type == "train":
        if do_shuffle:
            dataset = dataset.shuffle(seed=seed)
        if max_train_samples is not None:
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

        negative_size = n_passages - 1
        dataset.set_transform(transform_factory(negative_size, corpus_dict, use_dataset_instruction))

    elif data_type == "eval":
        if eval_negative_size is None:
            eval_negative_size = n_passages - 1
        dataset.set_transform(transform_factory(eval_negative_size, corpus_dict, use_dataset_instruction))

    else:
        raise ValueError(f"Invalid data type: {data_type}")

    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")

    return dataset


def _flatten_context_columns(data: dict, context_columns: tuple[str, ...]) -> dict:
    """Flatten a bi-encoder batch and repeat the extra per-query columns per document.

    ``flatten_bi_encoder_to_cross_encoder`` returns a fixed set of keys, so any column
    beyond question/doc_text is dropped. Context fields are per-query, so they repeat
    exactly like the question does.
    """
    flattened = flatten_bi_encoder_to_cross_encoder(data)
    docs_per_query = [len(group) for group in data["doc_image"]]
    for column in context_columns:
        values = data.get(column)
        if values is None:
            continue
        flattened[column] = [v for v, n in zip(values, docs_per_query) for _ in range(n)]
    return flattened


def _group_aware_split(dataset, validation_fraction: float, group_key: str | None,
                       data_type: str, seed: int):
    """Carve a deterministic held-out slice, keeping rows that share a group together.

    Splitting on rows alone leaks when one query contributes several rows -- a mixed
    dataset holding two labelings of the same query is the case that motivated this.
    Grouping on ``group_key`` puts every row of a group on the same side.
    """
    import random

    if group_key is None:
        groups = list(range(len(dataset)))
        row_groups = groups
    else:
        if group_key not in dataset.column_names:
            raise ValueError(
                f"validation_group_key={group_key!r} is not a column of the dataset; "
                f"available columns are {sorted(dataset.column_names)}. The split key must be "
                "requested as an extra column so it survives loading."
            )
        row_groups = dataset[group_key]
        # load_datasets fills absent extra columns with None, so a group key that is present
        # on some rows and missing on others reaches here as a mix of str and None. sorted()
        # then raises TypeError from inside dataset construction, which reads as a library
        # bug rather than as the data problem it is. Fail with the column name instead.
        missing = sum(1 for g in row_groups if g is None or (isinstance(g, str) and not g.strip()))
        if missing:
            raise ValueError(
                f"validation_group_key={group_key!r} is missing or blank on {missing} of "
                f"{len(row_groups)} rows; every row needs a group value or the split cannot "
                "keep a group on one side"
            )
        groups = sorted(set(row_groups))

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * validation_fraction))
    val_groups = set(shuffled[len(shuffled) - n_val:]) if n_val else set()

    keep_val = data_type in ("validation", "eval")
    indices = [i for i, g in enumerate(row_groups) if (g in val_groups) == keep_val]
    logging.info(
        "group-aware split on %r: %d groups -> %d validation, %d rows selected for %s",
        group_key, len(groups), len(val_groups), len(indices), data_type,
    )
    return dataset.select(indices)


def make_context_aware_retrieval_dataset(
    data_dir_list: Union[List[str], str],
    model_type: str = "cross_encoder",
    data_type: str = "train",
    n_passages: int = 8,
    validation_fraction: float = 0.0,
    validation_group_key: Optional[str] = None,
    reasoning_column: Optional[str] = None,
    global_query_column: Optional[str] = None,
    seed: int = 42,
    do_shuffle: bool = False,
    max_train_samples: Optional[int] = None,
    train_data_select_offset: int = 0,
):
    """Inline retrieval dataset that also carries per-query context columns.

    Same ``pos_doc``/``neg_doc`` schema and loader as :func:`make_retrieval_dataset`,
    plus two things it does not provide:

    * ``reasoning_column`` / ``global_query_column`` are passed through as ``reasoning``
      and ``global_query``. ``Qwen3RerankerCollator`` reads both with ``.get()`` and
      selects its prompt mode from whichever survive its drop probabilities, so rows
      missing them simply train in a narrower mode.
    * ``validation_fraction`` carves a held-out slice at the level of
      ``validation_group_key`` rather than the row, so rows sharing a group cannot land
      on opposite sides.

    Args:
        data_dir_list: Path(s) to inline JSON/JSONL with ``query``/``pos_doc``/``neg_doc``.
        model_type: ``"cross_encoder"`` or ``"bi_encoder"``.
        data_type: ``"train"``, or ``"validation"``/``"eval"`` for the held-out side.
        n_passages: Passages per query (1 positive + ``n_passages - 1`` negatives).
        validation_fraction: Fraction of groups held out. 0 uses the whole split.
        validation_group_key: Column defining a group; None groups by row.
        reasoning_column: Column holding the reasoning trace.
        global_query_column: Column holding the originating question.
        seed: Seeds the split and any shuffle.
        do_shuffle: Shuffle before subsetting (train only).
        max_train_samples: Cap on training rows, applied after the split.
        train_data_select_offset: Offset of the selected window.

    Returns:
        A ``Dataset`` whose transform emits question/doc_text plus the context columns.
    """
    _VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
    if data_type not in ("train", "validation", "eval"):
        raise ValueError(f"Invalid data type: {data_type}")

    requested = tuple(
        c for c in (reasoning_column, global_query_column, validation_group_key) if c
    )
    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True, extra_columns=requested)
    logging.info(f"Loaded dataset with {len(dataset)} examples")

    if validation_fraction > 0:
        dataset = _group_aware_split(dataset, validation_fraction, validation_group_key, data_type, seed)

    if data_type == "train":
        if do_shuffle:
            dataset = dataset.shuffle(seed=seed)
        if max_train_samples is not None:
            dataset = dataset.select(
                range(train_data_select_offset, min(train_data_select_offset + max_train_samples, len(dataset)))
            )

    context_columns = tuple(
        c for c in ((reasoning_column, "reasoning"), (global_query_column, "global_query")) if c[0]
    )
    negative_size = n_passages - 1

    def transform(examples):
        data = _retrieval_transform_func(examples, negative_size, corpus_dict)
        for source, target in context_columns:
            if source in examples:
                data[target] = examples[source]
        if model_type == "bi_encoder":
            return data
        return _flatten_context_columns(data, tuple(t for _, t in context_columns))

    dataset.set_transform(transform)
    logging.info(f"Created {data_type} dataset with {len(dataset)} examples")
    return dataset


@dataclass
class InlineRetrievalDatasetConfig:
    """Construction-time configuration for inline retrieval datasets."""

    data_dir_list: list[str] | str
    model_type: str = "bi_encoder"
    data_type: str = "train"
    n_passages: int = 5
    eval_negative_size: int | None = None
    seed: int = 42
    do_shuffle: bool = False
    max_train_samples: int | None = None
    train_data_select_offset: int = 0
    use_dataset_instruction: bool = False

    def build(self) -> Dataset:
        """Build the inline retrieval dataset from this config."""
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
        )


@dataclass
class ContextAwareRetrievalDatasetConfig:
    """Construction-time configuration for context-aware inline retrieval datasets."""

    data_dir_list: list[str] | str
    model_type: str = "cross_encoder"
    data_type: str = "train"
    n_passages: int = 8
    validation_fraction: float = 0.0
    validation_group_key: str | None = None
    reasoning_column: str | None = None
    global_query_column: str | None = None
    seed: int = 42
    do_shuffle: bool = False
    max_train_samples: int | None = None
    train_data_select_offset: int = 0

    def build(self) -> Dataset:
        """Build the context-aware retrieval dataset from this config."""
        return make_context_aware_retrieval_dataset(
            data_dir_list=self.data_dir_list,
            model_type=self.model_type,
            data_type=self.data_type,
            n_passages=self.n_passages,
            validation_fraction=self.validation_fraction,
            validation_group_key=self.validation_group_key,
            reasoning_column=self.reasoning_column,
            global_query_column=self.global_query_column,
            seed=self.seed,
            do_shuffle=self.do_shuffle,
            max_train_samples=self.max_train_samples,
            train_data_select_offset=self.train_data_select_offset,
        )
