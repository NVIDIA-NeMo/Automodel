# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Normalized portable retrieval dataset backed by local Arrow shards.

This format keeps the original retrieval data model:

- train rows contain query text plus positive/negative document ids;
- corpus shards store each referenced document/image once;
- training still resolves ``doc_id -> document`` through the retrieval transform.

It is intended as a portable alternative to the original corpus-id JSON plus
external Hugging Face corpus paths, not as a resolved-row format.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from nemo_automodel.components.datasets.llm.retrieval_dataset import (
    EXAMPLE_TEMPLATE,
    AbstractDataset,
    CorpusInfo,
    RetrievalTransform,
)
from nemo_automodel.shared.import_utils import safe_import

_VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")

logger = logging.getLogger(__name__)


def _import_datasets():
    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to read normalized retrieval Arrow shards. Install datasets.")
    return datasets


def _coerce_bundle_root(data_path: str | Path | os.PathLike[str]) -> Path:
    path = Path(data_path)
    if path.is_file() and path.name == "metadata.json":
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Normalized retrieval bundle does not exist: {path}")
    return path


def _parse_data_entries(data_path: str | Path | os.PathLike[str] | Sequence[Any]) -> list[tuple[Path, int | None]]:
    if isinstance(data_path, (str, Path, os.PathLike)):
        return [(_coerce_bundle_root(data_path), None)]
    if isinstance(data_path, dict):
        entries = [data_path]
    else:
        entries = list(data_path)

    if not entries:
        raise ValueError("Normalized retrieval data expects at least one bundle directory")

    parsed = []
    for entry in entries:
        if isinstance(entry, (str, Path, os.PathLike)):
            parsed.append((_coerce_bundle_root(entry), None))
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                "Normalized retrieval data entries must be paths or dictionaries with 'path' and optional "
                f"'num_samples', got {type(entry).__name__}"
            )
        allowed_keys = {"path", "num_samples"}
        unknown_keys = set(entry) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Unsupported normalized retrieval data entry field(s): {sorted(unknown_keys)}")
        if "path" not in entry:
            raise ValueError("Normalized retrieval data entry dictionaries must contain a 'path' field")
        num_samples = entry.get("num_samples")
        if num_samples is not None:
            if isinstance(num_samples, bool) or not isinstance(num_samples, int):
                raise ValueError(f"num_samples must be an integer or None, got {type(num_samples).__name__}")
            if num_samples < 1:
                raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        parsed.append((_coerce_bundle_root(entry["path"]), num_samples))
    return parsed


def _load_arrow_shards(paths: Sequence[Path]):
    datasets = _import_datasets()
    shards = [datasets.Dataset.from_file(str(path)) for path in paths]
    if not shards:
        raise ValueError("No Arrow shards were provided")
    return shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)


def _decode_image_bytes(image_bytes: bytes | None) -> Any:
    if not image_bytes:
        return ""
    has_pil, image_module = safe_import("PIL.Image")
    if not has_pil:
        raise ImportError("Pillow is required to decode normalized retrieval images. Install pillow.")
    with image_module.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _parse_doc_refs(value: Any, *, field_name: str) -> list[dict[str, str]]:
    if isinstance(value, str):
        refs = json.loads(value)
    else:
        refs = value
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise ValueError(f"Normalized retrieval field '{field_name}' must be a list, got {type(refs).__name__}")
    normalized = []
    for doc in refs:
        if isinstance(doc, dict) and "id" in doc:
            normalized.append({"id": str(doc["id"])})
        else:
            normalized.append({"id": str(doc)})
    return normalized


def _parse_json_ref_column(value: Any, *, field_name: str) -> Any:
    if isinstance(value, list):
        return [_parse_doc_refs(item, field_name=field_name) for item in value]
    return _parse_doc_refs(value, field_name=field_name)


class _JsonRefRetrievalTransform:
    """Adapter for version-1 bundles whose train refs were stored as JSON strings."""

    def __init__(self, transform: RetrievalTransform) -> None:
        self._transform = transform

    def __call__(self, examples):
        examples = dict(examples)
        examples["pos_doc"] = _parse_json_ref_column(examples.pop("pos_doc_json"), field_name="pos_doc")
        examples["neg_doc"] = _parse_json_ref_column(examples.pop("neg_doc_json"), field_name="neg_doc")
        return self._transform(examples)

    def set_epoch(self, epoch: int) -> None:
        self._transform.set_epoch(epoch)


class NormalizedArrowCorpusDataset(AbstractDataset):
    """Local Arrow corpus addressable by document id."""

    def __init__(self, dataset: Any, path: str) -> None:
        self.path = path
        self.data = dataset
        self.docid2idx = {str(doc_id): idx for idx, doc_id in enumerate(self.data["id"])}

    def get_document_by_id(self, id):
        row = self.data[self.docid2idx[str(id)]]
        example = deepcopy(EXAMPLE_TEMPLATE)
        example["text"] = "" if row.get("text") is None else str(row.get("text"))
        example["image"] = _decode_image_bytes(row.get("image_jpeg"))
        if row.get("nr_ocr") is not None:
            example["nr_ocr"] = str(row.get("nr_ocr"))
        if row.get("complex_ocr") is not None:
            example["complex_ocr"] = str(row.get("complex_ocr"))
        return example

    def get_all_ids(self):
        return sorted(self.docid2idx.keys())


def _load_metadata(bundle_root: Path) -> dict[str, Any]:
    metadata_path = bundle_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Normalized retrieval metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "nemo_automodel_normalized_vl_retrieval_arrow":
        raise ValueError(f"Unsupported normalized retrieval format: {metadata.get('format')!r}")
    return metadata


def _load_corpus_dict(bundle_root: Path, metadata: dict[str, Any]) -> dict[str, CorpusInfo]:
    corpus_dict = {}
    for corpus_meta in metadata.get("corpora", []):
        corpus_id = corpus_meta["corpus_id"]
        corpus_paths = [bundle_root / shard for shard in corpus_meta.get("shards", [])]
        corpus_dataset = _load_arrow_shards(corpus_paths)
        corpus_metadata = dict(corpus_meta.get("metadata", {}))
        corpus_metadata.setdefault("corpus_id", corpus_id)
        corpus_path = str(corpus_paths[0].parent) if corpus_paths else str(bundle_root)
        corpus_dict[corpus_id] = CorpusInfo(
            corpus_metadata,
            NormalizedArrowCorpusDataset(corpus_dataset, path=corpus_path),
        )
    if not corpus_dict:
        raise ValueError(f"Normalized retrieval bundle has no corpora: {bundle_root}")
    return corpus_dict


def _merge_corpus_dicts(corpus_dicts: Sequence[dict[str, CorpusInfo]]) -> dict[str, CorpusInfo]:
    datasets = _import_datasets()
    merged: dict[str, CorpusInfo] = {}
    for corpus_dict in corpus_dicts:
        for corpus_id, corpus_info in corpus_dict.items():
            if corpus_id not in merged:
                merged[corpus_id] = corpus_info
                continue

            existing = merged[corpus_id]
            if not isinstance(existing.corpus, NormalizedArrowCorpusDataset) or not isinstance(
                corpus_info.corpus, NormalizedArrowCorpusDataset
            ):
                raise TypeError(f"Cannot merge non-normalized corpus dataset for corpus_id={corpus_id!r}")
            combined_dataset = datasets.concatenate_datasets([existing.corpus.data, corpus_info.corpus.data])
            corpus_metadata = dict(existing.metadata)
            merged[corpus_id] = CorpusInfo(
                corpus_metadata,
                NormalizedArrowCorpusDataset(
                    combined_dataset,
                    path=f"{existing.path},{corpus_info.path}",
                ),
            )
    return merged


def _select_source_samples(dataset: Any, num_samples: int | None, *, seed: int) -> Any:
    if num_samples is None:
        return dataset
    if num_samples > len(dataset):
        logger.warning(
            "Requested %d normalized retrieval samples but only found %d row(s). Using all.",
            num_samples,
            len(dataset),
        )
        return dataset
    return dataset.shuffle(seed=seed).select(range(num_samples))


def _load_normalized_bundle_components(
    bundle_root: Path,
    *,
    num_samples: int | None,
    seed: int,
) -> tuple[Any, dict[str, CorpusInfo]]:
    metadata = _load_metadata(bundle_root)
    if "sources" in metadata:
        datasets = _import_datasets()
        source_datasets = []
        source_corpus_dicts = []
        for source in metadata["sources"]:
            source_dataset, source_corpus_dict = _load_normalized_bundle_components(
                bundle_root / source["path"],
                num_samples=None,
                seed=seed,
            )
            source_datasets.append(source_dataset)
            source_corpus_dicts.append(source_corpus_dict)
        if not source_datasets:
            raise ValueError(f"Normalized retrieval bundle has no sources: {bundle_root}")
        dataset = source_datasets[0] if len(source_datasets) == 1 else datasets.concatenate_datasets(source_datasets)
        return _select_source_samples(dataset, num_samples, seed=seed), _merge_corpus_dicts(source_corpus_dicts)

    train_paths = [bundle_root / shard for shard in metadata.get("train_shards", [])]
    dataset = _load_arrow_shards(train_paths)
    dataset = _select_source_samples(dataset, num_samples, seed=seed)
    logger.info("Loaded normalized retrieval dataset with %d examples from %s", len(dataset), bundle_root)
    return dataset, _load_corpus_dict(bundle_root, metadata)


def _build_transform(
    dataset: Any,
    corpus_dict: dict[str, CorpusInfo],
    *,
    model_type: str,
    data_type: str,
    n_passages: int,
    eval_negative_size: int | None,
    use_dataset_instruction: bool,
    cycle_positive_docs: bool,
):
    negative_size = n_passages - 1 if data_type == "train" else eval_negative_size
    if negative_size is None:
        negative_size = n_passages - 1
    transform = RetrievalTransform(
        negative_size,
        corpus_dict,
        use_dataset_instruction=use_dataset_instruction,
        model_type=model_type,
        cycle_positive_docs=cycle_positive_docs,
    )
    if "pos_doc_json" in dataset.column_names:
        return _JsonRefRetrievalTransform(transform)
    return transform


def make_normalized_retrieval_dataset(
    data_path: str | Path | os.PathLike[str] | Sequence[Any] | None = None,
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int = 5,
    eval_negative_size: int | None = None,
    seed: int = 42,
    do_shuffle: bool = False,
    max_train_samples: int | None = None,
    train_data_select_offset: int = 0,
    use_dataset_instruction: bool = False,
    cycle_positive_docs: bool = False,
    data_dir_list: str | Path | os.PathLike[str] | Sequence[Any] | None = None,
) -> Any:
    """Build a normalized portable retrieval dataset from a local Arrow bundle."""
    if data_path is not None and data_dir_list is not None:
        raise ValueError("Provide only one of data_path or data_dir_list")
    if data_path is None:
        data_path = data_dir_list
    if data_path is None:
        raise ValueError("data_path or data_dir_list is required")

    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
    if data_type not in {"train", "eval"}:
        raise ValueError(f"Invalid data type: {data_type}")
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")

    data_entries = _parse_data_entries(data_path)
    source_datasets = []
    source_corpus_dicts = []
    for source_path, num_samples in data_entries:
        source_dataset, source_corpus_dict = _load_normalized_bundle_components(
            source_path,
            num_samples=num_samples,
            seed=seed,
        )
        source_datasets.append(source_dataset)
        source_corpus_dicts.append(source_corpus_dict)

    datasets = _import_datasets()
    dataset = source_datasets[0] if len(source_datasets) == 1 else datasets.concatenate_datasets(source_datasets)

    if data_type == "train" and max_train_samples is not None:
        if do_shuffle:
            dataset = dataset.shuffle(seed=seed)
        start = train_data_select_offset
        stop = min(start + max_train_samples, len(dataset))
        dataset = dataset.select(range(start, stop))

    corpus_dict = _merge_corpus_dicts(source_corpus_dicts)
    transform = _build_transform(
        dataset,
        corpus_dict,
        model_type=model_type,
        data_type=data_type,
        n_passages=n_passages,
        eval_negative_size=eval_negative_size,
        use_dataset_instruction=use_dataset_instruction,
        cycle_positive_docs=cycle_positive_docs,
    )
    dataset.set_transform(transform)
    if cycle_positive_docs:
        dataset.set_epoch = transform.set_epoch
    logger.info("Created normalized %s dataset with %d examples", data_type, len(dataset))
    return dataset
