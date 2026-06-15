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

"""Resolved retrieval JSONL dataset.

This loader is for retrieval data whose expensive corpus join has already been
performed offline. Each JSONL row is one training example with query text and
the selected positive/negative passages inline:

```
{"question": "...", "doc_text": ["pos", "neg"], "doc_image": ["images/a.jpg", ""]}
```

It intentionally avoids HuggingFace corpus loading and document-id lookups in
the training hot path. Image paths are opened lazily inside DataLoader workers.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import flatten_bi_encoder_to_cross_encoder
from nemo_automodel.shared.import_utils import safe_import

_VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")

logger = logging.getLogger(__name__)


def _coerce_path_list(data_path: str | Sequence[str]) -> list[Path]:
    if isinstance(data_path, (str, os.PathLike)):
        entries = [data_path]
    else:
        entries = list(data_path)

    paths: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Resolved retrieval data path does not exist: {path}")

    if not paths:
        raise ValueError(f"No JSONL files found in resolved retrieval data path: {data_path}")
    return paths


def _count_jsonl_records(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


def _get_dist_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _count_items_for_rank(total_items: int, rank: int, world_size: int) -> int:
    if total_items <= rank:
        return 0
    return ((total_items - 1 - rank) // world_size) + 1


def _normalize_doc_list(value: Any, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Resolved retrieval field '{field_name}' must be a list, got {type(value).__name__}")
    return value


def _normalize_text_list(value: Any) -> list[str]:
    docs = _normalize_doc_list(value, field_name="doc_text")
    return ["" if text is None else str(text) for text in docs]


def _normalize_optional_string_list(value: Any, *, field_name: str, expected_len: int) -> list[str]:
    if value is None:
        return [""] * expected_len
    items = _normalize_doc_list(value, field_name=field_name)
    if len(items) != expected_len:
        raise ValueError(f"Resolved retrieval field '{field_name}' must have length {expected_len}, got {len(items)}")
    return ["" if item is None else str(item) for item in items]


def _resolve_image_path(image_ref: str, *, image_root: Path | None, record_dir: Path) -> Path:
    image_path = Path(image_ref)
    if image_path.is_absolute():
        return image_path
    if image_root is not None:
        return image_root / image_path
    return record_dir / image_path


def _open_image(image_ref: Any, *, image_root: Path | None, record_dir: Path, decode_images: bool) -> Any:
    if image_ref is None:
        return ""
    if isinstance(image_ref, str) and not image_ref:
        return ""
    if not decode_images:
        return str(image_ref)
    if not isinstance(image_ref, str):
        raise ValueError(
            f"Resolved retrieval doc_image entries must be string paths or empty strings, got {type(image_ref).__name__}"
        )

    has_pil, image_module = safe_import("PIL.Image")
    if not has_pil:
        raise ImportError(
            "Pillow is required to decode resolved retrieval images. Install pillow or set decode_images=False."
        )

    image_path = _resolve_image_path(image_ref, image_root=image_root, record_dir=record_dir)
    with image_module.open(image_path) as image:
        return image.convert("RGB")


def _normalize_resolved_record(
    record: dict[str, Any],
    *,
    image_root: Path | None,
    record_dir: Path,
    decode_images: bool,
    model_type: str,
    expected_n_passages: int | None,
) -> dict[str, Any]:
    question = record.get("question", record.get("query"))
    if question is None:
        raise ValueError("Resolved retrieval record must include 'question' or 'query'")

    doc_text = _normalize_text_list(record.get("doc_text"))
    if not doc_text:
        raise ValueError("Resolved retrieval record must include at least one document in 'doc_text'")
    if expected_n_passages is not None and len(doc_text) != expected_n_passages:
        raise ValueError(f"Resolved retrieval record has {len(doc_text)} document(s), expected {expected_n_passages}")

    doc_image_refs = record.get("doc_image", [""] * len(doc_text))
    doc_image_raw = _normalize_doc_list(doc_image_refs, field_name="doc_image")
    if len(doc_image_raw) != len(doc_text):
        raise ValueError(
            f"Resolved retrieval field 'doc_image' must have length {len(doc_text)}, got {len(doc_image_raw)}"
        )
    doc_image = [
        _open_image(image_ref, image_root=image_root, record_dir=record_dir, decode_images=decode_images)
        for image_ref in doc_image_raw
    ]

    result = {
        "question": str(question),
        "doc_text": doc_text,
        "doc_image": doc_image,
        "query_instruction": "" if record.get("query_instruction") is None else str(record.get("query_instruction")),
        "passage_instruction": ""
        if record.get("passage_instruction") is None
        else str(record.get("passage_instruction")),
    }

    if "doc_id" in record:
        result["doc_id"] = _normalize_optional_string_list(
            record["doc_id"], field_name="doc_id", expected_len=len(doc_text)
        )

    if model_type == "cross_encoder":
        return flatten_bi_encoder_to_cross_encoder(
            {
                "question": [result["question"]],
                "doc_text": [result["doc_text"]],
                "doc_image": [result["doc_image"]],
            }
        )
    return result


class ResolvedRetrievalJsonlDataset(IterableDataset):
    """Stream resolved retrieval JSONL shards across ranks and DataLoader workers."""

    def __init__(
        self,
        data_path: str | Sequence[str],
        *,
        image_root: str | None = None,
        repeat: int = 1,
        shuffle_files: bool = False,
        seed: int = 42,
        decode_images: bool = True,
        model_type: str = "bi_encoder",
        num_samples: int | None = None,
        expected_n_passages: int | None = None,
    ) -> None:
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")

        self.paths = _coerce_path_list(data_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.repeat = int(repeat)
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.decode_images = decode_images
        self.model_type = model_type
        self.expected_n_passages = expected_n_passages
        self.epoch = 0
        total_records = _count_jsonl_records(self.paths)
        if num_samples is None:
            self._num_records = total_records
        else:
            requested_records = int(num_samples)
            if requested_records < 1:
                raise ValueError(f"num_samples must be >= 1, got {requested_records}")
            if requested_records > total_records:
                logger.warning(
                    "Requested %d resolved retrieval samples but only found %d row(s). Using all.",
                    requested_records,
                    total_records,
                )
            self._num_records = min(requested_records, total_records)
        if self._num_records < 1:
            raise ValueError(f"Resolved retrieval dataset is empty: {data_path}")

    def __len__(self) -> int:
        rank, world_size = _get_dist_info()
        return _count_items_for_rank(self._num_records * self.repeat, rank, world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_paths(self, repeat_idx: int) -> list[Path]:
        paths = list(self.paths)
        if self.shuffle_files:
            rng = random.Random(self.seed + self.epoch + repeat_idx)
            rng.shuffle(paths)
        return paths

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank, world_size = _get_dist_info()

        global_idx = 0
        local_idx_for_rank = 0
        for repeat_idx in range(self.repeat):
            records_seen_this_repeat = 0
            for path in self._iter_paths(repeat_idx):
                record_dir = path.parent
                with path.open("r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        if records_seen_this_repeat >= self._num_records:
                            break
                        if global_idx % world_size == rank:
                            emit = local_idx_for_rank % num_workers == worker_id
                            local_idx_for_rank += 1
                        else:
                            emit = False
                        global_idx += 1
                        records_seen_this_repeat += 1
                        if not emit:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Failed to parse resolved retrieval JSONL at {path}:{line_no}: {e}"
                            ) from e
                        if not isinstance(record, dict):
                            raise ValueError(
                                f"Resolved retrieval JSONL record must be an object at {path}:{line_no}, "
                                f"got {type(record).__name__}"
                            )
                        yield _normalize_resolved_record(
                            record,
                            image_root=self.image_root,
                            record_dir=record_dir,
                            decode_images=self.decode_images,
                            model_type=self.model_type,
                            expected_n_passages=self.expected_n_passages,
                        )
                if records_seen_this_repeat >= self._num_records:
                    break


def make_resolved_retrieval_dataset(
    data_path: str | Sequence[str] | None = None,
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int | None = None,
    image_root: str | None = None,
    repeat: int = 1,
    shuffle_files: bool = False,
    seed: int = 42,
    decode_images: bool = True,
    num_samples: int | None = None,
    data_dir_list: str | Sequence[str] | None = None,
) -> ResolvedRetrievalJsonlDataset:
    """Build a streaming resolved retrieval JSONL dataset.

    Args:
        data_path: JSONL file, directory of JSONL shards, or list of files/directories.
        model_type: ``"bi_encoder"`` or ``"cross_encoder"``.
        data_type: Accepted for YAML parity with ``make_retrieval_dataset``.
        n_passages: Optional validation check for the number of docs per row.
        image_root: Optional root for relative ``doc_image`` paths.
        repeat: Number of times to repeat the resolved shards.
        shuffle_files: Shuffle shard order each epoch.
        seed: File shuffle seed.
        decode_images: Decode image paths to RGB PIL images. Set false for text-only/debug inspection.
        num_samples: Optional cap on the number of JSONL records to stream per repeat.
        data_dir_list: Alias for ``data_path`` to match existing retrieval configs.

    Returns:
        A rank/worker-sharded ``IterableDataset`` yielding retrieval examples.
    """
    if data_type not in {"train", "eval"}:
        raise ValueError(f"Invalid data type: {data_type}")
    if n_passages is not None and n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    if data_path is not None and data_dir_list is not None:
        raise ValueError("Provide only one of data_path or data_dir_list")
    if data_path is None:
        data_path = data_dir_list
    if data_path is None:
        raise ValueError("data_path or data_dir_list is required")
    return ResolvedRetrievalJsonlDataset(
        data_path=data_path,
        image_root=image_root,
        repeat=repeat,
        shuffle_files=shuffle_files,
        seed=seed,
        decode_images=decode_images,
        model_type=model_type,
        num_samples=num_samples,
        expected_n_passages=n_passages,
    )
