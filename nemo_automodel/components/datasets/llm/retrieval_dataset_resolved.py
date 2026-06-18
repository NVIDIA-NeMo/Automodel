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

"""Resolved retrieval dataset.

This loader is for retrieval data whose expensive corpus join has already been
performed offline. Each JSONL row or packed shard row is one training example
with query text and the selected positive/negative passages inline:

```
{"question": "...", "doc_text": ["pos", "neg"], "doc_image": ["images/a.jpg", ""]}
```

It intentionally avoids HuggingFace corpus loading and document-id lookups in
the training hot path. Loose image paths are opened lazily inside DataLoader
workers. Packed SQLite/Parquet shards store image bytes inline.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import flatten_bi_encoder_to_cross_encoder
from nemo_automodel.components.datasets.reservoir_sampler import ReservoirSampler
from nemo_automodel.shared.import_utils import safe_import

_VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")
_VALID_PARQUET_SHARDING = ("row_group", "row")

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
            paths.extend(sorted(path.glob("*.sqlite")))
            paths.extend(sorted(path.glob("*.parquet")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Resolved retrieval data path does not exist: {path}")

    if not paths:
        raise ValueError(f"No JSONL, SQLite, or Parquet files found in resolved retrieval data path: {data_path}")
    return paths


def _coerce_arrow_path_list(data_path: str | Sequence[str]) -> list[Path]:
    if isinstance(data_path, (str, os.PathLike)):
        entries = [data_path]
    else:
        entries = list(data_path)

    paths: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.arrow")))
        elif path.is_file() and _is_arrow_path(path):
            paths.append(path)
        elif not path.exists():
            raise FileNotFoundError(f"Resolved retrieval data path does not exist: {path}")

    return paths


def _is_sqlite_path(path: Path) -> bool:
    return path.suffix == ".sqlite"


def _is_parquet_path(path: Path) -> bool:
    return path.suffix == ".parquet"


def _is_arrow_path(path: Path) -> bool:
    return path.suffix == ".arrow"


def _import_datasets():
    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to read resolved retrieval Arrow shards. Install datasets.")
    return datasets


def _import_pyarrow_parquet():
    has_pq, pq = safe_import("pyarrow.parquet")
    if not has_pq:
        raise ImportError("pyarrow is required to read resolved retrieval Parquet shards. Install pyarrow.")
    return pq


def _count_resolved_records(paths: Sequence[Path]) -> int:
    count = 0
    for path in paths:
        if _is_sqlite_path(path):
            with _open_sqlite_connection(path) as conn:
                count += int(conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        elif _is_parquet_path(path):
            pq = _import_pyarrow_parquet()
            count += int(pq.ParquetFile(path).metadata.num_rows)
        else:
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


def _decode_image_bytes(image_bytes: bytes) -> Any:
    has_pil, image_module = safe_import("PIL.Image")
    if not has_pil:
        raise ImportError("Pillow is required to decode resolved retrieval images. Install pillow.")

    with image_module.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _open_image(
    image_ref: Any,
    *,
    image_root: Path | None,
    record_dir: Path,
    decode_images: bool,
    packed_image_bytes: bytes | None = None,
) -> Any:
    if packed_image_bytes is not None:
        if not decode_images:
            return str(image_ref)
        return _decode_image_bytes(packed_image_bytes)
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
    packed_images: dict[int, bytes] | None = None,
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
    doc_image = []
    for doc_idx, image_ref in enumerate(doc_image_raw):
        doc_image.append(
            _open_image(
                image_ref,
                image_root=image_root,
                record_dir=record_dir,
                decode_images=decode_images,
                packed_image_bytes=packed_images.get(doc_idx) if packed_images is not None else None,
            )
        )

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


def _open_sqlite_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _parse_sqlite_record(record_json: str) -> dict[str, Any]:
    record = json.loads(record_json)
    if not isinstance(record, dict):
        raise ValueError(f"Resolved retrieval SQLite record must be an object, got {type(record).__name__}")
    return record


def _load_sqlite_images(conn: sqlite3.Connection, record_id: int) -> dict[int, bytes]:
    return {
        int(doc_idx): bytes(image_jpeg)
        for doc_idx, image_jpeg in conn.execute(
            "SELECT doc_idx, image_jpeg FROM images WHERE record_id = ? ORDER BY doc_idx", (record_id,)
        )
    }


def _parquet_images_to_dict(image_values: Sequence[Any] | None) -> dict[int, bytes]:
    if image_values is None:
        return {}
    return {
        doc_idx: bytes(image_jpeg)
        for doc_idx, image_jpeg in enumerate(image_values)
        if image_jpeg is not None and len(image_jpeg) > 0
    }


class ResolvedRetrievalArrowDataset(Dataset):
    """Map-style resolved retrieval dataset backed by Hugging Face Arrow shards."""

    def __init__(
        self,
        data_path: str | Sequence[str],
        *,
        image_root: str | None = None,
        repeat: int = 1,
        decode_images: bool = True,
        model_type: str = "bi_encoder",
        num_samples: int | None = None,
        expected_n_passages: int | None = None,
    ) -> None:
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")

        self.paths = _coerce_arrow_path_list(data_path)
        if not self.paths:
            raise ValueError(f"No Arrow files found in resolved retrieval data path: {data_path}")

        datasets = _import_datasets()
        shards = [datasets.Dataset.from_file(str(path)) for path in self.paths]
        self._dataset = shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)
        self.image_root = Path(image_root) if image_root is not None else None
        self.repeat = int(repeat)
        self.decode_images = decode_images
        self.model_type = model_type
        self.expected_n_passages = expected_n_passages

        total_records = len(self._dataset)
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
            raise ValueError(f"Resolved retrieval Arrow dataset is empty: {data_path}")

    def __len__(self) -> int:
        return self._num_records * self.repeat

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        row = self._dataset[int(idx) % self._num_records]
        record = json.loads(row["record_json"])
        if not isinstance(record, dict):
            raise ValueError(f"Resolved retrieval Arrow record must be an object, got {type(record).__name__}")
        return _normalize_resolved_record(
            record,
            image_root=self.image_root,
            record_dir=Path("."),
            decode_images=self.decode_images,
            model_type=self.model_type,
            expected_n_passages=self.expected_n_passages,
            packed_images=_parquet_images_to_dict(row.get("image_jpeg")),
        )


class ResolvedRetrievalJsonlDataset(IterableDataset):
    """Stream resolved retrieval JSONL/SQLite/Parquet shards across ranks and DataLoader workers."""

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
        shuffle_buffer_size: int = 0,
        parquet_sharding: str = "row_group",
        shuffle_row_groups: bool = False,
    ) -> None:
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")
        if parquet_sharding not in _VALID_PARQUET_SHARDING:
            raise ValueError(f"parquet_sharding must be one of {_VALID_PARQUET_SHARDING}, got {parquet_sharding!r}")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        if shuffle_buffer_size < 0:
            raise ValueError(f"shuffle_buffer_size must be >= 0, got {shuffle_buffer_size}")

        self.paths = _coerce_path_list(data_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.repeat = int(repeat)
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.decode_images = decode_images
        self.model_type = model_type
        self.expected_n_passages = expected_n_passages
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.parquet_sharding = parquet_sharding
        self.shuffle_row_groups = bool(shuffle_row_groups)
        self.epoch = 0
        total_records = _count_resolved_records(self.paths)
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

    def _iter_jsonl_records(
        self,
        path: Path,
        *,
        state: dict[str, int],
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterable[tuple[dict[str, Any], dict[int, bytes] | None]]:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                if state["records_seen_this_repeat"] >= self._num_records:
                    break
                if state["global_idx"] % world_size == rank:
                    emit = state["local_idx_for_rank"] % num_workers == worker_id
                    state["local_idx_for_rank"] += 1
                else:
                    emit = False
                state["global_idx"] += 1
                state["records_seen_this_repeat"] += 1
                if not emit:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse resolved retrieval JSONL at {path}:{line_no}: {e}") from e
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Resolved retrieval JSONL record must be an object at {path}:{line_no}, "
                        f"got {type(record).__name__}"
                    )
                yield record, None

    def _iter_sqlite_records(
        self,
        path: Path,
        *,
        state: dict[str, int],
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterable[tuple[dict[str, Any], dict[int, bytes]]]:
        with _open_sqlite_connection(path) as conn:
            for record_id, record_json in conn.execute("SELECT id, record_json FROM records ORDER BY id"):
                if state["records_seen_this_repeat"] >= self._num_records:
                    break
                if state["global_idx"] % world_size == rank:
                    emit = state["local_idx_for_rank"] % num_workers == worker_id
                    state["local_idx_for_rank"] += 1
                else:
                    emit = False
                state["global_idx"] += 1
                state["records_seen_this_repeat"] += 1
                if not emit:
                    continue
                yield _parse_sqlite_record(record_json), _load_sqlite_images(conn, int(record_id))

    def _iter_parquet_records(
        self,
        path: Path,
        *,
        state: dict[str, int],
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterable[tuple[dict[str, Any], dict[int, bytes]]]:
        pq = _import_pyarrow_parquet()
        parquet_file = pq.ParquetFile(path)
        for row_group_idx in range(parquet_file.num_row_groups):
            if state["records_seen_this_repeat"] >= self._num_records:
                break
            row_group_rows = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
            rows_to_consume = min(row_group_rows, self._num_records - state["records_seen_this_repeat"])
            if self.parquet_sharding == "row_group":
                if state["parquet_group_idx"] % world_size == rank:
                    emit = state["parquet_local_group_idx_for_rank"] % num_workers == worker_id
                    state["parquet_local_group_idx_for_rank"] += 1
                else:
                    emit = False
                state["parquet_group_idx"] += 1
                state["records_seen_this_repeat"] += rows_to_consume
                if not emit:
                    continue

                emit_row_indices = range(rows_to_consume)
            else:
                emit_row_indices = []
                for row_idx in range(rows_to_consume):
                    if state["global_idx"] % world_size == rank:
                        emit = state["local_idx_for_rank"] % num_workers == worker_id
                        state["local_idx_for_rank"] += 1
                    else:
                        emit = False
                    state["global_idx"] += 1
                    state["records_seen_this_repeat"] += 1
                    if emit:
                        emit_row_indices.append(row_idx)
                state["parquet_group_idx"] += 1
                if not emit_row_indices:
                    continue

            table = parquet_file.read_row_group(row_group_idx, columns=["record_json", "image_jpeg"])
            record_json_col = table.column("record_json").combine_chunks()
            image_col = table.column("image_jpeg").combine_chunks()
            for row_idx in emit_row_indices:
                record_json = record_json_col[row_idx].as_py()
                image_values = image_col[row_idx].as_py()
                yield _parse_sqlite_record(record_json), _parquet_images_to_dict(image_values)

    def _parquet_row_group_shuffle_seed(self, repeat_idx: int) -> int:
        return self.seed + self.epoch * 1_000_003 + repeat_idx * 10_007

    def _iter_shuffled_parquet_row_group_records(
        self,
        paths: Sequence[Path],
        *,
        repeat_idx: int,
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterable[tuple[dict[str, Any], dict[int, bytes], Path]]:
        pq = _import_pyarrow_parquet()
        row_groups: list[tuple[Path, int, int]] = []
        records_seen_this_repeat = 0
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            for row_group_idx in range(parquet_file.num_row_groups):
                if records_seen_this_repeat >= self._num_records:
                    break
                row_group_rows = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
                rows_to_consume = min(row_group_rows, self._num_records - records_seen_this_repeat)
                row_groups.append((path, row_group_idx, rows_to_consume))
                records_seen_this_repeat += rows_to_consume
            if records_seen_this_repeat >= self._num_records:
                break

        rng = random.Random(self._parquet_row_group_shuffle_seed(repeat_idx))
        rng.shuffle(row_groups)

        parquet_files: dict[Path, Any] = {}
        local_group_idx_for_rank = 0
        for global_group_idx, (path, row_group_idx, rows_to_consume) in enumerate(row_groups):
            if global_group_idx % world_size != rank:
                continue
            emit = local_group_idx_for_rank % num_workers == worker_id
            local_group_idx_for_rank += 1
            if not emit:
                continue

            parquet_file = parquet_files.get(path)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(path)
                parquet_files[path] = parquet_file
            table = parquet_file.read_row_group(row_group_idx, columns=["record_json", "image_jpeg"])
            record_json_col = table.column("record_json").combine_chunks()
            image_col = table.column("image_jpeg").combine_chunks()
            for row_idx in range(rows_to_consume):
                record_json = record_json_col[row_idx].as_py()
                image_values = image_col[row_idx].as_py()
                yield _parse_sqlite_record(record_json), _parquet_images_to_dict(image_values), path.parent

    def _iter_raw_records_for_repeat(
        self,
        repeat_idx: int,
        *,
        state: dict[str, int],
        rank: int,
        world_size: int,
        worker_id: int,
        num_workers: int,
    ) -> Iterable[tuple[dict[str, Any], dict[int, bytes] | None, Path]]:
        state["records_seen_this_repeat"] = 0
        paths = self._iter_paths(repeat_idx)
        if (
            self.parquet_sharding == "row_group"
            and self.shuffle_row_groups
            and all(_is_parquet_path(path) for path in paths)
        ):
            yield from self._iter_shuffled_parquet_row_group_records(
                paths,
                repeat_idx=repeat_idx,
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            return

        for path in paths:
            record_dir = path.parent
            if _is_parquet_path(path):
                record_iter = self._iter_parquet_records(
                    path,
                    state=state,
                    rank=rank,
                    world_size=world_size,
                    worker_id=worker_id,
                    num_workers=num_workers,
                )
            elif _is_sqlite_path(path):
                record_iter = self._iter_sqlite_records(
                    path,
                    state=state,
                    rank=rank,
                    world_size=world_size,
                    worker_id=worker_id,
                    num_workers=num_workers,
                )
            else:
                record_iter = self._iter_jsonl_records(
                    path,
                    state=state,
                    rank=rank,
                    world_size=world_size,
                    worker_id=worker_id,
                    num_workers=num_workers,
                )
            for record, packed_images in record_iter:
                yield record, packed_images, record_dir
            if state["records_seen_this_repeat"] >= self._num_records:
                break

    def _shuffle_seed(self, repeat_idx: int, rank: int, worker_id: int) -> int:
        return self.seed + self.epoch * 1_000_003 + repeat_idx * 10_007 + rank * 101 + worker_id

    def __iter__(self) -> Iterable[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank, world_size = _get_dist_info()

        state = {
            "global_idx": 0,
            "local_idx_for_rank": 0,
            "parquet_group_idx": 0,
            "parquet_local_group_idx_for_rank": 0,
            "records_seen_this_repeat": 0,
        }
        for repeat_idx in range(self.repeat):
            raw_records = self._iter_raw_records_for_repeat(
                repeat_idx,
                state=state,
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            if self.shuffle_buffer_size > 0:
                raw_records = ReservoirSampler(
                    raw_records,
                    buffer_size=self.shuffle_buffer_size,
                    seed=self._shuffle_seed(repeat_idx, rank, worker_id),
                )

            for record, packed_images, record_dir in raw_records:
                yield _normalize_resolved_record(
                    record,
                    image_root=self.image_root,
                    record_dir=record_dir,
                    decode_images=self.decode_images,
                    model_type=self.model_type,
                    expected_n_passages=self.expected_n_passages,
                    packed_images=packed_images,
                )


def make_resolved_retrieval_dataset(
    data_path: str | Sequence[str] | None = None,
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int | None = None,
    image_root: str | None = None,
    repeat: int = 1,
    shuffle_files: bool | None = None,
    seed: int = 42,
    decode_images: bool = True,
    num_samples: int | None = None,
    data_dir_list: str | Sequence[str] | None = None,
    do_shuffle: bool | None = None,
    shuffle_buffer_size: int | None = None,
    parquet_sharding: str = "row_group",
    shuffle_row_groups: bool | None = None,
) -> ResolvedRetrievalArrowDataset | ResolvedRetrievalJsonlDataset:
    """Build a resolved retrieval dataset.

    Args:
        data_path: JSONL/SQLite/Parquet/Arrow file, directory of shards, or list of files/directories.
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
        do_shuffle: Compatibility alias for existing retrieval configs; maps to ``shuffle_files`` when set.
        shuffle_buffer_size: Number of already-sharded examples to buffer for streaming row-level shuffle.
            If omitted, ``do_shuffle=True`` uses a conservative buffer of 128 examples.
        parquet_sharding: ``"row_group"`` keeps Parquet reads efficient by assigning row groups to ranks/workers.
            ``"row"`` matches JSONL/SQLite row-level sharding but may duplicate Parquet reads across ranks.
        shuffle_row_groups: Shuffle Parquet row groups before rank/worker sharding. If omitted,
            ``do_shuffle=True`` enables row-group shuffle for Parquet row-group sharding.

    Returns:
        A map-style dataset for Arrow shards, otherwise a rank/worker-sharded ``IterableDataset``.
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

    arrow_paths = _coerce_arrow_path_list(data_path)
    if arrow_paths:
        return ResolvedRetrievalArrowDataset(
            data_path=data_path,
            image_root=image_root,
            repeat=repeat,
            decode_images=decode_images,
            model_type=model_type,
            num_samples=num_samples,
            expected_n_passages=n_passages,
        )

    if shuffle_files is None:
        shuffle_files = bool(do_shuffle)
    if shuffle_buffer_size is None:
        shuffle_buffer_size = 128 if (do_shuffle or shuffle_files) else 0
    if shuffle_row_groups is None:
        shuffle_row_groups = bool(do_shuffle or shuffle_files)
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
        shuffle_buffer_size=shuffle_buffer_size,
        parquet_sharding=parquet_sharding,
        shuffle_row_groups=shuffle_row_groups,
    )
