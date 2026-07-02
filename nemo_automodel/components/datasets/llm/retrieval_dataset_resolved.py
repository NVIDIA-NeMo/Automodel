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

"""Resolved retrieval dataset backed by packed Arrow shards.

This loader is for retrieval data whose expensive corpus join has already been
performed offline. Each Arrow row is one training example with query text and
the selected positive/negative passages inline. Image bytes are packed into the
same Arrow row.

For full VL retrieval datasets, prefer the normalized Arrow format. Resolved
Arrow intentionally duplicates document/image payload per training row and is
mainly useful for small, self-contained debug/repro datasets.
"""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset

from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import flatten_bi_encoder_to_cross_encoder
from nemo_automodel.shared.import_utils import safe_import

_VALID_MODEL_TYPES = ("bi_encoder", "cross_encoder")

logger = logging.getLogger(__name__)


def _is_arrow_path(path: Path) -> bool:
    return path.suffix == ".arrow"


def _parse_data_entries(
    data_path: str | os.PathLike[str] | Sequence[Any],
) -> list[tuple[str | os.PathLike[str], int | None]]:
    if isinstance(data_path, (str, os.PathLike)):
        entries = [data_path]
    elif isinstance(data_path, dict):
        entries = [data_path]
    else:
        entries = list(data_path)

    parsed = []
    for entry in entries:
        if isinstance(entry, (str, os.PathLike)):
            parsed.append((entry, None))
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                "Resolved retrieval data entries must be paths or dictionaries with 'path' and optional "
                f"'num_samples', got {type(entry).__name__}"
            )
        allowed_keys = {"path", "num_samples"}
        unknown_keys = set(entry) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Unsupported resolved retrieval data entry field(s): {sorted(unknown_keys)}")
        if "path" not in entry:
            raise ValueError("Resolved retrieval data entry dictionaries must contain a 'path' field")
        num_samples = entry.get("num_samples")
        if num_samples is not None:
            if isinstance(num_samples, bool) or not isinstance(num_samples, int):
                raise ValueError(f"num_samples must be an integer or None, got {type(num_samples).__name__}")
            if num_samples < 1:
                raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        parsed.append((entry["path"], num_samples))
    if not parsed:
        raise ValueError("Resolved retrieval data expects at least one Arrow path")
    return parsed


def _coerce_arrow_path_list(data_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[Path]:
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
        elif path.exists():
            raise ValueError(f"Resolved retrieval only supports Arrow shards, got: {path}")
        else:
            raise FileNotFoundError(f"Resolved retrieval data path does not exist: {path}")

    if not paths:
        raise ValueError(f"No Arrow files found in resolved retrieval data path: {data_path}")
    return paths


def _import_datasets():
    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to read resolved retrieval Arrow shards. Install datasets.")
    return datasets


def _decode_image_bytes(image_bytes: bytes) -> Any:
    has_pil, image_module = safe_import("PIL.Image")
    if not has_pil:
        raise ImportError("Pillow is required to decode resolved retrieval images. Install pillow.")

    with image_module.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _resolve_image_path(image_ref: str, *, image_root: Path | None, record_dir: Path) -> Path:
    image_path = Path(image_ref)
    if image_path.is_absolute():
        return image_path
    if image_root is not None:
        return image_root / image_path
    return record_dir / image_path


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


def _arrow_images_to_dict(image_values: Sequence[Any] | None) -> dict[int, bytes]:
    if image_values is None:
        return {}
    return {
        doc_idx: bytes(image_jpeg)
        for doc_idx, image_jpeg in enumerate(image_values)
        if image_jpeg is not None and len(image_jpeg) > 0
    }


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
    doc_image = [
        _open_image(
            image_ref,
            image_root=image_root,
            record_dir=record_dir,
            decode_images=decode_images,
            packed_image_bytes=packed_images.get(doc_idx) if packed_images is not None else None,
        )
        for doc_idx, image_ref in enumerate(doc_image_raw)
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


class ResolvedRetrievalArrowDataset(Dataset):
    """Map-style resolved retrieval dataset backed by packed Arrow shards."""

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

        data_entries = _parse_data_entries(data_path)
        datasets = _import_datasets()
        source_datasets = []
        self.paths = []
        for source_path, source_num_samples in data_entries:
            source_paths = _coerce_arrow_path_list(source_path)
            self.paths.extend(source_paths)
            shards = [datasets.Dataset.from_file(str(path)) for path in source_paths]
            source_dataset = shards[0] if len(shards) == 1 else datasets.concatenate_datasets(shards)
            if source_num_samples is not None:
                if source_num_samples > len(source_dataset):
                    logger.warning(
                        "Requested %d resolved retrieval samples but only found %d row(s). Using all.",
                        source_num_samples,
                        len(source_dataset),
                    )
                source_dataset = source_dataset.select(range(min(source_num_samples, len(source_dataset))))
            source_datasets.append(source_dataset)
        self._dataset = (
            source_datasets[0] if len(source_datasets) == 1 else datasets.concatenate_datasets(source_datasets)
        )
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
            packed_images=_arrow_images_to_dict(row.get("image_jpeg")),
        )


def make_resolved_retrieval_dataset(
    data_path: str | Sequence[str] | None = None,
    model_type: str = "bi_encoder",
    data_type: str = "train",
    n_passages: int | None = None,
    image_root: str | None = None,
    repeat: int = 1,
    seed: int = 42,
    decode_images: bool = True,
    num_samples: int | None = None,
    data_dir_list: str | Sequence[str] | None = None,
    do_shuffle: bool | None = None,
) -> ResolvedRetrievalArrowDataset:
    """Build a resolved retrieval dataset from packed Arrow shards.

    ``seed`` and ``do_shuffle`` are accepted for YAML compatibility. Use the
    DataLoader sampler/shuffle controls for map-style resolved Arrow data.
    """
    del seed, do_shuffle
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

    return ResolvedRetrievalArrowDataset(
        data_path=data_path,
        image_root=image_root,
        repeat=repeat,
        decode_images=decode_images,
        model_type=model_type,
        num_samples=num_samples,
        expected_n_passages=n_passages,
    )
