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

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

QwenImageEditBucketSignature = tuple[tuple[int, int], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]

_TARGET_COLUMNS = (
    "target_image",
    "target_path",
    "edited_image",
    "edited_image_path",
    "output_image",
    "output_path",
    "image",
    "image_path",
    "path",
)
_CONTEXT_COLUMNS = (
    "input_images",
    "input_image",
    "input_paths",
    "input_path",
    "context_images",
    "context_image",
    "context_paths",
    "context_path",
    "source_images",
    "source_image",
    "source_paths",
    "source_path",
)
_CONDITION_COLUMNS = (
    "condition_images",
    "condition_image",
    "condition_paths",
    "condition_path",
    "vl_images",
    "vl_image",
)
_PROMPT_COLUMNS = ("prompt", "caption", "instruction", "edit_prompt", "text")


class QwenImageEditDataset(Dataset):
    """CSV-backed raw image-edit dataset for Qwen-Image-Edit training."""

    def __init__(
        self,
        csv_path: str,
        max_image_area: int | float | None = None,
        max_context_image_area: int | float | None = None,
        max_condition_image_area: int | float | None = None,
        max_total_seq_area: int | float | None = None,
        add_hex_rate: float = 0.0,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.root = self.csv_path.parent
        self.max_image_area = _to_optional_int(max_image_area)
        self.max_context_image_area = _to_optional_int(max_context_image_area)
        self.max_condition_image_area = _to_optional_int(max_condition_image_area)
        self.max_total_seq_area = _to_optional_int(max_total_seq_area)
        self.add_hex_rate = float(add_hex_rate or 0.0)
        self.rows = self._load_rows()
        self._bucket_signature_cache: dict[int, QwenImageEditBucketSignature] = {}

        if not self.rows:
            raise ValueError(f"No rows found in CSV: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target_path, context_paths, condition_paths = self._row_image_paths(row)

        prompt = self._first_value(row, _PROMPT_COLUMNS, default="")
        if self.add_hex_rate > 0:
            prompt = str(prompt)

        return {
            "target_image": self._load_tensor(target_path, self.max_image_area, normalize=True),
            "context_images": [
                self._load_tensor(path, self.max_context_image_area, normalize=True) for path in context_paths
            ],
            "condition_images": [
                self._load_tensor(path, self.max_condition_image_area, normalize=False) for path in condition_paths
            ],
            "prompt": prompt,
            "target_path": str(target_path),
            "context_paths": [str(path) for path in context_paths],
            "condition_paths": [str(path) for path in condition_paths],
            "data_type": "image_edit",
        }

    def bucket_signature(self, index: int) -> QwenImageEditBucketSignature:
        """Return the strict post-resize shape signature used for same-shape batching."""
        cached = self._bucket_signature_cache.get(index)
        if cached is not None:
            return cached

        target_path, context_paths, condition_paths = self._row_image_paths(self.rows[index])
        signature = (
            self._resized_image_size(target_path, self.max_image_area),
            tuple(self._resized_image_size(path, self.max_context_image_area) for path in context_paths),
            tuple(self._resized_image_size(path, self.max_condition_image_area) for path in condition_paths),
        )
        self._bucket_signature_cache[index] = signature
        return signature

    @staticmethod
    def bucket_signature_image_tokens(signature: QwenImageEditBucketSignature) -> int:
        """Estimate target-plus-context packed image-token count for a bucket signature."""
        target_size, context_sizes, _ = signature
        return _packed_image_token_count(target_size) + sum(_packed_image_token_count(size) for size in context_sizes)

    def _load_rows(self) -> list[dict[str, str]]:
        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = []
            for row in reader:
                cleaned = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
                if any(cleaned.values()):
                    rows.append(cleaned)
            return rows

    def _first_value(self, row: dict[str, str], names: tuple[str, ...], default: str | None = None) -> str:
        for name in names:
            value = row.get(name)
            if value:
                return value
        if default is not None:
            return default
        raise ValueError(f"CSV row is missing one of these columns: {', '.join(names)}")

    def _first_path(self, row: dict[str, str], names: tuple[str, ...]) -> Path:
        return self._resolve_path(self._first_value(row, names))

    def _row_image_paths(self, row: dict[str, str]) -> tuple[Path, list[Path], list[Path]]:
        target_path = self._first_path(row, _TARGET_COLUMNS)
        context_paths = self._paths_from_columns(row, _CONTEXT_COLUMNS)
        condition_paths = self._paths_from_columns(row, _CONDITION_COLUMNS)

        if not context_paths:
            raise ValueError(
                f"Could not find an input/context image column. Supported names include {', '.join(_CONTEXT_COLUMNS)}."
            )
        if not condition_paths:
            condition_paths = context_paths
        return target_path, context_paths, condition_paths

    def _paths_from_columns(self, row: dict[str, str], names: tuple[str, ...]) -> list[Path]:
        values: list[str] = []

        for name in names:
            value = row.get(name)
            if value:
                values.extend(_split_paths(value))

        for prefix in names:
            matching_keys = sorted(k for k in row if k.startswith(f"{prefix}_") or k.startswith(f"{prefix}."))
            for key in matching_keys:
                if row[key]:
                    values.extend(_split_paths(row[key]))

        seen: set[Path] = set()
        paths = []
        for value in values:
            path = self._resolve_path(value)
            if path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _load_tensor(self, path: Path, max_area: int | None, normalize: bool) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = _resize_to_area(image, max_area)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if normalize:
            tensor = tensor * 2.0 - 1.0
        return tensor

    def _resized_image_size(self, path: Path, max_area: int | None) -> tuple[int, int]:
        with Image.open(path) as image:
            return _resize_size_to_area(image.size, max_area)


def _to_optional_int(value: int | float | str | None) -> int | None:
    if value is None or value == "":
        return None
    value = int(float(value))
    return value if value > 0 else None


def _split_paths(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []

    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]

    for delimiter in ("|", ";", ","):
        if delimiter in value:
            return [part.strip() for part in value.split(delimiter) if part.strip()]
    return [value]


def _resize_to_area(image: Image.Image, max_area: int | None) -> Image.Image:
    width, height = _resize_size_to_area(image.size, max_area)
    if (width, height) == image.size:
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _resize_size_to_area(size: tuple[int, int], max_area: int | None) -> tuple[int, int]:
    width, height = size
    if max_area is not None:
        area = width * height
        if area > max_area:
            scale = math.sqrt(max_area / area)
            width = max(16, int(width * scale))
            height = max(16, int(height * scale))

    width = max(16, width - width % 16)
    height = max(16, height - height % 16)
    return width, height


def _packed_image_token_count(size: tuple[int, int]) -> int:
    width, height = size
    return (height // 16) * (width // 16)
