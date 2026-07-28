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
"""Materialize Hugging Face media datasets for diffusion preprocessing."""

import json
import logging
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

HF_HUB_AVAILABLE, huggingface_hub = safe_import(
    "huggingface_hub",
    msg="Hugging Face dataset materialization requires huggingface_hub",
)

IMAGE_COLUMN_CANDIDATES = ("image", "jpg", "jpeg", "png", "webp")
VIDEO_COLUMN_CANDIDATES = ("video", "mp4", "file", "video_path", "path")
CAPTION_COLUMN_CANDIDATES = ("caption", "text", "prompt", "description")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EDIT_MEDIA_ROLES = {"target", "context", "condition"}


@dataclass(frozen=True)
class HFDatasetMediaMapping:
    """Map an image-edit media role to a Hugging Face dataset column.

    Attributes:
        role: Generic image-edit role: ``target``, ``context``, or ``condition``.
        column: Hugging Face dataset column containing image media.
    """

    role: str
    column: str


@dataclass(frozen=True)
class HFDatasetExport:
    """Summary of a materialized Hugging Face dataset split.

    Attributes:
        media_dir: Root directory containing the materialized media.
        total_items: Number of exported dataset rows.
        media_column: Dataset column used for the primary media.
        caption_column: Dataset column used for captions, when available.
        caption_file: Generated caption metadata path for image exports.
        media_mappings: Ordered image-edit role-to-column mappings.
        manifest_file: Generated generic image-edit manifest path.
        dataset_config_name: Dataset config selected by ``load_dataset``.
    """

    media_dir: Path
    total_items: int
    media_column: str
    caption_column: str | None
    caption_file: Path | None = None
    media_mappings: tuple[HFDatasetMediaMapping, ...] = ()
    manifest_file: Path | None = None
    dataset_config_name: str | None = None


def materialize_hf_dataset(
    dataset_name: str,
    output_dir: str | Path,
    *,
    media_type: str,
    split: str = "train",
    config_name: str | None = None,
    media_column: str | None = None,
    media_mappings: Sequence[HFDatasetMediaMapping] | None = None,
    caption_column: str | None = None,
    caption_field: str = "caption",
    max_items: int | None = None,
    streaming: bool = False,
    trust_remote_code: bool | None = None,
    download_timeout: int = 60,
) -> HFDatasetExport:
    """Download a Hugging Face split and export media plus captions locally.

    The diffusion preprocessors operate on local media paths. This helper adapts
    Hugging Face rows to that shape while preserving the existing image/video
    exports and providing a generic manifest for image-edit preprocessors.

    Args:
        dataset_name: Hugging Face dataset ID or local dataset path accepted by
            ``datasets.load_dataset``.
        output_dir: Directory where media files and caption metadata are written.
        media_type: ``"image"``, ``"video"``, or ``"image-edit"``.
        split: Dataset split to load.
        config_name: Optional dataset config/subset name.
        media_column: Optional source media column. If omitted, common names are
            inferred from features or the first row. This legacy option cannot
            be combined with ``media_mappings``.
        media_mappings: Ordered image-edit role-to-column mappings. Image-edit
            exports require one target and at least one context. Context and
            condition roles may be repeated.
        caption_column: Optional source caption column. If omitted, common text
            column names are inferred from features or the first row.
        caption_field: Caption key to write in generated metadata.
        max_items: Optional maximum number of rows to export.
        streaming: Whether to stream the dataset from Hugging Face.
        trust_remote_code: Optional value forwarded to ``load_dataset``.
        download_timeout: Timeout in seconds for URL-backed media cells.

    Returns:
        Metadata describing the exported dataset.

    Raises:
        ValueError: If columns cannot be resolved or media cells use an
            unsupported shape.
    """
    if media_type not in {"image", "video", "image-edit"}:
        raise ValueError(f"media_type must be 'image', 'video', or 'image-edit', got {media_type!r}")
    if media_column is not None and media_mappings:
        raise ValueError("--dataset_media_column cannot be combined with --dataset_media_mapping")
    if media_type == "image-edit":
        resolved_mappings = _validate_media_mappings(media_mappings)
    elif media_mappings:
        raise ValueError("media_mappings are only supported for media_type='image-edit'")
    else:
        resolved_mappings = ()

    output_path = Path(output_dir)
    has_exported_samples = output_path.exists() and (
        (output_path / "hf_internvl.json").exists()
        or (output_path / "hf_image_edit_manifest.jsonl").exists()
        or any(output_path.glob("hf_sample_*"))
        or any((output_path / "media").glob("hf_sample_*"))
    )
    if has_exported_samples:
        raise ValueError("HF materialization directory is not empty; choose a new directory")

    dataset = _load_hf_dataset(
        dataset_name,
        split=split,
        config_name=config_name,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    resolved_config_name = _resolve_dataset_config_name(dataset, config_name)

    available = _available_columns_from_dataset(dataset)
    if available:
        if media_type == "image-edit":
            _validate_mapping_columns(available, resolved_mappings)
            dataset = _maybe_cast_mapped_images_decode_false(dataset, resolved_mappings)
            media_column = _target_media_column(resolved_mappings)
        else:
            media_column = _resolve_media_column(available, media_type, media_column)
            dataset = _maybe_cast_decode_false(dataset, media_type, media_column)
        caption_column = _resolve_caption_column(available, caption_column)
        row_iter = iter(dataset)
        try:
            first_row = next(row_iter)
        except StopIteration as exc:
            raise ValueError(f"Dataset {dataset_name!r} split {split!r} is empty") from exc
    else:
        row_iter = iter(dataset)
        try:
            first_row = next(row_iter)
        except StopIteration as exc:
            raise ValueError(f"Dataset {dataset_name!r} split {split!r} is empty") from exc

        available = set(first_row.keys())
        if media_type == "image-edit":
            _validate_mapping_columns(available, resolved_mappings)
            dataset = _maybe_cast_mapped_images_decode_false(dataset, resolved_mappings)
            media_column = _target_media_column(resolved_mappings)
        else:
            media_column = _resolve_media_column(available, media_type, media_column)
            dataset = _maybe_cast_decode_false(dataset, media_type, media_column)
        caption_column = _resolve_caption_column(available, caption_column)

        # Re-create the iterator after an optional cast so rows reflect decode=False.
        row_iter = iter(dataset)
        try:
            first_row = next(row_iter)
        except StopIteration as exc:
            raise ValueError(f"Dataset {dataset_name!r} split {split!r} is empty after media casting") from exc

    output_path.mkdir(parents=True, exist_ok=True)

    caption_file = output_path / "hf_internvl.json" if media_type == "image" else None
    manifest_file = output_path / "hf_image_edit_manifest.jsonl" if media_type == "image-edit" else None
    if caption_file is not None:
        caption_file.write_text("", encoding="utf-8")
    if manifest_file is not None:
        manifest_file.write_text("", encoding="utf-8")

    total_items = 0
    for index, row in enumerate(_chain_first(first_row, row_iter)):
        if max_items is not None and index >= max_items:
            break

        caption = _get_caption(row, caption_column, fallback=f"hf sample {index:08d}")

        if media_type == "image":
            media_value = row[media_column]
            file_name = f"hf_sample_{index:08d}{_media_suffix(media_value, default='.png', allowed=IMAGE_EXTENSIONS)}"
            media_path = output_path / file_name
            _write_image(media_value, media_path, download_timeout)
            _append_jsonl_caption(caption_file, file_name, caption, caption_field)
        elif media_type == "video":
            media_value = row[media_column]
            suffix = _media_suffix(media_value, default=".mp4", allowed=VIDEO_EXTENSIONS)
            media_path = output_path / f"hf_sample_{index:08d}{suffix}"
            _write_binary_or_path(media_value, media_path, download_timeout)
            _write_sidecar_caption(media_path.with_suffix(".json"), caption, caption_field)
        else:
            _append_image_edit_manifest_row(
                manifest_file,
                row=row,
                index=index,
                prompt=caption,
                mappings=resolved_mappings,
                output_dir=output_path,
                dataset_name=dataset_name,
                dataset_config_name=resolved_config_name,
                dataset_split=split,
                caption_column=caption_column,
                download_timeout=download_timeout,
            )

        total_items += 1

    return HFDatasetExport(
        media_dir=output_path,
        total_items=total_items,
        media_column=media_column,
        caption_column=caption_column,
        caption_file=caption_file,
        media_mappings=resolved_mappings,
        manifest_file=manifest_file,
        dataset_config_name=resolved_config_name,
    )


def _resolve_dataset_config_name(dataset: object, requested_config_name: str | None) -> str | None:
    """Return the config name selected by ``datasets.load_dataset``."""
    dataset_info = getattr(dataset, "info", None)
    resolved_config_name = getattr(dataset_info, "config_name", None)
    if resolved_config_name is None or not isinstance(resolved_config_name, str) or not resolved_config_name:
        return requested_config_name
    return resolved_config_name


def _load_hf_dataset(
    dataset_name: str,
    *,
    split: str,
    config_name: str | None,
    streaming: bool,
    trust_remote_code: bool | None,
):
    """Load a Hugging Face dataset split."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if config_name is not None:
        kwargs["name"] = config_name
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code

    return load_dataset(dataset_name, **kwargs)


def _chain_first(first_row: dict[str, Any], rows):
    """Yield a consumed first row followed by the remaining iterator."""
    yield first_row
    yield from rows


def _validate_media_mappings(
    media_mappings: Sequence[HFDatasetMediaMapping] | None,
) -> tuple[HFDatasetMediaMapping, ...]:
    """Validate and freeze ordered image-edit media mappings."""
    mappings = tuple(media_mappings or ())
    if not mappings:
        raise ValueError("media_type='image-edit' requires media_mappings")

    invalid_roles = sorted({mapping.role for mapping in mappings} - IMAGE_EDIT_MEDIA_ROLES)
    if invalid_roles:
        raise ValueError(
            f"Unsupported image-edit media role(s): {', '.join(invalid_roles)}. "
            f"Expected roles: {', '.join(sorted(IMAGE_EDIT_MEDIA_ROLES))}"
        )

    empty_columns = [mapping.role for mapping in mappings if not mapping.column]
    if empty_columns:
        raise ValueError(f"Image-edit media columns cannot be empty for roles: {', '.join(empty_columns)}")

    target_count = sum(mapping.role == "target" for mapping in mappings)
    if target_count != 1:
        raise ValueError(f"Image-edit media mappings require exactly one target, got {target_count}")
    if not any(mapping.role == "context" for mapping in mappings):
        raise ValueError("Image-edit media mappings require at least one context")

    duplicate_mappings = sorted(
        {mapping for mapping in mappings if sum(candidate == mapping for candidate in mappings) > 1},
        key=lambda mapping: (mapping.role, mapping.column),
    )
    if duplicate_mappings:
        formatted = ", ".join(f"{mapping.role}={mapping.column}" for mapping in duplicate_mappings)
        raise ValueError(f"Duplicate image-edit media mapping(s): {formatted}")

    return mappings


def _validate_mapping_columns(available: set[str], mappings: tuple[HFDatasetMediaMapping, ...]) -> None:
    """Validate that all image-edit mapping columns exist in a dataset row."""
    missing = sorted({mapping.column for mapping in mappings} - available)
    if missing:
        raise ValueError(f"Image-edit media column(s) not found: {missing}. Available columns: {sorted(available)}")


def _target_media_column(mappings: tuple[HFDatasetMediaMapping, ...]) -> str:
    """Return the single validated target media column."""
    return next(mapping.column for mapping in mappings if mapping.role == "target")


def _maybe_cast_mapped_images_decode_false(dataset, mappings: tuple[HFDatasetMediaMapping, ...]):
    """Prefer path/bytes cells for every unique mapped image column."""
    for column in dict.fromkeys(mapping.column for mapping in mappings):
        dataset = _maybe_cast_decode_false(dataset, "image", column)
    return dataset


def _append_image_edit_manifest_row(
    manifest_file: Path,
    *,
    row: dict[str, Any],
    index: int,
    prompt: str,
    mappings: tuple[HFDatasetMediaMapping, ...],
    output_dir: Path,
    dataset_name: str,
    dataset_config_name: str | None,
    dataset_split: str,
    caption_column: str | None,
    download_timeout: int,
) -> None:
    """Materialize one image-edit row and append its generic manifest entry."""
    files_by_column: dict[str, str] = {}
    for media_index, column in enumerate(dict.fromkeys(mapping.column for mapping in mappings)):
        media_value = row[column]
        suffix = _media_suffix(media_value, default=".png", allowed=IMAGE_EXTENSIONS)
        file_name = f"media/hf_sample_{index:08d}_{media_index:02d}{suffix}"
        _write_image(media_value, output_dir / file_name, download_timeout)
        files_by_column[column] = file_name

    media = [{"role": mapping.role, "file_name": files_by_column[mapping.column]} for mapping in mappings]
    mapped_columns = set(files_by_column)
    row_metadata = {}
    for metadata_index, (column, value) in enumerate(row.items()):
        if column in mapped_columns or column == caption_column:
            continue
        row_metadata[column] = _serialize_metadata_value(
            value,
            output_dir=output_dir,
            index=index,
            metadata_index=metadata_index,
            column=column,
            download_timeout=download_timeout,
        )

    manifest_row = {
        "id": f"{dataset_split}:{index:08d}",
        "prompt": prompt,
        "media": media,
        "metadata": {
            "dataset_name": dataset_name,
            "dataset_config_name": dataset_config_name,
            "dataset_split": dataset_split,
            "row_index": index,
            "row": row_metadata,
        },
    }
    with manifest_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(manifest_row) + "\n")


def _serialize_metadata_value(
    value: Any,
    *,
    output_dir: Path,
    index: int,
    metadata_index: int,
    column: str,
    download_timeout: int,
) -> Any:
    """Convert row metadata to JSON values, materializing image objects by relative path."""
    from PIL import Image

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Image.Image):
        file_name = f"metadata/hf_sample_{index:08d}_{metadata_index:02d}_{_safe_name(column)}.png"
        _write_image(value, output_dir / file_name, download_timeout)
        return {"media_type": "image", "file_name": file_name}
    if isinstance(value, dict):
        return {
            str(key): _serialize_metadata_value(
                item,
                output_dir=output_dir,
                index=index,
                metadata_index=metadata_index,
                column=column,
                download_timeout=download_timeout,
            )
            for key, item in value.items()
            if key != "bytes"
        }
    if isinstance(value, (list, tuple)):
        return [
            _serialize_metadata_value(
                item,
                output_dir=output_dir,
                index=index,
                metadata_index=metadata_index,
                column=column,
                download_timeout=download_timeout,
            )
            for item in value
        ]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _safe_name(value: str) -> str:
    """Return a path-safe, readable fragment for generated metadata files."""
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe or "metadata"


def _resolve_media_column(available: set[str], media_type: str, requested: str | None) -> str:
    """Resolve the media column name."""
    if requested is not None:
        if requested not in available:
            raise ValueError(f"media column {requested!r} not found. Available columns: {sorted(available)}")
        return requested

    candidates = IMAGE_COLUMN_CANDIDATES if media_type == "image" else VIDEO_COLUMN_CANDIDATES
    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"Could not infer {media_type} column. Pass --dataset_media_column. Available columns: {sorted(available)}"
    )


def _resolve_caption_column(available: set[str], requested: str | None) -> str | None:
    """Resolve the caption column name."""
    if requested is not None:
        if requested not in available:
            raise ValueError(f"caption column {requested!r} not found. Available columns: {sorted(available)}")
        return requested

    for candidate in CAPTION_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate

    return None


def _available_columns_from_dataset(dataset) -> set[str]:
    """Return dataset column names from metadata without reading rows."""
    column_names = getattr(dataset, "column_names", None)
    if column_names:
        return set(column_names)
    features = getattr(dataset, "features", None)
    if features:
        return set(features.keys())
    return set()


def _maybe_cast_decode_false(dataset, media_type: str, media_column: str):
    """Prefer path/bytes cells for HF media features when available."""
    features = getattr(dataset, "features", None)
    if not features or media_column not in features:
        return dataset

    try:
        if media_type == "image":
            from datasets import Image

            if isinstance(features[media_column], Image):
                return dataset.cast_column(media_column, Image(decode=False))
        else:
            from datasets import Video

            if isinstance(features[media_column], Video):
                return dataset.cast_column(media_column, Video(decode=False))
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not cast %s column %s to decode=False: %s", media_type, media_column, exc)

    return dataset


def _get_caption(row: dict[str, Any], caption_column: str | None, fallback: str) -> str:
    """Read a caption from a row."""
    if caption_column is None:
        return fallback

    caption = row.get(caption_column)
    if caption is None:
        return fallback
    if isinstance(caption, list):
        return " ".join(str(part) for part in caption if part is not None).strip() or fallback
    return str(caption).strip() or fallback


def _append_jsonl_caption(caption_file: Path, file_name: str, caption: str, caption_field: str) -> None:
    """Append one image caption in the existing JSONL convention."""
    with caption_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"file_name": file_name, caption_field: caption}) + "\n")


def _write_sidecar_caption(caption_file: Path, caption: str, caption_field: str) -> None:
    """Write one video sidecar caption."""
    caption_file.write_text(json.dumps({caption_field: caption}, indent=2), encoding="utf-8")


def _write_image(media_value: Any, output_path: Path, download_timeout: int) -> None:
    """Write an image-like HF cell to disk."""
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(media_value, Image.Image):
        media_value.convert("RGB").save(output_path)
        return

    if isinstance(media_value, dict):
        raw_bytes = media_value.get("bytes")
        raw_path = media_value.get("path") or media_value.get("url")
        if raw_bytes is not None:
            with Image.open(_bytes_io(raw_bytes)) as image:
                image.convert("RGB").save(output_path)
            return
        if raw_path:
            _copy_or_download(raw_path, output_path, download_timeout)
            return

    if isinstance(media_value, (bytes, bytearray, memoryview)):
        with Image.open(_bytes_io(media_value)) as image:
            image.convert("RGB").save(output_path)
        return

    if isinstance(media_value, (str, Path)):
        _copy_or_download(media_value, output_path, download_timeout)
        return

    if hasattr(media_value, "__array__"):
        import numpy as np

        Image.fromarray(np.asarray(media_value)).convert("RGB").save(output_path)
        return

    raise ValueError(f"Unsupported image cell type: {type(media_value).__name__}")


def _write_binary_or_path(media_value: Any, output_path: Path, download_timeout: int) -> None:
    """Write a binary/path-like video HF cell to disk."""
    if isinstance(media_value, dict):
        raw_bytes = media_value.get("bytes")
        raw_path = media_value.get("path") or media_value.get("url")
        if raw_bytes is not None:
            output_path.write_bytes(bytes(raw_bytes))
            return
        if raw_path:
            _copy_or_download(raw_path, output_path, download_timeout)
            return

    if isinstance(media_value, (bytes, bytearray, memoryview)):
        output_path.write_bytes(bytes(media_value))
        return

    if isinstance(media_value, (str, Path)):
        _copy_or_download(media_value, output_path, download_timeout)
        return

    raise ValueError(
        f"Unsupported video cell type: {type(media_value).__name__}. "
        "Use a path, URL, bytes-backed cell, or cast the HF video column with decode=False."
    )


def _copy_or_download(source: str | Path, output_path: Path, timeout: int) -> None:
    """Copy a local file or download an HTTP(S) URL."""
    source_str = str(source)
    if _is_http_url(source_str):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source_str, timeout=timeout) as response:
            output_path.write_bytes(response.read())
        return
    if _is_hf_url(source_str):
        downloaded_path = _download_hf_url(source_str)
        _copy_or_download(downloaded_path, output_path, timeout)
        return

    source_path = Path(source_str)
    if not source_path.exists():
        raise ValueError(f"Media path does not exist locally: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.hardlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, output_path)


def _media_suffix(media_value: Any, *, default: str, allowed: set[str]) -> str:
    """Infer an output suffix from a path/URL cell."""
    source_path = None
    if isinstance(media_value, dict):
        source_path = media_value.get("path") or media_value.get("url")
    elif isinstance(media_value, (str, Path)):
        source_path = media_value

    if source_path:
        parsed_path = urlparse(str(source_path)).path
        suffix = Path(parsed_path).suffix.lower()
        if suffix in allowed:
            return suffix

    return default


def _is_http_url(value: str) -> bool:
    """Return whether a string is an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_hf_url(value: str) -> bool:
    """Return whether a string is a Hugging Face filesystem URL."""
    return urlparse(value).scheme == "hf"


def _download_hf_url(value: str) -> str:
    """Download a Hugging Face filesystem URL and return the local cache path."""
    from huggingface_hub import hf_hub_download

    parsed = urlparse(value)
    if parsed.netloc == "datasets":
        path = parsed.path.lstrip("/")
    else:
        path = parsed.path.lstrip("/")
        if path.startswith("datasets/"):
            path = path.removeprefix("datasets/")

    if "@" not in path:
        raise ValueError(f"Unsupported Hugging Face media URL without revision: {value}")

    repo_id, revision_and_file = path.split("@", 1)
    if "/" not in revision_and_file:
        raise ValueError(f"Unsupported Hugging Face media URL without file path: {value}")

    revision, filename = revision_and_file.split("/", 1)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type="dataset")


def _bytes_io(raw_bytes: bytes | bytearray | memoryview):
    """Wrap raw bytes as a file-like object."""
    import io

    return io.BytesIO(bytes(raw_bytes))
