# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Offline JSONL/local-image export to self-contained Parquet documents."""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import cast

from PIL import Image

from nemo_automodel.components.datasets.vlm.conversation import MediaValue
from nemo_automodel.components.datasets.zephon.vlm_input import normalize_image_document
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


def _embed_image(value: MediaValue, root: Path) -> MediaValue:
    if isinstance(value, str):
        if "://" in value:
            raise ValueError("Download URL media before export; this converter reads local image files only")
        return (root / value).read_bytes()
    if isinstance(value, dict) and value.get("bytes") is None and isinstance(value.get("path"), str):
        return _embed_image(value["path"], root)
    return value


def _embed_paths(payload: dict[str, MediaValue], root: Path) -> None:
    for name in ("image", "images"):
        if name in payload:
            value = payload[name]
            payload[name] = (
                [_embed_image(image, root) for image in value] if isinstance(value, list) else _embed_image(value, root)
            )
    for name in ("conversation", "messages", "conversations"):
        messages = payload.get(name)
        if isinstance(messages, str):
            messages = json.loads(messages)
            payload[name] = messages
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue  # The normalizer reports invalid schemas with document context.
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image" and block.get("image") is not None:
                            block["image"] = _embed_image(block["image"], root)


def export_jsonl(
    source: Path, destination: Path, *, media_root: Path | None = None, shard_bytes: int = 64 << 20
) -> int:
    """Stream canonical RGB/PNG documents to new, bounded Parquet shards.

    A single document may exceed the target shard size. At most one document and
    one row group (up to 32 documents or the byte target) are decoded at a time.
    The destination must not exist, preventing accidental shard replacement.
    """
    if shard_bytes < 1:
        raise ValueError("shard_bytes must be positive")
    available, arrow = safe_import("pyarrow")
    _, parquet = safe_import("pyarrow.parquet")
    if not available:
        raise ImportError("Parquet export requires pyarrow")
    schema = arrow.schema(
        [
            (
                "conversation",
                arrow.list_(
                    arrow.struct(
                        [
                            ("role", arrow.string()),
                            (
                                "content",
                                arrow.list_(
                                    arrow.struct(
                                        [("type", arrow.string()), ("text", arrow.string()), ("image", arrow.binary())]
                                    )
                                ),
                            ),
                        ]
                    )
                ),
            )
        ]
    )
    destination.mkdir(parents=True, exist_ok=False)
    root = source.parent if media_root is None else media_root
    count = 0
    shard_index = 0
    buffered_bytes = 0
    shard_size = 0
    records = []
    writer = None
    try:
        with source.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                payload = cast(dict[str, MediaValue], json.loads(line))
                _embed_paths(payload, root)
                document = normalize_image_document(payload, source_name=f"{source.name}:{line_number}")
                size = 0
                for message in document["conversation"]:
                    content = message["content"]
                    assert isinstance(content, list)
                    for block in content:
                        if block["type"] == "image":
                            image = block["image"]
                            assert isinstance(image, Image.Image)
                            buffer = io.BytesIO()
                            image.save(buffer, format="PNG")
                            block["image"] = buffer.getvalue()
                            size += len(buffer.getvalue())
                        else:
                            size += len(str(block["text"]).encode())
                records.append(document)
                count += 1
                buffered_bytes += size
                if len(records) >= 32 or buffered_bytes >= shard_bytes:
                    if writer is None:
                        writer = parquet.ParquetWriter(destination / f"part-{shard_index:05d}.parquet", schema)
                    writer.write_table(arrow.Table.from_pylist(records, schema=schema))
                    records.clear()
                    shard_size += buffered_bytes
                    buffered_bytes = 0
                    if shard_size >= shard_bytes:
                        writer.close()
                        writer = None
                        shard_index += 1
                        shard_size = 0
            if records:
                if writer is None:
                    writer = parquet.ParquetWriter(destination / f"part-{shard_index:05d}.parquet", schema)
                writer.write_table(arrow.Table.from_pylist(records, schema=schema))
    finally:
        if writer is not None:
            writer.close()
    logger.info("Exported %d image documents to %s", count, destination)
    return count


def main() -> None:
    """Export JSONL plus local media before starting Zephon training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--shard-bytes", type=int, default=64 << 20)
    args = parser.parse_args()
    export_jsonl(args.source, args.destination, media_root=args.media_root, shard_bytes=args.shard_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
