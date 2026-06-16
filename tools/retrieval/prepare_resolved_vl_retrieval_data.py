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

"""Resolve corpus-id VL retrieval data into streamable JSONL shards.

Example:

```
uv run python tools/retrieval/prepare_resolved_vl_retrieval_data.py \
  --config examples/retrieval/bi_encoder/nemotron_vl_1b/eagle_llama_1b_gmoreira_8_nodes_image.yaml \
  --output-dir /path/to/resolved_vl_retrieval \
  --num-build-shards 8 \
  --build-shard-index 0
```

The output can be consumed with
``nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any


def _patch_missing_torch_distribution_version() -> None:
    original_version = importlib.metadata.version

    def version(distribution_name: str) -> str:
        package_version = original_version(distribution_name)
        if distribution_name == "torch" and package_version is None:
            import torch

            package_version = torch.__version__
        return package_version

    importlib.metadata.version = version


_patch_missing_torch_distribution_version()

from nemo_automodel.components.datasets.llm.retrieval_dataset import _transform_func, load_datasets
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


class JsonlShardWriter:
    """Write records into fixed-size JSONL shards."""

    def __init__(self, output_dir: Path, samples_per_shard: int, *, filename_prefix: str = "shard") -> None:
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.filename_prefix = filename_prefix
        self._current_file = None
        self._current_shard_idx = -1
        self.num_records = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
        self._current_shard_idx += 1
        path = self.output_dir / f"{self.filename_prefix}-{self._current_shard_idx:05d}.jsonl"
        self.shard_paths.append(path.name)
        self._current_file = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self.num_records % self.samples_per_shard == 0:
            self._open_next_shard()
        assert self._current_file is not None
        self._current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.num_records += 1

    def close(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None


class SqliteShardWriter:
    """Write records and image bytes into fixed-size SQLite shards."""

    def __init__(self, output_dir: Path, samples_per_shard: int, *, filename_prefix: str = "shard") -> None:
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.filename_prefix = filename_prefix
        self._conn: sqlite3.Connection | None = None
        self._current_shard_idx = -1
        self._records_in_shard = 0
        self.num_records = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        self._close_current_shard()
        self._current_shard_idx += 1
        self._records_in_shard = 0
        path = self.output_dir / f"{self.filename_prefix}-{self._current_shard_idx:05d}.sqlite"
        self.shard_paths.append(path.name)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, record_json TEXT NOT NULL)")
        self._conn.execute(
            "CREATE TABLE images (record_id INTEGER NOT NULL, doc_idx INTEGER NOT NULL, "
            "image_jpeg BLOB NOT NULL, PRIMARY KEY (record_id, doc_idx))"
        )

    def _close_current_shard(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def write(self, record: dict[str, Any], image_blobs: dict[int, bytes]) -> None:
        if self.num_records % self.samples_per_shard == 0:
            self._open_next_shard()
        assert self._conn is not None
        record_id = self._records_in_shard
        self._conn.execute(
            "INSERT INTO records (id, record_json) VALUES (?, ?)",
            (record_id, json.dumps(record, ensure_ascii=False)),
        )
        self._conn.executemany(
            "INSERT INTO images (record_id, doc_idx, image_jpeg) VALUES (?, ?, ?)",
            [(record_id, doc_idx, blob) for doc_idx, blob in image_blobs.items()],
        )
        self.num_records += 1
        self._records_in_shard += 1

    def close(self) -> None:
        self._close_current_shard()


def _image_to_jpeg_bytes(image: Any, jpeg_quality: int) -> bytes:
    if not hasattr(image, "save"):
        raise ValueError(f"Unsupported image object type for resolved retrieval export: {type(image).__name__}")
    image = image.convert("RGB") if hasattr(image, "convert") else image
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _save_image(
    image: Any,
    images_dir: Path,
    image_rel_dir: Path,
    sample_idx: int,
    doc_idx: int,
    jpeg_quality: int,
) -> str:
    if image is None:
        return ""
    if isinstance(image, str):
        if not image:
            return ""
        return image
    if not hasattr(image, "save"):
        raise ValueError(f"Unsupported image object type for resolved retrieval export: {type(image).__name__}")

    image = image.convert("RGB") if hasattr(image, "convert") else image
    rel_path = image_rel_dir / f"{sample_idx:08d}_{doc_idx:02d}.jpg"
    image.save(images_dir / rel_path.name, format="JPEG", quality=jpeg_quality)
    return rel_path.as_posix()


def _resolve_record_images(
    images: list[Any],
    *,
    image_storage: str,
    images_dir: Path | None,
    image_rel_dir: Path,
    sample_idx: int,
    jpeg_quality: int,
) -> tuple[list[str], dict[int, bytes]]:
    image_refs: list[str] = []
    image_blobs: dict[int, bytes] = {}
    for doc_idx, image in enumerate(images):
        if image is None or (isinstance(image, str) and not image):
            image_refs.append("")
            continue
        if image_storage == "files":
            assert images_dir is not None
            image_refs.append(_save_image(image, images_dir, image_rel_dir, sample_idx, doc_idx, jpeg_quality))
            continue
        if isinstance(image, str):
            raise ValueError(
                "SQLite image storage requires resolved in-memory images, but got a string image reference. "
                f"Sample {sample_idx}, doc {doc_idx}: {image!r}"
            )
        image_refs.append(f"packed:{doc_idx}")
        image_blobs[doc_idx] = _image_to_jpeg_bytes(image, jpeg_quality)
    return image_refs, image_blobs


def _prepare_output_dir(
    output_dir: Path,
    *,
    filename_prefix: str,
    metadata_name: str,
    image_rel_dir: Path,
    image_storage: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(output_dir.glob(f"{filename_prefix}-*.jsonl"))
    existing_shards.extend(output_dir.glob(f"{filename_prefix}-*.sqlite"))
    metadata_path = output_dir / metadata_name
    if existing_shards or metadata_path.exists():
        raise FileExistsError(
            f"Output directory already contains artifacts for build shard prefix {filename_prefix!r}: {output_dir}. "
            "Choose a new directory or remove that shard's old artifacts explicitly."
        )
    images_dir = output_dir / image_rel_dir
    if image_storage == "files":
        if images_dir.exists() and any(images_dir.iterdir()):
            raise FileExistsError(
                f"Image directory already contains artifacts for build shard prefix {filename_prefix!r}: {images_dir}. "
                "Choose a new directory or remove that shard's old artifacts explicitly."
            )
        images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _validate_build_shard_args(num_build_shards: int, build_shard_index: int) -> None:
    if num_build_shards < 1:
        raise ValueError(f"num_build_shards must be >= 1, got {num_build_shards}")
    if build_shard_index < 0 or build_shard_index >= num_build_shards:
        raise ValueError(
            f"build_shard_index must satisfy 0 <= build_shard_index < num_build_shards, "
            f"got {build_shard_index} and {num_build_shards}"
        )


def _artifact_layout(num_build_shards: int, build_shard_index: int) -> tuple[str, str, Path]:
    if num_build_shards == 1:
        return "shard", "metadata.json", Path("images")
    build_prefix = f"shard-{build_shard_index:05d}"
    return build_prefix, f"metadata-build-shard-{build_shard_index:05d}.json", Path("images") / build_prefix


def _load_dataset_args_from_config(config_path: str) -> tuple[list[Any], int | None]:
    has_yaml, yaml = safe_import("yaml")
    if not has_yaml:
        raise ImportError("PyYAML is required to load --config. Install pyyaml or pass --data-dir-list explicitly.")

    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        dataset_cfg = cfg["dataloader"]["dataset"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Config does not contain dataloader.dataset: {config_path}") from e

    if "data_dir_list" not in dataset_cfg:
        raise ValueError(f"Config dataloader.dataset does not contain data_dir_list: {config_path}")
    return dataset_cfg["data_dir_list"], dataset_cfg.get("n_passages")


def resolve_dataset(
    data_dir_list: list[Any],
    output_dir: Path,
    *,
    n_passages: int,
    samples_per_shard: int,
    seed: int,
    max_samples: int | None,
    use_dataset_instruction: bool,
    jpeg_quality: int,
    image_storage: str = "files",
    num_build_shards: int = 1,
    build_shard_index: int = 0,
) -> dict[str, Any]:
    """Resolve corpus-id retrieval examples and write streamable JSONL shards."""
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    if image_storage not in {"files", "sqlite"}:
        raise ValueError(f"image_storage must be 'files' or 'sqlite', got {image_storage!r}")
    _validate_build_shard_args(num_build_shards, build_shard_index)
    filename_prefix, metadata_name, image_rel_dir = _artifact_layout(num_build_shards, build_shard_index)
    images_dir = _prepare_output_dir(
        output_dir,
        filename_prefix=filename_prefix,
        metadata_name=metadata_name,
        image_rel_dir=image_rel_dir,
        image_storage=image_storage,
    )

    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True, seed=seed)
    writer = (
        JsonlShardWriter(output_dir, samples_per_shard, filename_prefix=filename_prefix)
        if image_storage == "files"
        else SqliteShardWriter(output_dir, samples_per_shard, filename_prefix=filename_prefix)
    )

    try:
        for sample_idx, item in enumerate(dataset):
            if max_samples is not None and sample_idx >= max_samples:
                break
            if sample_idx % num_build_shards != build_shard_index:
                continue
            resolved = _transform_func(
                item,
                num_neg_docs=n_passages - 1,
                corpus_dict=corpus_dict,
                use_dataset_instruction=use_dataset_instruction,
            )
            image_refs, image_blobs = _resolve_record_images(
                resolved["doc_image"],
                image_storage=image_storage,
                images_dir=images_dir if image_storage == "files" else None,
                image_rel_dir=image_rel_dir,
                sample_idx=sample_idx,
                jpeg_quality=jpeg_quality,
            )
            record = {
                "question": resolved["question"],
                "doc_text": resolved["doc_text"],
                "doc_image": image_refs,
                "query_instruction": resolved["query_instruction"],
                "passage_instruction": resolved["passage_instruction"],
            }
            if "doc_id" in resolved:
                record["doc_id"] = resolved["doc_id"]
            if image_storage == "files":
                writer.write(record)
            else:
                writer.write(record, image_blobs)
    finally:
        writer.close()

    dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
    metadata = {
        "format": (
            "nemo_automodel_resolved_vl_retrieval_jsonl"
            if image_storage == "files"
            else "nemo_automodel_resolved_vl_retrieval_sqlite"
        ),
        "version": 1,
        "image_storage": image_storage,
        "data_dir_list": data_dir_list,
        "n_passages": n_passages,
        "dataset_size": dataset_size,
        "max_samples": max_samples,
        "num_build_shards": num_build_shards,
        "build_shard_index": build_shard_index,
        "num_records": writer.num_records,
        "samples_per_shard": samples_per_shard,
        "shards": writer.shard_paths,
        "image_dir": image_rel_dir.as_posix() if image_storage == "files" else None,
    }
    (output_dir / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d resolved retrieval records for build shard %d/%d to %s",
        writer.num_records,
        build_shard_index,
        num_build_shards,
        output_dir,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="AutoModel bi-encoder YAML. Uses dataloader.dataset.data_dir_list and n_passages.",
    )
    parser.add_argument("--data-dir-list", nargs="+", default=None, help="Corpus-id retrieval JSON files to resolve")
    parser.add_argument("--output-dir", required=True, help="Directory for resolved JSONL shards and images")
    parser.add_argument("--n-passages", type=int, default=None, help="Number of passages per query")
    parser.add_argument(
        "--samples-per-shard", type=int, default=10000, help="Number of examples per output JSONL shard"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for source sampling, if configured")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global sample cap for debug exports")
    parser.add_argument("--use-dataset-instruction", action="store_true", help="Write query/passage instructions")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for materialized images")
    parser.add_argument(
        "--image-storage",
        choices=("files", "sqlite"),
        default="files",
        help="Store images as loose JPEG files or packed SQLite BLOB shards.",
    )
    parser.add_argument(
        "--num-build-shards",
        type=int,
        default=1,
        help="Number of parallel CPU build shards. Each shard writes unique JSONL/image outputs.",
    )
    parser.add_argument(
        "--build-shard-index",
        type=int,
        default=0,
        help="Index of this CPU build shard in [0, num_build_shards).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the resolver CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.config is None and args.data_dir_list is None:
        raise ValueError("Provide either --config or --data-dir-list")
    if args.config is not None and args.data_dir_list is not None:
        raise ValueError("Provide only one of --config or --data-dir-list")

    config_n_passages = None
    if args.config is not None:
        data_dir_list, config_n_passages = _load_dataset_args_from_config(args.config)
    else:
        data_dir_list = args.data_dir_list

    n_passages = args.n_passages if args.n_passages is not None else config_n_passages
    if n_passages is None:
        n_passages = 5

    metadata = resolve_dataset(
        data_dir_list=data_dir_list,
        output_dir=Path(args.output_dir),
        n_passages=n_passages,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
        max_samples=args.max_samples,
        use_dataset_instruction=args.use_dataset_instruction,
        jpeg_quality=args.jpeg_quality,
        image_storage=args.image_storage,
        num_build_shards=args.num_build_shards,
        build_shard_index=args.build_shard_index,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
