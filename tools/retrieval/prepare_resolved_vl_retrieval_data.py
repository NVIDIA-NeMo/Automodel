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

"""Resolve corpus-id VL retrieval data into packed Arrow debug shards.

For full portable datasets, prefer
``tools/retrieval/prepare_normalized_vl_retrieval_data.py``. Resolved Arrow
stores document/image payload in each training row, so it is mainly useful for
small, self-contained reproduction datasets.

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


class ArrowShardWriter:
    """Write fixed-size resolved retrieval Arrow shards."""

    def __init__(
        self,
        output_dir: Path,
        samples_per_shard: int,
        *,
        filename_prefix: str = "shard",
        writer_batch_size: int = 100,
    ) -> None:
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        has_datasets, datasets = safe_import("datasets")
        has_arrow_writer, arrow_writer = safe_import("datasets.arrow_writer")
        if not has_datasets or not has_arrow_writer:
            raise ImportError("datasets is required to write resolved retrieval Arrow shards. Install datasets.")

        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.filename_prefix = filename_prefix
        self.writer_batch_size = writer_batch_size
        self._arrow_writer_cls = arrow_writer.ArrowWriter
        self._features = datasets.Features(
            {
                "id": datasets.Value("int64"),
                "record_json": datasets.Value("string"),
                "image_jpeg": datasets.Sequence(datasets.Value("binary")),
            }
        )
        self._writer = None
        self._current_shard_idx = -1
        self._records_in_shard = 0
        self.num_records = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        self.close()
        self._current_shard_idx += 1
        self._records_in_shard = 0
        path = self.output_dir / f"{self.filename_prefix}-{self._current_shard_idx:05d}.arrow"
        self.shard_paths.append(path.name)
        self._writer = self._arrow_writer_cls(
            features=self._features,
            path=str(path),
            writer_batch_size=self.writer_batch_size,
        )

    def write(self, record: dict[str, Any], image_blobs: dict[int, bytes]) -> None:
        if self.num_records % self.samples_per_shard == 0:
            self._open_next_shard()
        assert self._writer is not None
        record_id = self._records_in_shard
        doc_images = record.get("doc_image", [])
        self._writer.write(
            {
                "id": record_id,
                "record_json": json.dumps(record, ensure_ascii=False),
                "image_jpeg": [image_blobs.get(doc_idx, b"") for doc_idx in range(len(doc_images))],
            }
        )
        self.num_records += 1
        self._records_in_shard += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.finalize()
            self._writer = None


def _image_to_jpeg_bytes(image: Any, jpeg_quality: int) -> bytes:
    if not hasattr(image, "save"):
        raise ValueError(f"Unsupported image object type for resolved retrieval export: {type(image).__name__}")
    image = image.convert("RGB") if hasattr(image, "convert") else image
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _resolve_record_images(
    images: list[Any],
    *,
    sample_idx: int,
    jpeg_quality: int,
) -> tuple[list[str], dict[int, bytes]]:
    image_refs: list[str] = []
    image_blobs: dict[int, bytes] = {}
    for doc_idx, image in enumerate(images):
        if image is None or (isinstance(image, str) and not image):
            image_refs.append("")
            continue
        if isinstance(image, str):
            raise ValueError(
                "Resolved Arrow requires resolved in-memory images, but got a string image reference. "
                f"Sample {sample_idx}, doc {doc_idx}: {image!r}"
            )
        image_refs.append(f"packed:{doc_idx}")
        image_blobs[doc_idx] = _image_to_jpeg_bytes(image, jpeg_quality)
    return image_refs, image_blobs


def _prepare_output_dir(output_dir: Path, *, filename_prefix: str, metadata_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(output_dir.glob(f"{filename_prefix}-*.arrow"))
    metadata_path = output_dir / metadata_name
    if existing_shards or metadata_path.exists():
        raise FileExistsError(
            f"Output directory already contains artifacts for build shard prefix {filename_prefix!r}: {output_dir}. "
            "Choose a new directory or remove that shard's old artifacts explicitly."
        )


def _validate_build_shard_args(num_build_shards: int, build_shard_index: int) -> None:
    if num_build_shards < 1:
        raise ValueError(f"num_build_shards must be >= 1, got {num_build_shards}")
    if build_shard_index < 0 or build_shard_index >= num_build_shards:
        raise ValueError(
            f"build_shard_index must satisfy 0 <= build_shard_index < num_build_shards, "
            f"got {build_shard_index} and {num_build_shards}"
        )


def _artifact_layout(num_build_shards: int, build_shard_index: int) -> tuple[str, str]:
    if num_build_shards == 1:
        return "shard", "metadata.json"
    build_prefix = f"shard-{build_shard_index:05d}"
    return build_prefix, f"metadata-build-shard-{build_shard_index:05d}.json"


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
    num_build_shards: int = 1,
    build_shard_index: int = 0,
) -> dict[str, Any]:
    """Resolve corpus-id retrieval examples and write packed Arrow shards."""
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    _validate_build_shard_args(num_build_shards, build_shard_index)
    filename_prefix, metadata_name = _artifact_layout(num_build_shards, build_shard_index)
    _prepare_output_dir(output_dir, filename_prefix=filename_prefix, metadata_name=metadata_name)

    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True, seed=seed)
    writer = ArrowShardWriter(output_dir, samples_per_shard, filename_prefix=filename_prefix)

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
            writer.write(record, image_blobs)
    finally:
        writer.close()

    dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
    metadata = {
        "format": "nemo_automodel_resolved_vl_retrieval_arrow",
        "version": 1,
        "image_storage": "arrow",
        "data_dir_list": data_dir_list,
        "n_passages": n_passages,
        "dataset_size": dataset_size,
        "max_samples": max_samples,
        "num_build_shards": num_build_shards,
        "build_shard_index": build_shard_index,
        "num_records": writer.num_records,
        "samples_per_shard": samples_per_shard,
        "shards": writer.shard_paths,
    }
    (output_dir / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d resolved Arrow retrieval records for build shard %d/%d to %s",
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
    parser.add_argument("--output-dir", required=True, help="Directory for resolved Arrow shards")
    parser.add_argument("--n-passages", type=int, default=None, help="Number of passages per query")
    parser.add_argument(
        "--samples-per-shard", type=int, default=10000, help="Number of examples per output Arrow shard"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for source sampling, if configured")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global sample cap for debug exports")
    parser.add_argument("--use-dataset-instruction", action="store_true", help="Write query/passage instructions")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for packed images")
    parser.add_argument(
        "--num-build-shards",
        type=int,
        default=1,
        help="Number of parallel CPU build shards. Each shard writes unique Arrow outputs.",
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
        num_build_shards=args.num_build_shards,
        build_shard_index=args.build_shard_index,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
