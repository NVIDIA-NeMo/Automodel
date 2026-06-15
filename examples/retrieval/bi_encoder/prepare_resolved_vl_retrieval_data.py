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
uv run python examples/retrieval/bi_encoder/prepare_resolved_vl_retrieval_data.py \
  --data-dir-list training_a.json training_b.json \
  --output-dir /path/to/resolved_vl_retrieval \
  --n-passages 5
```

The output can be consumed with
``nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemo_automodel.components.datasets.llm.retrieval_dataset import _transform_func, load_datasets


class JsonlShardWriter:
    """Write records into fixed-size JSONL shards."""

    def __init__(self, output_dir: Path, samples_per_shard: int) -> None:
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self._current_file = None
        self._current_shard_idx = -1
        self.num_records = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
        self._current_shard_idx += 1
        path = self.output_dir / f"shard-{self._current_shard_idx:05d}.jsonl"
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


def _save_image(image: Any, images_dir: Path, sample_idx: int, doc_idx: int, jpeg_quality: int) -> str:
    if image is None:
        return ""
    if isinstance(image, str):
        if not image:
            return ""
        return image
    if not hasattr(image, "save"):
        raise ValueError(f"Unsupported image object type for resolved retrieval export: {type(image).__name__}")

    image = image.convert("RGB") if hasattr(image, "convert") else image
    rel_path = Path("images") / f"{sample_idx:08d}_{doc_idx:02d}.jpg"
    image.save(images_dir / rel_path.name, format="JPEG", quality=jpeg_quality)
    return rel_path.as_posix()


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("shard-*.jsonl")) or (output_dir / "metadata.json").exists():
        raise FileExistsError(
            f"Output directory already contains resolved retrieval artifacts: {output_dir}. "
            "Choose a new directory or remove the old artifacts explicitly."
        )
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    return images_dir


def resolve_dataset(
    data_dir_list: list[str],
    output_dir: Path,
    *,
    n_passages: int,
    samples_per_shard: int,
    seed: int,
    max_samples: int | None,
    use_dataset_instruction: bool,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Resolve corpus-id retrieval examples and write streamable JSONL shards."""
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    images_dir = _prepare_output_dir(output_dir)

    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True, seed=seed)
    writer = JsonlShardWriter(output_dir, samples_per_shard)

    try:
        for sample_idx, item in enumerate(dataset):
            if max_samples is not None and sample_idx >= max_samples:
                break
            resolved = _transform_func(
                item,
                num_neg_docs=n_passages - 1,
                corpus_dict=corpus_dict,
                use_dataset_instruction=use_dataset_instruction,
            )
            image_paths = [
                _save_image(image, images_dir, sample_idx, doc_idx, jpeg_quality)
                for doc_idx, image in enumerate(resolved["doc_image"])
            ]
            record = {
                "question": resolved["question"],
                "doc_text": resolved["doc_text"],
                "doc_image": image_paths,
                "query_instruction": resolved["query_instruction"],
                "passage_instruction": resolved["passage_instruction"],
            }
            if "doc_id" in resolved:
                record["doc_id"] = resolved["doc_id"]
            writer.write(record)
    finally:
        writer.close()

    metadata = {
        "format": "nemo_automodel_resolved_vl_retrieval_jsonl",
        "version": 1,
        "data_dir_list": data_dir_list,
        "n_passages": n_passages,
        "num_records": writer.num_records,
        "samples_per_shard": samples_per_shard,
        "shards": writer.shard_paths,
        "image_dir": "images",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir-list", nargs="+", required=True, help="Corpus-id retrieval JSON files to resolve")
    parser.add_argument("--output-dir", required=True, help="Directory for resolved JSONL shards and images")
    parser.add_argument("--n-passages", type=int, default=5, help="Number of passages per query")
    parser.add_argument(
        "--samples-per-shard", type=int, default=10000, help="Number of examples per output JSONL shard"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for source sampling, if configured")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debug exports")
    parser.add_argument("--use-dataset-instruction", action="store_true", help="Write query/passage instructions")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for materialized images")
    return parser.parse_args()


def main() -> None:
    """Run the resolver CLI."""
    args = parse_args()
    metadata = resolve_dataset(
        data_dir_list=args.data_dir_list,
        output_dir=Path(args.output_dir),
        n_passages=args.n_passages,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
        max_samples=args.max_samples,
        use_dataset_instruction=args.use_dataset_instruction,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
