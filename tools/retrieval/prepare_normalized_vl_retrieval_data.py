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

"""Build a portable normalized VL retrieval Arrow bundle.

Unlike resolved retrieval shards, this keeps the corpus-id data model:

- train shards contain query text and positive/negative document ids;
- corpus shards store each referenced document/image once;
- training resolves documents through local Arrow corpus lookup.

Example:

```
uv run python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config examples/retrieval/bi_encoder/nemotron_vl_1b/eagle_llama_1b_gmoreira_8_nodes_image.yaml \
  --output-dir /path/to/normalized_vl_retrieval \
  --max-samples 16000
```

The output can be consumed with
``nemo_automodel.components.datasets.llm.make_normalized_retrieval_dataset``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import shutil
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

from nemo_automodel.components.datasets.llm.retrieval_dataset import load_datasets
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


class ArrowShardWriter:
    """Write fixed-size Arrow shards with Hugging Face's ArrowWriter."""

    def __init__(
        self,
        output_dir: Path,
        samples_per_shard: int,
        features: Any,
        *,
        filename_prefix: str,
        writer_batch_size: int = 100,
    ) -> None:
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1, got {samples_per_shard}")
        has_arrow_writer, arrow_writer = safe_import("datasets.arrow_writer")
        if not has_arrow_writer:
            raise ImportError("datasets is required to write normalized retrieval Arrow shards. Install datasets.")

        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.features = features
        self.filename_prefix = filename_prefix
        self.writer_batch_size = writer_batch_size
        self._arrow_writer_cls = arrow_writer.ArrowWriter
        self._writer = None
        self._current_shard_idx = -1
        self.num_records = 0
        self.shard_paths: list[str] = []

    def _open_next_shard(self) -> None:
        self.close()
        self._current_shard_idx += 1
        path = self.output_dir / f"{self.filename_prefix}-{self._current_shard_idx:05d}.arrow"
        self.shard_paths.append(path.name)
        self._writer = self._arrow_writer_cls(
            features=self.features,
            path=str(path),
            writer_batch_size=self.writer_batch_size,
        )

    def write(self, record: dict[str, Any]) -> None:
        if self.num_records % self.samples_per_shard == 0:
            self._open_next_shard()
        assert self._writer is not None
        self._writer.write(record)
        self.num_records += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.finalize()
            self._writer = None


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


def _normalize_doc_refs(value: Any, *, field_name: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Retrieval field '{field_name}' must be a list, got {type(value).__name__}")
    refs = []
    for doc in value:
        if isinstance(doc, dict) and "id" in doc:
            refs.append({"id": str(doc["id"])})
        else:
            refs.append({"id": str(doc)})
    return refs


def _image_to_jpeg_bytes(image: Any, jpeg_quality: int) -> bytes:
    if image is None or (isinstance(image, str) and not image):
        return b""
    if isinstance(image, str):
        has_pil, image_module = safe_import("PIL.Image")
        if not has_pil:
            raise ImportError("Pillow is required to pack string image paths into normalized Arrow.")
        with image_module.open(image) as loaded:
            image = loaded.convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image object type for normalized retrieval export: {type(image).__name__}")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _corpus_text(doc: dict[str, Any]) -> str:
    text = doc.get("text", "")
    return "" if text is None else str(text)


def _prepare_output_dir(output_dir: Path, *, resume: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if not resume and (metadata_path.exists() or (output_dir / "train").exists() or (output_dir / "corpus").exists()):
        raise FileExistsError(
            f"Output directory already contains normalized retrieval artifacts: {output_dir}. "
            "Choose a new directory or remove old artifacts explicitly."
        )
    (output_dir / "train").mkdir(exist_ok=resume)
    (output_dir / "corpus").mkdir(exist_ok=resume)


def _prepare_multi_source_output_dir(output_dir: Path, *, resume: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    if not resume and (
        metadata_path.exists()
        or (output_dir / "sources").exists()
        or (output_dir / "train").exists()
        or (output_dir / "corpus").exists()
    ):
        raise FileExistsError(
            f"Output directory already contains normalized retrieval artifacts: {output_dir}. "
            "Choose a new directory or remove old artifacts explicitly."
        )
    (output_dir / "sources").mkdir(exist_ok=resume)


def _split_data_dir_list(data_dir_list: Any) -> list[Any]:
    if isinstance(data_dir_list, (str, dict)):
        return [data_dir_list]
    if isinstance(data_dir_list, list):
        if not data_dir_list:
            raise ValueError("data_dir_list must contain at least one source")
        return data_dir_list
    raise ValueError(f"Unsupported data_dir_list type: {type(data_dir_list).__name__}")


def _safe_corpus_dir_name(corpus_id: str) -> str:
    return corpus_id.replace("/", "__")


def _arrow_shard_paths(path: Path, filename_prefix: str) -> list[Path]:
    return sorted(path.glob(f"{filename_prefix}-*.arrow"))


def _parse_doc_refs(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    return _normalize_doc_refs(value, field_name="doc refs")


def _collect_refs_from_train_shards(train_dir: Path) -> tuple[list[str], int, dict[str, set[str]]] | None:
    """Read existing train shards from a previous interrupted run."""
    train_paths = _arrow_shard_paths(train_dir, "train")
    if not train_paths:
        return None

    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to resume normalized retrieval Arrow shards. Install datasets.")

    refs_by_corpus: dict[str, set[str]] = {}
    num_records = 0
    try:
        for path in train_paths:
            shard = datasets.Dataset.from_file(str(path))
            column_names = set(shard.column_names)
            for row in shard:
                corpus_id = str(row["corpus_id"])
                refs = refs_by_corpus.setdefault(corpus_id, set())
                if "pos_doc_json" in column_names:
                    pos_doc = _parse_doc_refs(row["pos_doc_json"])
                    neg_doc = _parse_doc_refs(row["neg_doc_json"])
                else:
                    pos_doc = _parse_doc_refs(row["pos_doc"])
                    neg_doc = _parse_doc_refs(row["neg_doc"])
                refs.update(doc["id"] for doc in pos_doc)
                refs.update(doc["id"] for doc in neg_doc)
            num_records += len(shard)
    except Exception:
        logger.warning("Could not read existing train shards in %s; rewriting train shards", train_dir, exc_info=True)
        return None

    logger.info("Reusing %d existing normalized train records from %s", num_records, train_dir)
    return [f"train/{path.name}" for path in train_paths], num_records, refs_by_corpus


def _existing_corpus_metadata_if_complete(
    corpus_id: str,
    corpus_info: Any,
    corpus_output_dir: Path,
    refs: set[str],
) -> dict[str, Any] | None:
    """Return metadata for a complete existing corpus directory, or None if it must be rewritten."""
    shard_paths = _arrow_shard_paths(corpus_output_dir, "corpus")
    if not shard_paths:
        return None

    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to resume normalized retrieval Arrow shards. Install datasets.")

    num_records = 0
    try:
        for path in shard_paths:
            num_records += len(datasets.Dataset.from_file(str(path)))
    except Exception:
        logger.warning("Corpus %s has unreadable existing shards in %s; rewriting", corpus_id, corpus_output_dir)
        return None

    if num_records != len(refs):
        logger.warning(
            "Corpus %s has %d existing docs but %d are expected; rewriting",
            corpus_id,
            num_records,
            len(refs),
        )
        return None

    logger.info("Reusing %d normalized corpus docs for corpus_id=%s", num_records, corpus_id)
    return {
        "corpus_id": corpus_id,
        "metadata": corpus_info.metadata,
        "num_docs": num_records,
        "shards": [f"corpus/{corpus_output_dir.name}/{path.name}" for path in shard_paths],
    }


def _write_train_shards(
    dataset: Any,
    output_dir: Path,
    *,
    max_samples: int | None,
    samples_per_shard: int,
    seed: int,
) -> tuple[list[str], int, dict[str, set[str]]]:
    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to write normalized retrieval Arrow shards. Install datasets.")

    features = datasets.Features(
        {
            "question_id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "corpus_id": datasets.Value("string"),
            "pos_doc": [{"id": datasets.Value("string")}],
            "neg_doc": [{"id": datasets.Value("string")}],
        }
    )
    writer = ArrowShardWriter(
        output_dir / "train",
        samples_per_shard,
        features,
        filename_prefix="train",
    )
    refs_by_corpus: dict[str, set[str]] = {}
    try:
        for sample_idx, item in enumerate(dataset):
            if max_samples is not None and sample_idx >= max_samples:
                break
            corpus_id = str(item["corpus_id"])
            pos_doc = _normalize_doc_refs(item["pos_doc"], field_name="pos_doc")
            neg_doc = _normalize_doc_refs(item["neg_doc"], field_name="neg_doc")
            refs = refs_by_corpus.setdefault(corpus_id, set())
            refs.update(doc["id"] for doc in pos_doc)
            refs.update(doc["id"] for doc in neg_doc)
            writer.write(
                {
                    "question_id": str(item.get("question_id", sample_idx)),
                    "question": str(item["question"]),
                    "corpus_id": corpus_id,
                    "pos_doc": pos_doc,
                    "neg_doc": neg_doc,
                }
            )
    finally:
        writer.close()
    logger.info("Wrote %d normalized train records using seed %d", writer.num_records, seed)
    return [f"train/{path}" for path in writer.shard_paths], writer.num_records, refs_by_corpus


def _write_corpus_shards(
    corpus_dict: dict[str, Any],
    refs_by_corpus: dict[str, set[str]],
    output_dir: Path,
    *,
    docs_per_shard: int,
    jpeg_quality: int,
    resume: bool = False,
) -> list[dict[str, Any]]:
    has_datasets, datasets = safe_import("datasets")
    if not has_datasets:
        raise ImportError("datasets is required to write normalized retrieval Arrow shards. Install datasets.")

    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "text": datasets.Value("string"),
            "image_jpeg": datasets.Value("binary"),
            "nr_ocr": datasets.Value("string"),
            "complex_ocr": datasets.Value("string"),
        }
    )
    corpus_metadata: list[dict[str, Any]] = []
    for corpus_id in sorted(refs_by_corpus):
        if corpus_id not in corpus_dict:
            raise ValueError(f"Unknown corpus_id {corpus_id!r}; available corpora: {sorted(corpus_dict)}")
        corpus_info = corpus_dict[corpus_id]
        corpus_dir_name = _safe_corpus_dir_name(corpus_id)
        corpus_output_dir = output_dir / "corpus" / corpus_dir_name
        if resume and corpus_output_dir.exists():
            existing_metadata = _existing_corpus_metadata_if_complete(
                corpus_id,
                corpus_info,
                corpus_output_dir,
                refs_by_corpus[corpus_id],
            )
            if existing_metadata is not None:
                corpus_metadata.append(existing_metadata)
                continue
            shutil.rmtree(corpus_output_dir)
        corpus_output_dir.mkdir(parents=True)
        writer = ArrowShardWriter(
            corpus_output_dir,
            docs_per_shard,
            features,
            filename_prefix="corpus",
        )
        doc_ids = sorted(refs_by_corpus[corpus_id])
        try:
            for doc_id in doc_ids:
                doc = corpus_info.get_document_by_id(doc_id)
                writer.write(
                    {
                        "id": doc_id,
                        "text": _corpus_text(doc),
                        "image_jpeg": _image_to_jpeg_bytes(doc.get("image", ""), jpeg_quality),
                        "nr_ocr": "" if doc.get("nr_ocr") is None else str(doc.get("nr_ocr")),
                        "complex_ocr": "" if doc.get("complex_ocr") is None else str(doc.get("complex_ocr")),
                    }
                )
        finally:
            writer.close()
        shards = [f"corpus/{corpus_dir_name}/{path}" for path in writer.shard_paths]
        corpus_metadata.append(
            {
                "corpus_id": corpus_id,
                "metadata": corpus_info.metadata,
                "num_docs": writer.num_records,
                "shards": shards,
            }
        )
        logger.info("Wrote %d normalized corpus docs for corpus_id=%s", writer.num_records, corpus_id)
    return corpus_metadata


def _prepare_single_normalized_dataset(
    data_dir_list: list[Any],
    output_dir: Path,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Write a normalized portable Arrow retrieval bundle."""
    _prepare_output_dir(output_dir, resume=resume)
    dataset, corpus_dict = load_datasets(data_dir_list, concatenate=True, seed=seed)
    existing_train = _collect_refs_from_train_shards(output_dir / "train") if resume else None
    if existing_train is None:
        if resume and (output_dir / "train").exists():
            shutil.rmtree(output_dir / "train")
            (output_dir / "train").mkdir()
        train_shards, num_records, refs_by_corpus = _write_train_shards(
            dataset,
            output_dir,
            max_samples=max_samples,
            samples_per_shard=samples_per_shard,
            seed=seed,
        )
    else:
        train_shards, num_records, refs_by_corpus = existing_train
    corpora = _write_corpus_shards(
        corpus_dict,
        refs_by_corpus,
        output_dir,
        docs_per_shard=docs_per_shard,
        jpeg_quality=jpeg_quality,
        resume=resume,
    )
    dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
    metadata = {
        "format": "nemo_automodel_normalized_vl_retrieval_arrow",
        "version": 2,
        "data_dir_list": data_dir_list,
        "dataset_size": dataset_size,
        "max_samples": max_samples,
        "num_records": num_records,
        "samples_per_shard": samples_per_shard,
        "docs_per_shard": docs_per_shard,
        "train_shards": train_shards,
        "corpora": corpora,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote normalized retrieval bundle with %d train records to %s", num_records, output_dir)
    return metadata


def prepare_normalized_dataset(
    data_dir_list: list[Any],
    output_dir: Path,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Write a normalized portable Arrow retrieval bundle."""
    source_entries = _split_data_dir_list(data_dir_list)
    _prepare_multi_source_output_dir(output_dir, resume=resume)
    source_metadata = []
    total_records = 0
    for source_idx, source_entry in enumerate(source_entries):
        source_dir = output_dir / "sources" / f"source-{source_idx:05d}"
        metadata = _prepare_single_normalized_dataset(
            data_dir_list=[source_entry],
            output_dir=source_dir,
            samples_per_shard=samples_per_shard,
            docs_per_shard=docs_per_shard,
            seed=seed,
            max_samples=max_samples,
            jpeg_quality=jpeg_quality,
            resume=resume,
        )
        total_records += metadata["num_records"]
        source_metadata.append(
            {
                "source_index": source_idx,
                "source_entry": source_entry,
                "path": str(source_dir.relative_to(output_dir)),
                "num_records": metadata["num_records"],
            }
        )

    if max_samples is not None and len(source_entries) > 1:
        logger.warning(
            "--max-samples is applied to each normalized source independently. "
            "Use per-source num_samples in the original config or at training time for source-specific caps."
        )

    metadata = {
        "format": "nemo_automodel_normalized_vl_retrieval_arrow",
        "version": 3,
        "data_dir_list": data_dir_list,
        "num_records": total_records,
        "samples_per_shard": samples_per_shard,
        "docs_per_shard": docs_per_shard,
        "sources": source_metadata,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Wrote normalized retrieval bundle with %d source(s), %d total train records to %s",
        len(source_metadata),
        total_records,
        output_dir,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="AutoModel bi-encoder YAML. Uses dataloader.dataset.data_dir_list.",
    )
    parser.add_argument("--data-dir-list", nargs="+", default=None, help="Corpus-id retrieval JSON files to bundle")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized Arrow train/corpus shards")
    parser.add_argument("--samples-per-shard", type=int, default=10000, help="Train examples per Arrow shard")
    parser.add_argument("--docs-per-shard", type=int, default=10000, help="Corpus documents per Arrow shard")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for source sampling, if configured")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global train sample cap")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for packed corpus images")
    parser.add_argument("--resume", action="store_true", help="Reuse complete shards in an existing output directory")
    return parser.parse_args()


def main() -> None:
    """Run the normalized retrieval bundle builder CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.config is None and args.data_dir_list is None:
        raise ValueError("Provide either --config or --data-dir-list")
    if args.config is not None and args.data_dir_list is not None:
        raise ValueError("Provide only one of --config or --data-dir-list")

    if args.config is not None:
        data_dir_list, _ = _load_dataset_args_from_config(args.config)
    else:
        data_dir_list = args.data_dir_list

    metadata = prepare_normalized_dataset(
        data_dir_list=data_dir_list,
        output_dir=Path(args.output_dir),
        samples_per_shard=args.samples_per_shard,
        docs_per_shard=args.docs_per_shard,
        seed=args.seed,
        max_samples=args.max_samples,
        jpeg_quality=args.jpeg_quality,
        resume=args.resume,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
