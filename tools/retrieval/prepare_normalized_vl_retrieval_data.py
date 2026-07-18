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

This keeps the corpus-id data model:

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
import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from nemo_automodel.components.datasets.llm.retrieval_dataset import load_datasets
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

_TRAIN_COMPLETION_FILENAME = ".complete.json"


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

    dataset_cfg = cfg.get("dataset") if isinstance(cfg, dict) else None
    if dataset_cfg is None and isinstance(cfg, dict):
        dataloader_cfg = cfg.get("dataloader")
        if isinstance(dataloader_cfg, dict):
            dataset_cfg = dataloader_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Config does not contain dataset or legacy dataloader.dataset: {config_path}")

    if "data_dir_list" not in dataset_cfg:
        raise ValueError(f"Config dataset does not contain data_dir_list: {config_path}")
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


def _as_data_dir_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _source_entry_key(source_entry: Any) -> str:
    payload = json.dumps(source_entry, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _source_entry_name(source_entry: Any) -> str:
    source_path = source_entry.get("path") if isinstance(source_entry, dict) else source_entry
    name = Path(str(source_path)).stem or "source"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "source"


def _loaded_dataset_fingerprint(dataset: Any) -> str | None:
    fingerprint = getattr(dataset, "_fingerprint", None)
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def _loaded_source_fingerprint(dataset: Any, corpus_dict: dict[str, Any]) -> str | None:
    """Fingerprint the loaded query and corpus datasets used by one source."""
    query_fingerprint = _loaded_dataset_fingerprint(dataset)
    if query_fingerprint is None:
        return None

    corpus_fingerprints = []
    for corpus_id in sorted(corpus_dict):
        corpus_info = corpus_dict[corpus_id]
        corpus = getattr(corpus_info, "corpus", None)
        corpus_dataset = getattr(corpus, "_data", None)
        if corpus_dataset is None:
            corpus_dataset = getattr(corpus, "data", None)
        corpus_fingerprint = _loaded_dataset_fingerprint(corpus_dataset)
        if corpus_fingerprint is None:
            return None
        corpus_fingerprints.append(
            {
                "corpus_id": corpus_id,
                "metadata": corpus_info.metadata,
                "fingerprint": corpus_fingerprint,
            }
        )

    payload = json.dumps(
        {"query": query_fingerprint, "corpora": corpus_fingerprints},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(path)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _validate_or_write_resume_state(
    output_dir: Path,
    data_dir_list: list[Any],
    source_fingerprint: str | None,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
    resume: bool,
) -> None:
    """Persist the inputs that determine whether existing shards are reusable."""
    state_path = output_dir / ".resume-state.json"
    expected_state = {
        "source_key": _source_entry_key(data_dir_list),
        "source_fingerprint": source_fingerprint,
        "samples_per_shard": samples_per_shard,
        "docs_per_shard": docs_per_shard,
        "seed": seed,
        "max_samples": max_samples,
        "jpeg_quality": jpeg_quality,
    }
    if resume and state_path.is_file() and source_fingerprint is None:
        raise ValueError(
            f"Cannot safely resume normalized retrieval prep in {output_dir}: "
            "the loaded source does not provide verifiable content fingerprints. Choose a new output directory."
        )
    if state_path.is_file():
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        if existing_state != expected_state:
            raise ValueError(
                f"Cannot resume normalized retrieval prep in {output_dir}: "
                "the source or prep options changed, including the loaded source contents. "
                "Use the original inputs or choose a new output directory."
            )
        return

    has_existing_shards = any((output_dir / "train").glob("*.arrow")) or any((output_dir / "corpus").glob("**/*.arrow"))
    if resume and has_existing_shards:
        raise ValueError(
            f"Cannot safely resume normalized retrieval prep in {output_dir}: .resume-state.json is missing. "
            "Choose a new output directory."
        )
    _write_json_atomic(state_path, expected_state)


def _load_top_level_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Normalized retrieval metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "nemo_automodel_normalized_vl_retrieval_arrow":
        raise ValueError(f"Unsupported normalized retrieval format: {metadata.get('format')!r}")
    if "sources" not in metadata:
        raise ValueError(
            f"Cannot append to legacy normalized bundle without top-level sources metadata: {metadata_path}"
        )
    return metadata


def _source_dir_index(path: Path) -> int | None:
    if not path.name.startswith("source-"):
        return None
    suffix = path.name.removeprefix("source-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _next_source_index(output_dir: Path, metadata: dict[str, Any]) -> int:
    indices = [
        int(source["source_index"])
        for source in metadata.get("sources", [])
        if isinstance(source, dict) and "source_index" in source
    ]
    sources_dir = output_dir / "sources"
    if sources_dir.exists():
        indices.extend(index for path in sources_dir.iterdir() if (index := _source_dir_index(path)) is not None)
    return max(indices, default=-1) + 1


def _source_metadata_from_dir(output_dir: Path, source_idx: int, source_entry: Any) -> dict[str, Any]:
    source_dir = output_dir / "sources" / f"source-{source_idx:05d}"
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Normalized source metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("data_dir_list") != [source_entry]:
        raise ValueError(
            f"Normalized source {source_dir} was built from a different source entry. "
            "Choose a new output directory or rebuild the mismatched source."
        )
    return {
        "source_index": source_idx,
        "source_name": _source_entry_name(source_entry),
        "source_key": _source_entry_key(source_entry),
        "source_entry": source_entry,
        "path": str(source_dir.relative_to(output_dir)),
        "num_records": metadata["num_records"],
    }


def _existing_source_keys(metadata: dict[str, Any]) -> set[str]:
    keys = set()
    for source in metadata.get("sources", []):
        if not isinstance(source, dict):
            continue
        if "source_key" in source:
            keys.add(str(source["source_key"]))
        elif "source_entry" in source:
            keys.add(_source_entry_key(source["source_entry"]))
    return keys


def _safe_corpus_dir_name(corpus_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", corpus_id).strip("._-") or "corpus"
    digest = hashlib.sha256(corpus_id.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:48]}-{digest}"


def _corpus_output_dir(output_dir: Path, corpus_id: str) -> Path:
    corpus_root = (output_dir / "corpus").resolve()
    corpus_output_dir = (corpus_root / _safe_corpus_dir_name(corpus_id)).resolve()
    if corpus_output_dir.parent != corpus_root:
        raise ValueError(f"Corpus output path escapes the normalized corpus directory: {corpus_output_dir}")
    return corpus_output_dir


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
    completion_path = train_dir / _TRAIN_COMPLETION_FILENAME
    if not completion_path.is_file():
        if _arrow_shard_paths(train_dir, "train"):
            logger.warning("Existing train shards in %s have no completion manifest; rewriting them", train_dir)
        return None

    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected_paths = completion["shards"]
        expected_num_records = completion["num_records"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Train completion manifest is invalid in %s; rewriting train shards", train_dir, exc_info=True)
        return None
    if not isinstance(expected_paths, list) or not all(isinstance(path, str) for path in expected_paths):
        logger.warning("Train completion manifest has invalid shard paths in %s; rewriting train shards", train_dir)
        return None
    if not isinstance(expected_num_records, int) or expected_num_records < 0:
        logger.warning("Train completion manifest has an invalid record count in %s; rewriting train shards", train_dir)
        return None

    train_paths = _arrow_shard_paths(train_dir, "train")
    if [path.name for path in train_paths] != expected_paths:
        logger.warning("Train shards do not match the completion manifest in %s; rewriting them", train_dir)
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

    if num_records != expected_num_records:
        logger.warning(
            "Train shards contain %d records but the completion manifest expects %d in %s; rewriting them",
            num_records,
            expected_num_records,
            train_dir,
        )
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

    existing_ids: set[str] = set()
    num_records = 0
    try:
        for path in shard_paths:
            shard = datasets.Dataset.from_file(str(path))
            num_records += len(shard)
            existing_ids.update(str(doc_id) for doc_id in shard["id"])
    except Exception:
        logger.warning("Corpus %s has unreadable existing shards in %s; rewriting", corpus_id, corpus_output_dir)
        return None

    if num_records != len(existing_ids) or existing_ids != refs:
        logger.warning(
            "Corpus %s existing document IDs do not match the %d expected references; rewriting %d row(s)",
            corpus_id,
            len(refs),
            num_records,
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
    _write_json_atomic(
        output_dir / "train" / _TRAIN_COMPLETION_FILENAME,
        {"version": 1, "shards": writer.shard_paths, "num_records": writer.num_records},
    )
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
        corpus_output_dir = _corpus_output_dir(output_dir, corpus_id)
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
    _validate_or_write_resume_state(
        output_dir,
        data_dir_list,
        _loaded_source_fingerprint(dataset, corpus_dict),
        samples_per_shard=samples_per_shard,
        docs_per_shard=docs_per_shard,
        seed=seed,
        max_samples=max_samples,
        jpeg_quality=jpeg_quality,
        resume=resume,
    )
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


def _prepare_source_bundle(
    source_entry: Any,
    output_dir: Path,
    source_idx: int,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
    resume: bool,
    staged: bool,
) -> tuple[dict[str, Any], Path]:
    sources_dir = output_dir / "sources"
    source_dir = sources_dir / f"source-{source_idx:05d}"
    if staged:
        if source_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing normalized source directory: {source_dir}")
        temp_dir = sources_dir / f".source-{source_idx:05d}.tmp-{uuid.uuid4().hex}"
        try:
            metadata = _prepare_single_normalized_dataset(
                data_dir_list=[source_entry],
                output_dir=temp_dir,
                samples_per_shard=samples_per_shard,
                docs_per_shard=docs_per_shard,
                seed=seed,
                max_samples=max_samples,
                jpeg_quality=jpeg_quality,
                resume=False,
            )
            temp_dir.rename(source_dir)
        except BaseException:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    else:
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
    return (
        {
            "source_index": source_idx,
            "source_name": _source_entry_name(source_entry),
            "source_key": _source_entry_key(source_entry),
            "source_entry": source_entry,
            "path": str(source_dir.relative_to(output_dir)),
            "num_records": metadata["num_records"],
        },
        source_dir,
    )


def _append_normalized_sources(
    data_dir_list: list[Any],
    output_dir: Path,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
) -> dict[str, Any]:
    metadata = _load_top_level_metadata(output_dir)
    sources_dir = output_dir / "sources"
    if not sources_dir.is_dir():
        raise FileNotFoundError(f"Normalized retrieval sources directory not found: {sources_dir}")

    source_entries = _split_data_dir_list(data_dir_list)
    next_source_idx = _next_source_index(output_dir, metadata)
    existing_source_keys = _existing_source_keys(metadata)
    appended_sources = []
    appended_dirs = []
    try:
        source_idx = next_source_idx
        for source_entry in source_entries:
            source_key = _source_entry_key(source_entry)
            if source_key in existing_source_keys:
                logger.warning("Skipping duplicate normalized source entry during append: %s", source_entry)
                continue
            source_metadata, source_dir = _prepare_source_bundle(
                source_entry,
                output_dir,
                source_idx,
                samples_per_shard=samples_per_shard,
                docs_per_shard=docs_per_shard,
                seed=seed,
                max_samples=max_samples,
                jpeg_quality=jpeg_quality,
                resume=False,
                staged=True,
            )
            appended_sources.append(source_metadata)
            appended_dirs.append(source_dir)
            existing_source_keys.add(source_key)
            source_idx += 1

        if not appended_sources:
            logger.info("No new normalized retrieval sources to append to %s", output_dir)
            return metadata

        updated_metadata = dict(metadata)
        updated_sources = list(metadata.get("sources", [])) + appended_sources
        updated_metadata["sources"] = updated_sources
        updated_metadata["data_dir_list"] = _as_data_dir_list(metadata.get("data_dir_list")) + [
            source["source_entry"] for source in appended_sources
        ]
        updated_metadata["num_records"] = sum(int(source["num_records"]) for source in updated_sources)
        _write_json_atomic(output_dir / "metadata.json", updated_metadata)
        logger.info(
            "Appended %d source(s), %d total train records to normalized retrieval bundle %s",
            len(appended_sources),
            updated_metadata["num_records"],
            output_dir,
        )
        return updated_metadata
    except BaseException:
        for source_dir in reversed(appended_dirs):
            if source_dir.exists():
                shutil.rmtree(source_dir)
        raise


def _prepare_normalized_source(
    data_dir_list: list[Any],
    output_dir: Path,
    source_index: int,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
    seed: int,
    max_samples: int | None,
    jpeg_quality: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Write one source bundle under a top-level normalized output directory."""
    source_entries = _split_data_dir_list(data_dir_list)
    if source_index < 0 or source_index >= len(source_entries):
        raise ValueError(f"source_index must satisfy 0 <= source_index < {len(source_entries)}, got {source_index}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sources").mkdir(exist_ok=True)
    source_metadata, _ = _prepare_source_bundle(
        source_entries[source_index],
        output_dir,
        source_index,
        samples_per_shard=samples_per_shard,
        docs_per_shard=docs_per_shard,
        seed=seed,
        max_samples=max_samples,
        jpeg_quality=jpeg_quality,
        resume=resume,
        staged=False,
    )
    return source_metadata


def _finalize_normalized_sources(
    data_dir_list: list[Any],
    output_dir: Path,
    *,
    samples_per_shard: int,
    docs_per_shard: int,
) -> dict[str, Any]:
    """Write top-level metadata for source bundles prepared independently."""
    source_entries = _split_data_dir_list(data_dir_list)
    source_metadata = [
        _source_metadata_from_dir(output_dir, source_idx, source_entry)
        for source_idx, source_entry in enumerate(source_entries)
    ]
    metadata = {
        "format": "nemo_automodel_normalized_vl_retrieval_arrow",
        "version": 3,
        "data_dir_list": data_dir_list,
        "num_records": sum(int(source["num_records"]) for source in source_metadata),
        "samples_per_shard": samples_per_shard,
        "docs_per_shard": docs_per_shard,
        "sources": source_metadata,
    }
    _write_json_atomic(output_dir / "metadata.json", metadata)
    logger.info(
        "Finalized normalized retrieval bundle with %d source(s), %d total train records at %s",
        len(source_metadata),
        metadata["num_records"],
        output_dir,
    )
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
    append: bool = False,
) -> dict[str, Any]:
    """Write a normalized portable Arrow retrieval bundle."""
    if resume and append:
        raise ValueError("--resume and --append are mutually exclusive")
    if append:
        return _append_normalized_sources(
            data_dir_list,
            output_dir,
            samples_per_shard=samples_per_shard,
            docs_per_shard=docs_per_shard,
            seed=seed,
            max_samples=max_samples,
            jpeg_quality=jpeg_quality,
        )

    source_entries = _split_data_dir_list(data_dir_list)
    _prepare_multi_source_output_dir(output_dir, resume=resume)
    for source_idx in range(len(source_entries)):
        _prepare_normalized_source(
            data_dir_list,
            output_dir,
            source_idx,
            samples_per_shard=samples_per_shard,
            docs_per_shard=docs_per_shard,
            seed=seed,
            max_samples=max_samples,
            jpeg_quality=jpeg_quality,
            resume=resume,
        )

    if max_samples is not None and len(source_entries) > 1:
        logger.warning(
            "--max-samples is applied to each normalized source independently. "
            "Use per-source num_samples in the original config or at training time for source-specific caps."
        )

    return _finalize_normalized_sources(
        data_dir_list,
        output_dir,
        samples_per_shard=samples_per_shard,
        docs_per_shard=docs_per_shard,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="AutoModel bi-encoder YAML. Uses dataset.data_dir_list (or legacy dataloader.dataset).",
    )
    parser.add_argument("--data-dir-list", nargs="+", default=None, help="Corpus-id retrieval JSON files to bundle")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized Arrow train/corpus shards")
    parser.add_argument("--samples-per-shard", type=int, default=10000, help="Train examples per Arrow shard")
    parser.add_argument("--docs-per-shard", type=int, default=10000, help="Corpus documents per Arrow shard")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for source sampling, if configured")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global train sample cap")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for packed corpus images")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified completed output from an interrupted preparation run",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Prepare one data_dir_list source for a Slurm array, then run --finalize-sources.",
    )
    parser.add_argument(
        "--finalize-sources",
        action="store_true",
        help="Write top-level metadata from independently prepared source directories.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the provided source(s) to an existing top-level normalized bundle.",
    )
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

    if args.source_index is not None and args.finalize_sources:
        raise ValueError("Use only one of --source-index or --finalize-sources")
    if args.append and (args.source_index is not None or args.finalize_sources):
        raise ValueError("--append cannot be combined with --source-index or --finalize-sources")

    if args.source_index is not None:
        metadata = _prepare_normalized_source(
            data_dir_list=data_dir_list,
            output_dir=Path(args.output_dir),
            source_index=args.source_index,
            samples_per_shard=args.samples_per_shard,
            docs_per_shard=args.docs_per_shard,
            seed=args.seed,
            max_samples=args.max_samples,
            jpeg_quality=args.jpeg_quality,
            resume=args.resume,
        )
    elif args.finalize_sources:
        metadata = _finalize_normalized_sources(
            data_dir_list=data_dir_list,
            output_dir=Path(args.output_dir),
            samples_per_shard=args.samples_per_shard,
            docs_per_shard=args.docs_per_shard,
        )
    else:
        metadata = prepare_normalized_dataset(
            data_dir_list=data_dir_list,
            output_dir=Path(args.output_dir),
            samples_per_shard=args.samples_per_shard,
            docs_per_shard=args.docs_per_shard,
            seed=args.seed,
            max_samples=args.max_samples,
            jpeg_quality=args.jpeg_quality,
            resume=args.resume,
            append=args.append,
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
