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

"""Prepare AfriInstruct for text-only Gemma 4 fine-tuning.

The utility streams ``llama-lang-adapt/AfriInstruct-Data``, converts each
instruction/output pair to the conversation schema consumed by
``make_meta_dataset``, writes bounded JSONL shards, and creates train and
validation meta manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparationConfig:
    dataset_id: str
    dataset_config: str | None
    source_split: str
    output_dir: Path
    validation_fraction: float
    sample_fraction: float
    seed: int
    shard_size: int
    max_samples: int | None
    languages: frozenset[str]
    tasks: frozenset[str]
    sources: frozenset[str]


@dataclass
class _PreparationStats:
    input_records: int = 0
    malformed_records: int = 0
    duplicate_records: int = 0
    filtered_records: int = 0
    sampled_out_records: int = 0
    train_records: int = 0
    validation_records: int = 0
    by_language: Counter[str] = field(default_factory=Counter)
    by_task: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)

    @property
    def written_records(self) -> int:
        return self.train_records + self.validation_records

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable preparation statistics."""
        return {
            "input_records": self.input_records,
            "malformed_records": self.malformed_records,
            "duplicate_records": self.duplicate_records,
            "filtered_records": self.filtered_records,
            "sampled_out_records": self.sampled_out_records,
            "train_records": self.train_records,
            "validation_records": self.validation_records,
            "written_records": self.written_records,
            "by_language": dict(sorted(self.by_language.items())),
            "by_task": dict(sorted(self.by_task.items())),
            "by_source": dict(sorted(self.by_source.items())),
        }


@dataclass
class _ShardState:
    shard_index: int = 0
    records_in_shard: int = 0


class _ExactDeduplicator:
    """Disk-backed exact deduplication for instruction/output pairs."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        self._pending = 0

    def add(self, instruction: str, output: str) -> bool:
        """Record a pair and return whether it had not been seen before."""
        digest = hashlib.blake2b(f"{instruction}\x1f{output}".encode("utf-8"), digest_size=16).digest()
        cursor = self._connection.execute("INSERT OR IGNORE INTO seen (digest) VALUES (?)", (digest,))
        self._pending += 1
        if self._pending >= 10_000:
            self._connection.commit()
            self._pending = 0
        return cursor.rowcount == 1

    def close(self) -> None:
        """Commit pending entries and close the database."""
        self._connection.commit()
        self._connection.close()


class _ShardWriter:
    """Write partitioned JSONL shards and build matching meta manifests."""

    def __init__(self, output_dir: Path, shard_size: int, max_open_files: int = 64) -> None:
        self._output_dir = output_dir
        self._shard_size = shard_size
        self._max_open_files = max_open_files
        self._states: dict[tuple[str, str, str], _ShardState] = {}
        self._handles: OrderedDict[Path, TextIO] = OrderedDict()
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "validation": {}}

    def write(self, split: str, language: str, task: str, record: Mapping[str, Any]) -> None:
        """Write one conversation record to its language/task shard."""
        partition = _partition_name(language, task)
        state_key = (split, language, task)
        state = self._states.setdefault(state_key, _ShardState())
        if state.records_in_shard >= self._shard_size:
            state.shard_index += 1
            state.records_in_shard = 0

        relative_path = Path("processed") / split / f"{partition}-{state.shard_index:05d}.jsonl"
        absolute_path = self._output_dir / relative_path
        if state.records_in_shard == 0:
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_name = f"{partition}-{state.shard_index:05d}"
            self.manifests[split][manifest_name] = {
                "file_name": relative_path.as_posix(),
                "columns": {"messages": "messages"},
                "sample_ratio": 1.0,
            }

        handle = self._get_handle(absolute_path)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        state.records_in_shard += 1

    def _get_handle(self, path: Path) -> TextIO:
        handle = self._handles.pop(path, None)
        if handle is None:
            if len(self._handles) >= self._max_open_files:
                _, oldest_handle = self._handles.popitem(last=False)
                oldest_handle.close()
            handle = path.open("a", encoding="utf-8")
        self._handles[path] = handle
        return handle

    def close(self) -> None:
        """Close all open shard handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:48] or "unknown"


def _partition_name(language: str, task: str) -> str:
    original = f"{language}\x1f{task}"
    suffix = hashlib.blake2b(original.encode("utf-8"), digest_size=4).hexdigest()
    return f"{_slug(language)}__{_slug(task)}__{suffix}"


def _record_hash_fraction(record: Mapping[str, Any], seed: int, namespace: str) -> float:
    identity = "\x1f".join(str(record.get(key, "")) for key in ("instruction", "output", "lang", "task", "source"))
    digest = hashlib.blake2b(
        identity.encode("utf-8"),
        digest_size=8,
        person=f"{namespace}:{seed}".encode("utf-8")[:16],
    ).digest()
    return int.from_bytes(digest, byteorder="big") / 2**64


def _is_validation_record(record: Mapping[str, Any], validation_fraction: float, seed: int) -> bool:
    return _record_hash_fraction(record, seed, "validation") < validation_fraction


def _is_sampled_record(record: Mapping[str, Any], sample_fraction: float, seed: int) -> bool:
    return sample_fraction == 1.0 or _record_hash_fraction(record, seed, "sample") < sample_fraction


def _normalize_record(raw_record: Mapping[str, Any]) -> dict[str, Any] | None:
    instruction = raw_record.get("instruction")
    output = raw_record.get("output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        return None

    instruction = instruction.strip()
    output = output.strip()
    if not instruction or not output:
        return None

    language = str(raw_record.get("lang") or "unknown").strip()
    task = str(raw_record.get("task") or "unknown").strip()
    source = str(raw_record.get("source") or "unknown").strip()
    return {
        "instruction": instruction,
        "output": output,
        "lang": language,
        "task": task,
        "source": source,
    }


def _passes_filters(record: Mapping[str, Any], config: _PreparationConfig) -> bool:
    return (
        (not config.languages or record["lang"] in config.languages)
        and (not config.tasks or record["task"] in config.tasks)
        and (not config.sources or record["source"] in config.sources)
    )


def _to_conversation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ],
        "lang": record["lang"],
        "task": record["task"],
        "source": record["source"],
    }


def _load_source(config: _PreparationConfig) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": config.dataset_id,
        "split": config.source_split,
        "streaming": True,
    }
    if config.dataset_config is not None:
        kwargs["name"] = config.dataset_config
    return load_dataset(**kwargs)


def _prepare_dataset(config: _PreparationConfig, records: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {config.output_dir}. "
            "Use a new versioned directory for each preparation run."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    stats = _PreparationStats()
    writer = _ShardWriter(config.output_dir, config.shard_size)
    database_path = config.output_dir / ".dedup.sqlite3"
    deduplicator = _ExactDeduplicator(database_path)

    try:
        for raw_record in records if records is not None else _load_source(config):
            stats.input_records += 1
            record = _normalize_record(raw_record)
            if record is None:
                stats.malformed_records += 1
                continue
            if not _passes_filters(record, config):
                stats.filtered_records += 1
                continue
            if not _is_sampled_record(record, config.sample_fraction, config.seed):
                stats.sampled_out_records += 1
                continue
            if not deduplicator.add(record["instruction"], record["output"]):
                stats.duplicate_records += 1
                continue

            split = "validation" if _is_validation_record(record, config.validation_fraction, config.seed) else "train"
            writer.write(split, record["lang"], record["task"], _to_conversation(record))
            if split == "validation":
                stats.validation_records += 1
            else:
                stats.train_records += 1
            stats.by_language[record["lang"]] += 1
            stats.by_task[record["task"]] += 1
            stats.by_source[record["source"]] += 1

            if config.max_samples is not None and stats.written_records >= config.max_samples:
                break
    finally:
        writer.close()
        deduplicator.close()
        database_path.unlink(missing_ok=True)

    if stats.written_records == 0:
        raise ValueError("No records were written. Check the source schema and filter values.")

    for split, manifest in writer.manifests.items():
        manifest_path = config.output_dir / f"{split}_meta.json"
        manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "source_split": config.source_split,
        "validation_fraction": config.validation_fraction,
        "sample_fraction": config.sample_fraction,
        "seed": config.seed,
        "shard_size": config.shard_size,
        "max_samples": config.max_samples,
        "filters": {
            "languages": sorted(config.languages),
            "tasks": sorted(config.tasks),
            "sources": sorted(config.sources),
        },
        "stats": stats.to_dict(),
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Prepared %d records (%d train, %d validation) in %s",
        stats.written_records,
        stats.train_records,
        stats.validation_records,
        config.output_dir,
    )
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _validation_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("validation fraction must be between zero and one")
    return parsed


def _sample_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("sample fraction must be greater than zero and at most one")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AfriInstruct JSONL shards for NeMo AutoModel.")
    parser.add_argument("--dataset-id", default="llama-lang-adapt/AfriInstruct-Data")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=_validation_fraction, default=0.01)
    parser.add_argument("--sample-fraction", type=_sample_fraction, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=_positive_int, default=100_000)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument("--languages", nargs="*", default=())
    parser.add_argument("--tasks", nargs="*", default=())
    parser.add_argument("--sources", nargs="*", default=())
    return parser


def main() -> int:
    """Prepare AfriInstruct according to command-line arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    config = _PreparationConfig(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        source_split=args.source_split,
        output_dir=args.output_dir.resolve(),
        validation_fraction=args.validation_fraction,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        languages=frozenset(args.languages),
        tasks=frozenset(args.tasks),
        sources=frozenset(args.sources),
    )
    _prepare_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
