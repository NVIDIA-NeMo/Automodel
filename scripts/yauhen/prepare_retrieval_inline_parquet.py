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

"""Prepare corpus-id retrieval JSON as inline-text Parquet shards.

This script is intended to run as an offline CPU/data preparation step before
launching GPU training. It converts the corpus-id format used by
``retrieval_dataset.py`` into the inline-text format consumed by
``retrieval_dataset_inline.py`` so training does not spend startup time parsing a
multi-GB JSON file or indexing the full corpus.


python scripts/yauhen/prepare_retrieval_inline_parquet.py \
  --input-json /home/bschifferer/llm2/team_folder/01_datasets/03_training_datasets/multilingual_wiki_ccnews_5hn_withMoreDS_more_reddit_s2orc_allLang_12M.json \
  --output-dir /lustre/fsw/portfolios/llmservice/users/ybabakhin/data/rteb/multilingual_wiki_ccnews_5hn_withMoreDS_more_reddit_s2orc_allLang_12M_inline_nneg4 \
  --corpus-index-path /lustre/fsw/portfolios/llmservice/users/ybabakhin/data/rteb/corpus/multilingual_wiki_ccnews_5hn_withMoreDS_more_reddit_s2orc_allLang_12M_docs.sqlite \
  --num-negatives 4 \
  --records-per-shard 50000


python scripts/yauhen/prepare_retrieval_inline_parquet.py \
  --input-json /lustre/fsw/portfolios/llmservice/users/ybabakhin/data/nv-embed-v2-non-retrieval/training_datasets/nq.json \
  --output-dir /lustre/fsw/portfolios/llmservice/users/ybabakhin/data/rteb/nq_inline_nneg4 \
  --corpus-index-path /lustre/fsw/portfolios/llmservice/users/ybabakhin/data/rteb/corpus/nq_inline.sqlite \
  --num-negatives 4 \
  --records-per-shard 50000

"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nemo_automodel.shared.import_utils import safe_import

_HAVE_PA, pa = safe_import("pyarrow", msg="pyarrow is required to write inline retrieval Parquet shards.")
_HAVE_PQ, pq = safe_import("pyarrow.parquet", msg="pyarrow.parquet is required to read and write Parquet shards.")

LOGGER = logging.getLogger(__name__)


def _require_pyarrow() -> None:
    if not _HAVE_PA or not _HAVE_PQ:
        raise RuntimeError("pyarrow and pyarrow.parquet are required to prepare inline retrieval Parquet shards.")


def _discover_corpus_dir(input_json: Path) -> Path:
    """Read the top-level ``corpus.path`` field without loading the JSON data array."""
    header_lines = []
    with input_json.open("r", encoding="utf-8") as f:
        for line in f:
            if '"data"' in line:
                break
            header_lines.append(line)

    header = "".join(header_lines).rstrip()
    if header.endswith(","):
        header = header[:-1].rstrip()
    header = f"{header}\n}}"
    payload = json.loads(header)
    corpus = payload["corpus"]
    if isinstance(corpus, list):
        if len(corpus) != 1:
            raise ValueError("Automatic corpus discovery expects exactly one corpus entry.")
        corpus = corpus[0]
    return Path(corpus["path"])


def _iter_training_records(
    input_json: Path, max_records: int | None = None, chunk_size: int = 1 << 20
) -> Iterator[dict]:
    """Yield records from top-level ``data`` without materializing the full JSON file."""
    decoder = json.JSONDecoder()
    yielded = 0
    with input_json.open("r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find top-level 'data' array in {input_json}")
            buffer += chunk
            data_key_idx = buffer.find('"data"')
            if data_key_idx < 0:
                continue
            array_start_idx = buffer.find("[", data_key_idx)
            if array_start_idx < 0:
                continue
            buffer = buffer[array_start_idx + 1 :]
            break

        while True:
            if max_records is not None and yielded >= max_records:
                return

            buffer = buffer.lstrip()
            while not buffer:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buffer += chunk
                buffer = buffer.lstrip()

            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue

            while True:
                try:
                    record, end_idx = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise ValueError(f"Failed to parse JSON record near record {yielded}: {exc}") from exc
                    buffer += chunk

            yielded += 1
            yield record
            buffer = buffer[end_idx:]


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if files:
            return files
    raise ValueError(f"Could not find Parquet files in corpus path: {path}")


def _remove_sqlite_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _sqlite_has_docs_table(index_path: Path) -> bool:
    if not index_path.exists():
        return False
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'docs'").fetchone()
        return row is not None
    finally:
        conn.close()


def build_corpus_index(
    corpus_dir: Path,
    index_path: Path,
    *,
    batch_size: int,
    rebuild: bool = False,
) -> None:
    """Build or reuse a SQLite ``doc_id -> text`` index for the corpus Parquet."""
    _require_pyarrow()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if _sqlite_has_docs_table(index_path) and not rebuild:
        LOGGER.info("Reusing existing corpus index: %s", index_path)
        return
    if index_path.exists():
        if not rebuild:
            raise ValueError(f"Corpus index exists but is invalid. Use --rebuild-corpus-index: {index_path}")
        _remove_sqlite_files(index_path)

    parquet_files = _parquet_files(corpus_dir)
    LOGGER.info("Building corpus index at %s from %d Parquet file(s)", index_path, len(parquet_files))
    conn = sqlite3.connect(index_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -1048576")
        conn.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, text TEXT NOT NULL) WITHOUT ROWID")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")

        row_count = 0
        for parquet_file in parquet_files:
            parquet_reader = pq.ParquetFile(parquet_file)
            for batch in parquet_reader.iter_batches(batch_size=batch_size, columns=["id", "text"]):
                ids = batch.column("id").to_pylist()
                texts = batch.column("text").to_pylist()
                conn.executemany(
                    "INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)",
                    ((str(doc_id), "" if text is None else str(text)) for doc_id, text in zip(ids, texts)),
                )
                row_count += len(ids)
                if row_count % 1_000_000 == 0:
                    conn.commit()
                    LOGGER.info("Indexed %d corpus rows", row_count)

        conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("corpus_dir", str(corpus_dir)))
        conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("row_count", str(row_count)))
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("created_at", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        LOGGER.info("Finished corpus index with %d rows", row_count)
    finally:
        conn.close()


def _doc_id(doc: Any) -> str:
    if isinstance(doc, dict) and "id" in doc:
        return str(doc["id"])
    return str(doc)


def _normalize_record(record: dict, *, num_positives: int, num_negatives: int) -> dict:
    required_fields = ("question_id", "question", "corpus_id", "pos_doc", "neg_doc")
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Missing required field(s) {missing} in record: {record}")

    pos_ids = [_doc_id(doc) for doc in record["pos_doc"][:num_positives]]
    neg_ids = [_doc_id(doc) for doc in record["neg_doc"][:num_negatives]]
    if not pos_ids:
        raise ValueError(f"Record has no positive document: {record['question_id']}")

    return {
        "question_id": str(record["question_id"]),
        "question": "" if record["question"] is None else str(record["question"]),
        "corpus_id": str(record["corpus_id"]),
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
    }


def _fetch_texts(conn: sqlite3.Connection, doc_ids: list[str], *, fetch_size: int) -> dict[str, str]:
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    doc_text: dict[str, str] = {}
    for start_idx in range(0, len(unique_doc_ids), fetch_size):
        chunk = unique_doc_ids[start_idx : start_idx + fetch_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(f"SELECT id, text FROM docs WHERE id IN ({placeholders})", chunk)
        doc_text.update((str(doc_id), "" if text is None else str(text)) for doc_id, text in rows)

    missing = [doc_id for doc_id in unique_doc_ids if doc_id not in doc_text]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(f"Missing {len(missing)} document id(s) in corpus index. First missing ids: {preview}")
    return doc_text


def _doc_payload(doc_id: str, text: str) -> dict[str, str]:
    return {
        "id": doc_id,
        "text": text,
        "image": "",
        "nr_ocr": "",
    }


def _inline_schema() -> Any:
    doc_type = pa.struct(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("image", pa.string()),
            ("nr_ocr", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("question_id", pa.string()),
            ("question", pa.string()),
            ("corpus_id", pa.string()),
            ("pos_doc", pa.list_(doc_type)),
            ("neg_doc", pa.list_(doc_type)),
        ]
    )


def _write_shard(
    records: list[dict],
    *,
    conn: sqlite3.Connection,
    output_dir: Path,
    shard_idx: int,
    fetch_size: int,
    compression: str,
) -> Path:
    _require_pyarrow()
    doc_ids = [doc_id for record in records for doc_id in record["pos_ids"] + record["neg_ids"]]
    doc_text = _fetch_texts(conn, doc_ids, fetch_size=fetch_size)
    rows = []
    for record in records:
        rows.append(
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "corpus_id": record["corpus_id"],
                "pos_doc": [_doc_payload(doc_id, doc_text[doc_id]) for doc_id in record["pos_ids"]],
                "neg_doc": [_doc_payload(doc_id, doc_text[doc_id]) for doc_id in record["neg_ids"]],
            }
        )

    output_path = output_dir / f"part-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows, schema=_inline_schema())
    pq.write_table(table, output_path, compression=compression)
    LOGGER.info("Wrote %s with %d records", output_path, len(records))
    return output_path


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"Output directory is not empty. Use --overwrite-output to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _materialize_hf_dataset(parquet_files: list[Path], hf_dataset_dir: Path, overwrite: bool) -> None:
    have_datasets, datasets_module = safe_import("datasets", msg="datasets is required for --hf-dataset-dir.")
    if not have_datasets:
        raise RuntimeError("datasets is required for --hf-dataset-dir.")
    if hf_dataset_dir.exists() and any(hf_dataset_dir.iterdir()):
        if not overwrite:
            raise ValueError(
                f"HuggingFace dataset directory is not empty. Use --overwrite-output to replace it: {hf_dataset_dir}"
            )
        shutil.rmtree(hf_dataset_dir)
    hf_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets_module.load_dataset(
        "parquet",
        data_files=[str(path) for path in parquet_files],
        split="train",
    )
    dataset.save_to_disk(str(hf_dataset_dir))
    LOGGER.info("Saved HuggingFace dataset to %s", hf_dataset_dir)


def prepare_inline_parquet(args: argparse.Namespace) -> None:
    """Prepare inline retrieval Parquet shards from a corpus-id JSON file."""
    if args.num_positives <= 0:
        raise ValueError("--num-positives must be greater than 0")
    if args.num_negatives < 0:
        raise ValueError("--num-negatives must be greater than or equal to 0")
    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be greater than 0")

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else _discover_corpus_dir(input_json)
    corpus_index_path = Path(args.corpus_index_path) if args.corpus_index_path else output_dir / "corpus_index.sqlite"

    _prepare_output_dir(output_dir, args.overwrite_output)
    build_corpus_index(
        corpus_dir,
        corpus_index_path,
        batch_size=args.corpus_batch_size,
        rebuild=args.rebuild_corpus_index,
    )

    conn = sqlite3.connect(f"file:{corpus_index_path}?mode=ro", uri=True)
    written_files: list[Path] = []
    total_records = 0
    shard_idx = 0
    buffer: list[dict] = []
    try:
        for record in _iter_training_records(input_json, max_records=args.max_records):
            buffer.append(
                _normalize_record(
                    record,
                    num_positives=args.num_positives,
                    num_negatives=args.num_negatives,
                )
            )
            if len(buffer) >= args.records_per_shard:
                written_files.append(
                    _write_shard(
                        buffer,
                        conn=conn,
                        output_dir=output_dir,
                        shard_idx=shard_idx,
                        fetch_size=args.sqlite_fetch_size,
                        compression=args.compression,
                    )
                )
                total_records += len(buffer)
                buffer = []
                shard_idx += 1

        if buffer:
            written_files.append(
                _write_shard(
                    buffer,
                    conn=conn,
                    output_dir=output_dir,
                    shard_idx=shard_idx,
                    fetch_size=args.sqlite_fetch_size,
                    compression=args.compression,
                )
            )
            total_records += len(buffer)
    finally:
        conn.close()

    manifest = {
        "input_json": str(input_json),
        "corpus_dir": str(corpus_dir),
        "corpus_index_path": str(corpus_index_path),
        "output_dir": str(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_records": total_records,
        "num_positives": args.num_positives,
        "num_negatives": args.num_negatives,
        "records_per_shard": args.records_per_shard,
        "compression": args.compression,
        "parquet_files": [str(path) for path in written_files],
        "training_target": "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote manifest to %s", manifest_path)

    if args.hf_dataset_dir:
        _materialize_hf_dataset(written_files, Path(args.hf_dataset_dir), args.overwrite_output)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, help="Corpus-id retrieval JSON to convert.")
    parser.add_argument("--output-dir", required=True, help="Directory where inline Parquet shards will be written.")
    parser.add_argument(
        "--corpus-dir", default=None, help="Corpus Parquet directory. Defaults to top-level corpus.path."
    )
    parser.add_argument(
        "--corpus-index-path",
        default=None,
        help="SQLite doc-id index path. Defaults to <output-dir>/corpus_index.sqlite.",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Number of hard negatives to inline per query. Use 4 to match n_passages=5.",
    )
    parser.add_argument("--num-positives", type=int, default=1, help="Number of positives to inline per query.")
    parser.add_argument("--records-per-shard", type=int, default=50_000, help="Rows per output Parquet shard.")
    parser.add_argument("--corpus-batch-size", type=int, default=100_000, help="Rows per corpus indexing batch.")
    parser.add_argument("--sqlite-fetch-size", type=int, default=5_000, help="Document IDs per SQLite fetch.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke tests or subsets.")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--rebuild-corpus-index", action="store_true", help="Rebuild an existing SQLite corpus index.")
    parser.add_argument(
        "--hf-dataset-dir",
        default=None,
        help="Optional Dataset.save_to_disk output path materialized from the Parquet shards.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the data preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prepare_inline_parquet(parse_args())


if __name__ == "__main__":
    main()
