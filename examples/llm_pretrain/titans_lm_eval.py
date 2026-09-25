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

#!/usr/bin/env python3
"""Correctness-first standard language-model evaluation for Titans checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

DATASETS = {
    "wikitext103": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "text_field": "text",
    },
    "lambada": {
        "path": "EleutherAI/lambada_openai",
        "name": "en",
        "split": "test",
        "revision": "900124bf3b8235c6daf21033af9948b3f07346c4",
        "text_field": "text",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "text_field": "text",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def partial_path(output_dir: Path, benchmark: str, start: int, end: int | None) -> Path:
    """Return the non-colliding raw-partial path for a requested source range."""
    end_label = "end" if end is None else f"{end:012d}"
    return output_dir / "raw" / f"{benchmark}.{start:012d}-{end_label}.jsonl"


def metric_totals(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token-weighted causal-NLL and benchmark-specific metrics."""
    nll_sum = 0.0
    token_count = 0
    byte_count = 0
    samples = 0
    scored_samples = 0
    truncated_samples = 0
    exact_sum = 0
    token_exact_sum = 0
    exact_count = 0
    for record in records:
        samples += 1
        nll_sum += float(record["nll_sum"])
        token_count += int(record["token_count"])
        byte_count += int(record["byte_count"])
        scored_samples += int(record["token_count"] > 0)
        truncated_samples += int(record.get("truncated", False))
        if record.get("last_word_exact") is not None:
            exact_sum += int(record["last_word_exact"])
            token_exact_sum += int(record["last_word_token_exact"])
            exact_count += 1

    mean_nll = nll_sum / token_count if token_count else None
    return {
        "samples": samples,
        "scored_samples": scored_samples,
        "truncated_samples": truncated_samples,
        "nll_sum": nll_sum,
        "token_count": token_count,
        "byte_count": byte_count,
        "loss": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll is not None else None,
        "bits_per_byte": nll_sum / (byte_count * math.log(2)) if byte_count else None,
        "last_word_exact_accuracy": exact_sum / exact_count if exact_count else None,
        "last_word_token_exact_accuracy": token_exact_sum / exact_count if exact_count else None,
        "last_word_samples": exact_count,
    }


def _last_word(text: str) -> tuple[int, str]:
    match = re.search(r"(\S+)\s*$", text)
    if match is None:
        raise ValueError("LAMBADA text has no non-whitespace last word")
    return match.start(1), match.group(1)


def lambada_token_boundary(tokenizer: Any, text: str) -> dict[str, Any]:
    """Split LAMBADA at the first token overlapping the final lexical word."""
    word_start, answer = _last_word(text)
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    target_start = next(
        (index for index, (start, end) in enumerate(offsets) if end > word_start and end > start),
        None,
    )
    if target_start is None or target_start == 0:
        raise ValueError("Could not derive a non-empty LAMBADA prompt from tokenizer offsets")
    return {
        "input_ids": input_ids,
        "target_start": target_start,
        "prompt_ids": input_ids[:target_start],
        "target_ids": input_ids[target_start:],
        "answer": answer,
        "word_start": word_start,
        "boundary_offset": offsets[target_start],
    }


def normalize_last_word(text: str) -> str:
    """Return the final non-whitespace span used for exact LAMBADA matching."""
    match = re.search(r"(\S+)\s*$", text)
    return match.group(1) if match is not None else ""


def lambada_exact_result(
    tokenizer: Any, generated_ids: list[int], target_ids: list[int], answer: str
) -> dict[str, Any]:
    """Compare generated continuation by both token identity and decoded final word."""
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "last_word_exact": normalize_last_word(generated_text) == answer,
        "last_word_token_exact": generated_ids == target_ids,
        "generated_continuation": generated_text,
        "expected_last_word": answer,
    }


def _load_local(path: Path, text_field: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if text_field not in row:
                    raise ValueError(f"{path}:{line_number}: missing {text_field!r}")
                yield row
        else:
            for line in stream:
                yield {text_field: line.rstrip("\n")}


def iter_source(
    benchmark: str,
    input_file: Path | None,
    *,
    start: int,
    end: int | None,
    text_field: str,
) -> Iterator[tuple[int, str]]:
    """Yield stable source indices and text from a local fixture or pinned HF dataset."""
    if input_file is not None:
        source: Iterable[dict[str, Any]] = _load_local(input_file, text_field)
    else:
        from datasets import load_dataset

        metadata = DATASETS[benchmark]
        source = load_dataset(
            metadata["path"],
            metadata["name"],
            split=metadata["split"],
            revision=metadata["revision"],
            streaming=True,
        )

    for index, row in enumerate(source):
        if index < start:
            continue
        if end is not None and index >= end:
            break
        text = row[text_field]
        if not isinstance(text, str):
            raise TypeError(f"Source index {index} field {text_field!r} is not text")
        yield index, text


def causal_nll(logits: torch.Tensor, input_ids: torch.Tensor, score_start: int = 1) -> tuple[float, int]:
    """Return summed next-token NLL and count for targets at or after score_start."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes must match")
    first_target = max(1, score_start)
    if first_target >= input_ids.shape[1]:
        return 0.0, 0
    token_losses = F.cross_entropy(
        logits[:, first_target - 1 : -1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, first_target:].reshape(-1),
        reduction="none",
    )
    return float(token_losses.double().sum().item()), int(token_losses.numel())


@torch.no_grad()
def evaluate_text(generator: Any, benchmark: str, index: int, text: str, max_tokens: int | None) -> dict[str, Any]:
    """Evaluate one independent document, resetting Titans fast memory at its boundary."""
    tokenizer = generator.tokenizer
    boundary = lambada_token_boundary(tokenizer, text) if benchmark == "lambada" and text.strip() else None
    if boundary is None:
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        score_start = 1
    else:
        input_ids = torch.tensor([boundary["input_ids"]], dtype=torch.long)
        score_start = int(boundary["target_start"])

    original_tokens = input_ids.shape[1]
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    truncated = input_ids.shape[1] < original_tokens
    input_ids = input_ids.to(generator.device)

    if input_ids.shape[1] < 2:
        nll_sum, token_count = 0.0, 0
    else:
        logits = generator.model(
            input_ids=input_ids,
            enable_ttt_updates=generator.enable_ttt_updates,
        ).logits
        nll_sum, token_count = causal_nll(logits, input_ids, score_start)
    scored_ids = input_ids[0, max(1, score_start) :].tolist()
    byte_count = len(tokenizer.decode(scored_ids, skip_special_tokens=True).encode("utf-8"))

    record: dict[str, Any] = {
        "index": index,
        "nll_sum": nll_sum,
        "token_count": token_count,
        "byte_count": byte_count,
        "input_tokens": int(input_ids.shape[1]),
        "original_tokens": original_tokens,
        "truncated": truncated,
        "last_word_exact": None,
        "last_word_token_exact": None,
    }
    if boundary is not None and not truncated:
        prompt_ids = torch.tensor([boundary["prompt_ids"]], dtype=torch.long, device=generator.device)
        generated = generator.model.generate_full_prefix(
            prompt_ids,
            max_new_tokens=len(boundary["target_ids"]),
            enable_ttt_updates=generator.enable_ttt_updates,
        )
        generated_ids = generated[0, prompt_ids.shape[1] :].tolist()
        record.update(lambada_exact_result(tokenizer, generated_ids, boundary["target_ids"], boundary["answer"]))
        record["target_token_count"] = len(boundary["target_ids"])
        record["target_boundary_offset"] = list(boundary["boundary_offset"])
    return record


@torch.no_grad()
def evaluate_token_window(generator: Any, index: int, input_ids: torch.Tensor) -> dict[str, Any]:
    """Evaluate one pre-tokenized causal-LM window with independent fast memory."""
    input_ids = input_ids.to(dtype=torch.long, device=generator.device).unsqueeze(0)
    logits = generator.model(
        input_ids=input_ids,
        enable_ttt_updates=generator.enable_ttt_updates,
    ).logits
    nll_sum, token_count = causal_nll(logits, input_ids)
    scored_ids = input_ids[0, 1:].tolist()
    return {
        "index": index,
        "nll_sum": nll_sum,
        "token_count": token_count,
        "byte_count": len(generator.tokenizer.decode(scored_ids, skip_special_tokens=True).encode("utf-8")),
        "input_tokens": int(input_ids.shape[1]),
        "original_tokens": int(input_ids.shape[1]),
        "truncated": False,
        "last_word_exact": None,
        "last_word_token_exact": None,
    }


def iter_token_windows(
    path: Path,
    *,
    sequence_length: int,
    start: int,
    end: int | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield overlapping pre-tokenized windows that score every token once."""
    from nemo_automodel.components.datasets.llm.nanogpt_dataset import load_bin_shard

    tokens = load_bin_shard(path)
    stride = sequence_length - 1
    window_count = max(0, math.ceil((len(tokens) - 1) / stride))
    stop = window_count if end is None else min(end, window_count)
    for index in range(start, stop):
        offset = index * stride
        window = tokens[offset : min(offset + sequence_length, len(tokens))]
        if len(window) >= 2:
            yield index, window


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    dataset = dict(DATASETS[args.benchmark])
    if args.input_bin is not None:
        dataset = {
            "path": str(args.input_bin),
            "sha256": _sha256_file(args.input_bin),
            "name": "NanoGPT token shard",
            "split": "validation",
            "revision": None,
            "text_field": None,
        }
    elif args.input_file is not None:
        dataset = {
            "path": str(args.input_file),
            "sha256": _sha256_file(args.input_file),
            "name": None,
            "split": "local",
            "revision": None,
            "text_field": args.text_field,
        }
    return {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer": args.tokenizer,
        "dataset": dataset,
        "range": {"start": args.start, "end": args.end},
        "max_tokens": args.max_tokens,
        "enable_ttt_updates": not args.disable_ttt_updates,
        "document_policy": (
            "Pre-tokenized input uses max_tokens windows with max_tokens-1 stride; Titans fast memory resets per window."
            if args.input_bin is not None
            else "Each source row is an independent document; Titans fast memory resets per row."
        ),
        "nll_policy": "Natural-log causal NLL, summed over scored target tokens and token-count weighted at merge.",
        "byte_policy": "UTF-8 bytes of tokenizer-decoded scored target IDs; bits_per_byte=nll_sum/(bytes*ln(2)).",
        "lambada_policy": (
            "NLL scores final-word tokens. The target begins at the first offset token overlapping the final "
            "non-whitespace word; exact accuracy compares decoded generated final word, with token exact also reported."
        ),
    }


def evaluate_range(args: argparse.Namespace) -> Path:
    """Evaluate or resume one explicit source range into a unique raw partial."""
    from titans_generate import TitansGenerator

    output = partial_path(args.output_dir, args.benchmark, args.start, args.end)
    metadata = _metadata(args)
    metadata_hash = hashlib.sha256(_canonical_json(metadata).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)

    completed: set[int] = set()
    if output.exists():
        with output.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if row.get("metadata_hash") != metadata_hash:
                    raise ValueError(f"{output}:{line_number}: metadata differs from this invocation")
                completed.add(int(row["index"]))

    generator = TitansGenerator(args.checkpoint, args.tokenizer, args.device)
    generator.enable_ttt_updates = not args.disable_ttt_updates
    with output.open("a", encoding="utf-8") as stream:
        if args.input_bin is not None:
            if args.max_tokens is None:
                raise ValueError("--input-bin requires --max-tokens")
            for index, input_ids in iter_token_windows(
                args.input_bin,
                sequence_length=args.max_tokens,
                start=args.start,
                end=args.end,
            ):
                if index in completed:
                    continue
                record = evaluate_token_window(generator, index, input_ids)
                stream.write(_canonical_json({"metadata": metadata, "metadata_hash": metadata_hash, **record}) + "\n")
                stream.flush()
        else:
            for index, text in iter_source(
                args.benchmark,
                args.input_file,
                start=args.start,
                end=args.end,
                text_field=args.text_field,
            ):
                if index in completed:
                    continue
                record = evaluate_text(generator, args.benchmark, index, text, args.max_tokens)
                stream.write(_canonical_json({"metadata": metadata, "metadata_hash": metadata_hash, **record}) + "\n")
                stream.flush()
    return output


def merge_partials(output_dir: Path, benchmark: str) -> tuple[Path, Path]:
    """Deterministically merge raw partials, rejecting overlaps or mixed metadata."""
    paths = sorted((output_dir / "raw").glob(f"{benchmark}.*-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No raw partials found for {benchmark} in {output_dir / 'raw'}")

    records: dict[int, dict[str, Any]] = {}
    compatible_metadata: dict[str, Any] | None = None
    files = []
    ignored_metadata_keys = {"range"}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row_count = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_count += 1
                index = int(row["index"])
                if index in records:
                    raise ValueError(f"Duplicate source index {index} in {path}")
                current = {key: value for key, value in row["metadata"].items() if key not in ignored_metadata_keys}
                if compatible_metadata is None:
                    compatible_metadata = current
                elif current != compatible_metadata:
                    raise ValueError(f"Incompatible metadata in {path}:{line_number}")
                records[index] = {key: value for key, value in row.items() if key not in {"metadata", "metadata_hash"}}
        files.append({"path": str(path.relative_to(output_dir)), "sha256": digest, "rows": row_count})

    ordered = [records[index] for index in sorted(records)]
    summary = {
        "schema_version": 1,
        "benchmark": benchmark,
        "metadata": compatible_metadata,
        "metrics": metric_totals(ordered),
        "source_indices": sorted(records),
    }
    summary_path = output_dir / f"{benchmark}.summary.json"
    manifest_path = output_dir / f"{benchmark}.manifest.json"
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "benchmark": benchmark,
            "partials": files,
            "summary": {
                "path": summary_path.name,
                "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            },
        },
    )
    return summary_path, manifest_path


def parse_args() -> argparse.Namespace:
    """Parse evaluate or deterministic-merge command-line arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--benchmark", choices=sorted(DATASETS), required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    inputs = evaluate.add_mutually_exclusive_group()
    inputs.add_argument("--input-file", type=Path)
    inputs.add_argument("--input-bin", type=Path)
    evaluate.add_argument("--text-field", default="text")
    evaluate.add_argument("--start", type=int, default=0)
    evaluate.add_argument("--end", type=int)
    evaluate.add_argument("--max-tokens", type=int)
    evaluate.add_argument("--disable-ttt-updates", action="store_true")
    evaluate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    merge = subparsers.add_parser("merge")
    merge.add_argument("--benchmark", choices=sorted(DATASETS), required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one evaluation range or merge its raw partials."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.command == "evaluate":
        if args.start < 0 or (args.end is not None and args.end <= args.start):
            raise ValueError("Require 0 <= start < end")
        if args.max_tokens is not None and args.max_tokens < 2:
            raise ValueError("--max-tokens must be at least 2")
        if args.input_bin is not None and args.max_tokens is None:
            raise ValueError("--input-bin requires --max-tokens as the evaluation sequence length")
        LOGGER.info("Raw partial: %s", evaluate_range(args))
    else:
        summary, manifest = merge_partials(args.output_dir, args.benchmark)
        LOGGER.info("Summary: %s; manifest: %s", summary, manifest)


if __name__ == "__main__":
    main()
