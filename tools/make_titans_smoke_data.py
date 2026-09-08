#!/usr/bin/env python3
"""Create deterministic NanoGPT shards for the Titans training smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nanogpt_data_processor import BinaryDataWriter


def _write_split(path: Path, *, documents: int, seq_len: int, vocab_size: int, bos_token_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = BinaryDataWriter(str(path), bos_token_id=bos_token_id, vocab_size=vocab_size)
    try:
        for document in range(documents):
            payload = 3 + (np.arange(seq_len, dtype=np.int64) + document * 17) % (vocab_size - 3)
            writer.write(np.concatenate(([bos_token_id], payload)))
    finally:
        writer.close()


def main() -> None:
    """Parse arguments and write training and validation smoke shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-documents", type=int, default=8)
    parser.add_argument("--validation-documents", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--bos-token-id", type=int, default=1)
    args = parser.parse_args()

    if args.vocab_size <= 3:
        parser.error("--vocab-size must be greater than 3")

    train_path = args.output_dir / "train" / "fineweb_edu_smoke.bin"
    validation_path = args.output_dir / "validation" / "fineweb_edu_smoke.bin"
    _write_split(
        train_path,
        documents=args.train_documents,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        bos_token_id=args.bos_token_id,
    )
    _write_split(
        validation_path,
        documents=args.validation_documents,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        bos_token_id=args.bos_token_id,
    )

    manifest = {
        "format": "nanogpt",
        "synthetic": True,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "bos_token_id": args.bos_token_id,
        "train_documents": args.train_documents,
        "validation_documents": args.validation_documents,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote deterministic Titans smoke data to {args.output_dir}")


if __name__ == "__main__":
    main()
