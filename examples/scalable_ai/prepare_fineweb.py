#!/usr/bin/env python3
"""Download FineWeb-edu parquet files and tokenise them into NeMo Automodel ``NanogptDataset`` shards.

Shard format (nemo_automodel/components/datasets/llm/nanogpt_dataset.py, "new" format): int32[256] header
[MAGIC=278895051, VERSION=1, num_tokens, bytes_per_token], then uint32 tokens (the Moonshot tokenizer has 163,840
ids, so uint16 is not enough). A sibling ``.bos.idx`` file holds int32 positions of BOS tokens. Every document is
prefixed with BOS (Moonshot: 163584) and truncated to ``--max-doc-tokens``; no EOS is added, matching Automodel's
tools/nanogpt_data_processor.py. The validation shard is written first, then the training shards.

Example (one 8x H100 pre-training run of Moonlight-V4-16B-A3B, ~560M train tokens, ~2.3 GB on disk):
  python examples/scalable_ai/prepare_fineweb.py --tokenizer /workspace/models/Moonlight-V4-16B-A3B \
      --out /workspace/data/fineweb_edu_moonshot --num-files 2 --train-tokens 560M --val-tokens 8M --workers 32
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

MAGIC, VERSION, HEADER_SIZE = 278895051, 1, 256
_tok = None
_bos = None
_max_doc = None


def parse_tokens(s: str) -> int:
    """Parse token counts such as "560M" or "8K"."""
    s = s.strip().upper()
    mult = {"K": 10**3, "M": 10**6, "B": 10**9}.get(s[-1], 1)
    return int(float(s[:-1]) * mult) if s[-1] in "KMB" else int(s)


class ShardWriter:
    """Writes one NanogptDataset shard (new format, uint32 tokens) plus its BOS index."""

    def __init__(self, path: str, bos_id: int):
        self.path, self.bos_id, self.n = path, bos_id, 0
        self.fp = open(path, "wb")
        self.idx = open(path[:-4] + ".bos.idx", "wb")
        self.fp.write(np.zeros(HEADER_SIZE, dtype=np.int32).tobytes())

    def write(self, toks: np.ndarray):
        pos = self.n
        self.fp.write(toks.astype(np.uint32).tobytes())
        self.idx.write((pos + np.flatnonzero(toks == self.bos_id)).astype(np.int32).tobytes())
        self.n += toks.size

    def close(self):
        header = np.array([MAGIC, VERSION, self.n, 4] + [0] * (HEADER_SIZE - 4), dtype=np.int32)
        self.fp.seek(0)
        self.fp.write(header.tobytes())
        self.fp.close()
        self.idx.close()


def _init_worker(tokenizer_path: str, max_doc_tokens: int):
    global _tok, _bos, _max_doc
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _bos = _tok.bos_token_id
    _max_doc = max_doc_tokens


def _encode_batch(texts):
    """Tokenise a list of documents into one BOS-prefixed uint32 array (worker process)."""
    ids = _tok(texts, add_special_tokens=False)["input_ids"]
    return np.concatenate([np.array([_bos] + d[: _max_doc - 1], dtype=np.uint32) for d in ids])


def doc_batches(files, text_col="text", batch_rows=512):
    """Yield lists of document strings from parquet files."""
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[text_col]):
            yield batch.column(text_col).to_pylist()


def download(repo: str, subdir: str, num_files: int, dest: str) -> list[str]:
    """Download the first ``num_files`` parquet files under ``subdir`` of a Hub dataset repo."""
    from huggingface_hub import HfApi, hf_hub_download

    names = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith(subdir + "/") and f.endswith(".parquet")
    )
    assert names, f"no parquet files under {subdir} in {repo}"
    paths = []
    for name in names[:num_files]:
        t0 = time.time()
        p = hf_hub_download(repo, name, repo_type="dataset", local_dir=dest)
        print(f"downloaded {name} ({os.path.getsize(p) / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)
        paths.append(p)
    return paths


def main():
    """Download, tokenise and write shards."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subdir", default="sample/10BT")
    ap.add_argument("--num-files", default=2, type=int, help="parquet files to download (each ~2 GB, ~500M tokens)")
    ap.add_argument("--parquet", default=None, help="glob of already downloaded parquet files (skips the download)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", default="560M", type=parse_tokens)
    ap.add_argument("--val-tokens", default="8M", type=parse_tokens)
    ap.add_argument("--shard-tokens", default="100M", type=parse_tokens)
    ap.add_argument("--max-doc-tokens", default=32768, type=int)
    ap.add_argument("--workers", default=32, type=int)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    if args.parquet:
        files = sorted(glob.glob(os.path.expanduser(args.parquet)))
    else:
        files = download(args.repo, args.subdir, args.num_files, os.path.join(out, "raw"))
    assert files, "no parquet files"

    from transformers import AutoTokenizer

    bos = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True).bos_token_id
    assert bos is not None

    plan = [("val", args.val_tokens, args.val_tokens), ("train", args.train_tokens, args.shard_tokens)]
    t0, total, n_docs = time.time(), 0, 0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer, args.max_doc_tokens)) as pool:
        stream = pool.imap(_encode_batch, doc_batches(files), chunksize=1)
        for split, budget, shard_size in plan:
            written, shard_id, writer = 0, 0, None
            while written < budget:
                if writer is None:
                    writer = ShardWriter(os.path.join(out, f"fineweb_{split}_{shard_id:04d}.bin"), bos)
                try:
                    buf = next(stream)
                except StopIteration:
                    print(f"[{split}] ran out of documents after {written / 1e6:.1f}M tokens", flush=True)
                    break
                writer.write(buf)
                written += buf.size
                total += buf.size
                n_docs += int((buf == bos).sum())
                if writer.n >= shard_size:
                    writer.close()
                    writer = None
                    shard_id += 1
                if total // 50_000_000 != (total - buf.size) // 50_000_000:
                    print(
                        f"[{split}] {written / 1e6:8.1f}M / {budget / 1e6:.0f}M tokens, {n_docs} docs, {total / (time.time() - t0) / 1e6:.2f} Mtok/s",
                        flush=True,
                    )
            if writer is not None:
                writer.close()
            print(f"[{split}] done: {written / 1e6:.1f}M tokens", flush=True)
        pool.terminate()
    print(f"total {total / 1e6:.1f}M tokens, {n_docs} docs, {time.time() - t0:.0f}s -> {out}")
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(
            f"tokenizer={args.tokenizer} bos={bos} train_tokens={args.train_tokens} val_tokens={args.val_tokens} "
            f"source={args.repo}/{args.subdir} files={[os.path.basename(p) for p in files]}\n"
        )


if __name__ == "__main__":
    main()
