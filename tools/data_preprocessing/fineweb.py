"""FineWeb dataset preprocessing script (ported from *modded-NanoGPT*).

This tool downloads the FineWeb dataset from Hugging Face, tokenizes with
`tiktoken` (GPT-2 BPE), and writes memory-mapped binary shards compatible
with `BinTokenDataset` for efficient streaming pre-training.

Usage (typical):

```bash
python tools/data_preprocessing/fineweb.py --split sample-10BT -m 500M
```

See the bottom of the file (`__main__`) for CLI options.
"""
# NOTE: the body below is copied verbatim (minus this header) from
#       modded-nanogpt/data/fineweb.py so that users do not have to clone the
#       NanoGPT repo to generate binary shards.

import os
import argparse
import multiprocessing as mp

try:
    # Use 'spawn' start method instead of default 'fork' to avoid PyGILState_Release crashes
    # that occur when forking processes with C extensions on macOS/Python 3.12.
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method may have been set already (e.g., inside a child process). Ignore in that case.
    pass

import numpy as np
import tiktoken
from datasets import load_dataset
from functools import partial
from tqdm import tqdm

# ------------------------------------------
# helpers


def write_datafile(filename, toks, *, bos_token: int = 50256):
    """Save *toks* to *filename* and also emit a BOS index file.

    The shard layout remains compatible with previous versions:
    1. 256 × ``int32`` header (magic, version, *num_tokens*, ...)
    2. ``uint16[num_tokens]`` tokens

    In addition, a *BOS index* is written to the same directory.  For a shard
    called ``xyz.bin`` the companion index is named ``xyz.bos.idx``.
    """

    assert len(toks) < 2 ** 31, "token count too large"  # ~2.1B tokens

    # --------------------------------- write .bin shard
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(toks)  # number of tokens in *toks*

    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        # Guard against token ids that exceed uint16 range
        maxtok = 2 ** 16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.asarray(toks, dtype=np.uint16)
    else:
        toks_np = toks

    print(f"writing {len(toks_np):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

    # --------------------------------- write BOS index
    bos_positions = np.where(toks_np == bos_token)[0].astype(np.int32)
    idx_filename = filename.replace(".bin", ".bos.idx")
    print(f"writing {len(bos_positions):,} BOS positions to {idx_filename}")
    with open(idx_filename, "wb") as idx_f:
        np.asarray([len(bos_positions)], dtype=np.int32).tofile(idx_f)
        bos_positions.tofile(idx_f)


class _parse_tokens_arg(int):
    """An int subclass that can parse human-friendly token counts (e.g. 500M)."""

    _UNIT_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    def __new__(cls, value):
        if isinstance(value, int):
            return super().__new__(cls, value)
        if value is None:
            return super().__new__(cls, 0)
        if isinstance(value, str):
            val = value.strip()
            if val.isdigit():
                return super().__new__(cls, int(val))
            import re

            m = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)\s*([KMB])", val)
            if m:
                num = float(m.group(1))
                unit = m.group(2).upper()
                return super().__new__(cls, int(num * cls._UNIT_MULTIPLIER[unit]))
        raise argparse.ArgumentTypeError(
            f"Could not parse token count '{value}'. Expected integer or number followed by K/M/B."
        )

    def __repr__(self):
        value = int(self)
        if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
            return f"{value // 1_000_000_000}B"
        if value >= 1_000_000 and value % 1_000_000 == 0:
            return f"{value // 1_000_000}M"
        if value >= 1_000 and value % 1_000 == 0:
            return f"{value // 1_000}K"
        return str(value)


# ------------------------------------------
# tokenization helpers


def tokenize_doc(doc, eot, enc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def process_tokens(tokens, state, args, data_cache_dir):
    shard_index = state["shard_index"]
    token_count = state["token_count"]
    tokens_written = state["tokens_written"]
    all_tokens_np = state["all_tokens_np"]
    progress_bar = state["progress_bar"]

    if args.max_tokens is not None:
        remaining = args.max_tokens - tokens_written
        if remaining <= 0:
            return False
        if len(tokens) > remaining:
            tokens = tokens[:remaining]

    if token_count + len(tokens) < args.shard_size:
        all_tokens_np[token_count : token_count + len(tokens)] = tokens
        token_count += len(tokens)
        if progress_bar is None:
            progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
    else:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"fineweb_{split}_{shard_index:06d}.bin")
        remainder = args.shard_size - token_count
        if progress_bar is None:
            progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(remainder)
        all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        progress_bar = None
        all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
        token_count = len(tokens) - remainder

    tokens_written += len(tokens)

    state.update(
        {
            "shard_index": shard_index,
            "token_count": token_count,
            "tokens_written": tokens_written,
            "progress_bar": progress_bar,
        }
    )

    if args.max_tokens is not None and tokens_written >= args.max_tokens:
        return False
    return True


def make_parser():
    parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
    parser.add_argument(
        "--split", choices=["sample-10BT", "sample-100BT"], help="Split of fineweb to use"
    )
    parser.add_argument("-s", "--shard_size", type=int, default=10 ** 8, help="Size of each shard in tokens")
    parser.add_argument(
        "-m",
        "--max_tokens",
        "--max-tokens",
        type=_parse_tokens_arg,
        default=None,
        help=(
            "If set, stop after processing this many tokens. "
            "You can use K/M/B suffixes, e.g. 500M for 500 million tokens."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() - 2),
        help=(
            "Number of workers to use for processing the dataset. If not set, will use all available cores minus 2."
        ),
    )
    return parser


def main(args):
    print(args)
    local_dir = args.split.replace("sample-", "fineweb_")
    if args.max_tokens:
        local_dir += f"_max_{str(args.max_tokens)}"
    print("Writing to: ", local_dir)
    data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    fw = load_dataset("HuggingFaceFW/fineweb", name=args.split, split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    tokenize = partial(tokenize_doc, eot=eot, enc=enc)

    state = {
        "shard_index": 0,
        "token_count": 0,
        "tokens_written": 0,
        "all_tokens_np": np.empty((args.shard_size,), dtype=np.uint16),
        "progress_bar": None,
    }

    with mp.Pool(args.num_workers) as pool:
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if not process_tokens(tokens, state, args, data_cache_dir):
                break

    if state["token_count"] != 0:
        split = "val" if state["shard_index"] == 0 else "train"
        filename = os.path.join(data_cache_dir, f"fineweb_{split}_{state['shard_index']:06d}.bin")
        write_datafile(filename, state["all_tokens_np"][: state["token_count"]])

    if state["progress_bar"] is not None:
        state["progress_bar"].close()


if __name__ == "__main__":
    main(make_parser().parse_args()) 