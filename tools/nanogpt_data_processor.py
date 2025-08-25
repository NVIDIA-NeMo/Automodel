# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
FineWeb dataset preprocessing script

This tool downloads a dataset from the Hugging Face Hub (default: FineWeb),
tokenizes the data (default: tiktoken + gpt2), and writes memory-mapped binary shards compatible
with `BinTokenDataset` for efficient streaming pre-training.

Usage (typical):

```bash
python tools/nanogpt_data_processor.py \
    --dataset HuggingFaceFW/fineweb \
    --set-name sample-10BT \
    --max-tokens 500M
```

See the make_parser function for CLI options or run `python tools/nanogpt_data_processor.py --help`.
"""

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import queue  # for Empty exception handling
from functools import lru_cache

import numpy as np
from transformers import PreTrainedTokenizer

try:
    from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
        HEADER_SIZE,
        MAGIC,
        VERSION,
    )
except ImportError:
    logging.warning("nemo_automodel not installed, using local constants; this is not recommended;")
    logging.warning("Please install nemo_automodel or modify the PYTHONPATH to include the nemo_automodel directory")
    HEADER_SIZE = 256
    MAGIC = 2788_95051
    VERSION = 1


class _parse_tokens_arg(int):
    """An int subclass that can parse human-friendly token counts (e.g. 500M)."""

    _UNIT_MULTIPLIER = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    def __new__(cls, value):
        """
        Parse a human-friendly token count (e.g. 500M) into an integer.

        Args:
            value: str | int
                The token count to parse.

        Returns:
            int: The parsed token count.
        """
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
        """
        Return a human-readable string representation of the token count.

        Returns:
            str: The human-readable string representation of the token count.
        """
        value = int(self)
        if value >= 1_000_000_000 and value % 1_000_000_000 == 0:
            return f"{value // 1_000_000_000}B"
        if value >= 1_000_000 and value % 1_000_000 == 0:
            return f"{value // 1_000_000}M"
        if value >= 1_000 and value % 1_000 == 0:
            return f"{value // 1_000}K"
        return str(value)


def make_parser():
    """
    Create an argument parser for the data preprocessing script.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Dataset preprocessing script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="Dataset to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to write the dataset to. If not set, will use the dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to use",
    )
    parser.add_argument(
        "--set-name",
        default="sample-10BT",
        help="Split of fineweb to use; default: sample-10BT for fineweb.",
    )
    parser.add_argument(
        "-m",
        "--max_tokens",
        "--max-tokens",
        type=_parse_tokens_arg,
        default=2**32,
        help=(
            "If set, stop after processing this many tokens. "
            "You can use K/M/B suffixes, e.g. 500M for 500 million tokens."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() - 2),
        help=("Number of workers to use for processing the dataset. If not set, will use all available cores minus 2."),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use for tokenization; model-id on HF hub.",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of sequences to use in each tokenizer worker",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=128,
        help="Number of chunks to prefetch from the dataset",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="Maximum length of the tokens to encode. If the text's token length exceeds this, it will be truncated.",
    )
    return parser


class BinaryDataWriter:
    def __init__(self, filename, bos_token_id, vocab_size):
        """
        Initialize the binary data writer.

        The binary file will be written to ``filename.bin`` and the index file
        will be written to ``filename.bos.idx``.

        Args:
            filename: str
                Name of the binary file to write to.
            dtype: np.dtype
                Data type of the tokens.
            vocab_size: int
                Size of the vocabulary.
        """
        self.filename = filename
        self.bos_token_id = bos_token_id
        # allow both instance and type
        if vocab_size < 2**16:
            dtype = np.uint16()
            print(f"Using uint16 for vocab size {vocab_size}")
        elif vocab_size < 2**32:
            dtype = np.uint32()
            print(f"Using uint32 for vocab size {vocab_size}")
        else:
            raise ValueError(f"Vocab size {vocab_size} is too large for uint32")

        self.dtype = dtype
        # header
        self.header = np.zeros(HEADER_SIZE, dtype=np.int32)
        self.header[0] = MAGIC
        self.header[1] = VERSION
        self.header[2] = 0  # number of tokens in *toks*
        self.header[3] = dtype.itemsize  # bytes per token

        self.bin_fp = None
        self.idx_fp = None
        self.bytes_written = 0

    def _write_header(self):
        """
        Write the header to the binary and index files.
        """
        bin_fp = open(self.filename, "wb")
        bin_fp.write(self.header.tobytes())
        idx_fp = open(self.filename.replace(".bin", ".bos.idx"), "wb")
        return bin_fp, idx_fp

    def write(self, tokens: np.ndarray | list):
        """
        Write tokens to the binary and index files.

        Args:
            tokens: np.ndarray | list
                Tokens to write to the binary and index files.

        Returns:
            int: Number of tokens written.
        """
        if self.bin_fp is None:
            self.bin_fp, self.idx_fp = self._write_header()

        if isinstance(tokens, list):
            tokens = np.array(tokens)
            assert (0 <= tokens).all() and (tokens < 2 ** (self.dtype.itemsize * 8)).all(), (
                "token dictionary too large for uint16"
            )
            tokens = tokens.astype(self.dtype)

        pos = self.bin_fp.tell()
        assert pos + tokens.size * self.dtype.itemsize < 2**32 - 1, "token count too large"
        # write chunk tokens
        tok_bytes = tokens.tobytes()
        self.bin_fp.write(tok_bytes)

        # write BOS index
        self.idx_fp.write((pos + np.where(tokens == self.bos_token_id)[0].astype(np.int32)).tobytes())
        self.bytes_written += len(tok_bytes)
        return len(tok_bytes)

    def __del__(self):
        """
        Close the binary and index files.

        Writes the number of tokens written to the header.
        """
        if self.bin_fp is not None:
            # Write the number of tokens written to the header
            self.bin_fp.seek(2 * 4, 0)
            b = np.zeros(1, dtype=np.int32)
            b[0] = self.items_written
            self.bin_fp.write(b.tobytes())
            self.bin_fp.close()
        if self.idx_fp is not None:
            self.idx_fp.close()

    @property
    def items_written(self):
        """
        Return the number of tokens written (bytes / dtype.itemsize)
        """
        return self.bytes_written // self.dtype.itemsize


def dataset_reader(
    dataset_name: str,
    set_name: str,
    split: str,
    data_cache_dir: str | None,
    out_queue: "mp.Queue",
    chunk_size: int,
):
    """
    Stream the dataset and push samples to out_queue.

    Args:
        dataset_name: str
            The dataset identifier on the HuggingFace hub (e.g. ``"HuggingFaceFW/fineweb"``).
        set_name: str
            Name of the subset / configuration to use (e.g. ``"sample-10BT"``).
        split: str
            Which split to stream (e.g. ``"train"``).
        data_cache_dir: str | None
            Directory where the dataset will be cached locally. If ``None`` the
            default HF cache location is used.
        out_queue: mp.Queue
            Queue used to hand over raw records to the main process. A ``None``
            sentinel is pushed after the stream ends to signal completion.
        chunk_size: int
            Number of samples to read from the dataset at a time.
    """

    from datasets import load_dataset

    # HF dataset streaming loader
    dataset_iter = load_dataset(
        dataset_name,
        name=set_name,
        split=split,
        streaming=True,
        cache_dir=data_cache_dir,
    )

    try:
        chunk = []
        for sample in dataset_iter:
            chunk.append(sample)
            if len(chunk) == chunk_size:
                out_queue.put(chunk)
                chunk = []
        if chunk:
            out_queue.put(chunk)
    finally:
        # Always push sentinel so consumer can terminate.
        out_queue.put(None)


@lru_cache(maxsize=None)
def _get_tokenizer(tokenizer_name: str) -> tuple[PreTrainedTokenizer, int]:
    """
    Build the tokenizer once per Python process.

    Args:
        tokenizer_name: str
            The name of the tokenizer to use.

    Returns:
        tuple[PreTrainedTokenizer, int]
            A tuple containing the tokenizer and the end-of-text token ID.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos_token_id = tokenizer.bos_token_id
    return tokenizer, bos_token_id


def tokenize_chunk(chunk: list[dict], tokenizer_name: str, max_length: int) -> list[list[int]]:
    """
    Tokenize a chunk of text.

    Args:
        chunk: list[dict]
            A list of dictionaries, each containing a "text" key.
        tokenizer_name: str
            The name of the tokenizer to use.
        max_length: int
            The maximum length of the tokens to encode. If the text's token length exceeds this, it will be truncated.

    Returns:
        list[list[int]]
            A list of lists of token IDs.
    """
    tokenizer, bos_token_id = _get_tokenizer(tokenizer_name)  # first call builds, later calls reuse
    out = []
    for doc in chunk:
        tokens = tokenizer.encode(doc["text"], max_length=max_length, truncation=False)
        if tokens and tokens[0] != bos_token_id:
            tokens = [bos_token_id] + tokens
        out.append(tokens)
    return out


def main(args):
    """
    Main function to run the data preprocessing pipeline.

    This function:
    1. Prepares output/cache directories
    2. Creates a dataset-reader process to stream raw samples
    3. Creates a writer process to write binary data to disk
    4. Creates parallel tokenisation workers to tokenize chunks
    5. Writes tokens to disk until the token budget is exhausted or the dataset stream is exhausted.
    6. Cleans up resources

    Args:
        args: argparse.Namespace
            The parsed command line arguments.
    """
    print(args, json.dumps(vars(args), indent=4))

    # Prepare output/cache directories
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), args.dataset.split("/")[1])
    else:
        output_dir = args.output_dir
    if args.max_tokens:
        output_dir += f"_max_tokens_{str(args.max_tokens)}"

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Writing to:", output_dir)

    data_cache_dir = args.data_cache_dir or output_dir
    os.makedirs(data_cache_dir, exist_ok=True)

    # Dataset-reader process
    data_queue: mp.Queue = mp.Queue(maxsize=args.prefetch)
    reader_proc = mp.Process(
        target=dataset_reader,
        args=(
            args.dataset,
            args.set_name,
            args.split,
            data_cache_dir,
            data_queue,
            args.chunk_size,
        ),
        daemon=True,
    )
    reader_proc.start()

    # This process writes the binary data to disk.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    writer = BinaryDataWriter(
        os.path.join(output_dir, "dataset.bin"), bos_token_id=tokenizer.bos_token_id, vocab_size=tokenizer.vocab_size
    )
    del tokenizer

    # Parallel tokenisation workers
    futures: list[concurrent.futures.Future] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Keep looping until dataset stream is exhausted *and* all futures done.
        stream_finished = False
        while not stream_finished or futures:
            # Pull raw samples from the queue to build chunks
            while not stream_finished or len(futures) < args.num_workers * args.prefetch:
                try:
                    chunk = data_queue.get(block=False)
                except queue.Empty:  # No new sample available right now - proceed to consume futures.
                    break
                if chunk is None:  # Sentinel received - no more data coming.
                    stream_finished = True
                    break
                futures.append(executor.submit(tokenize_chunk, chunk, args.tokenizer, args.max_length))

            # Consume completed futures to write tokens to disk
            max_i = 0
            for i in range(len(futures)):
                if not futures[i].done():  # early exit if future is not done
                    break
                for tokens in futures[i].result():
                    writer.write(tokens)
                max_i = i + 1
            futures = futures[max_i:]

            # Stop early if token budget exhausted
            if writer.items_written >= args.max_tokens:
                for fut in futures:
                    fut.cancel()
                break

    del futures
    # Explicitly close queue to prevent leaked semaphores at interpreter shutdown.
    if reader_proc.is_alive():
        reader_proc.terminate()
        reader_proc.join()
    assert not reader_proc.is_alive()
    # close queue
    data_queue.close()
    data_queue.join_thread()


if __name__ == "__main__":
    main(make_parser().parse_args())
