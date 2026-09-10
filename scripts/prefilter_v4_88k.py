# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Drop over-length rows from ``vuhaian/v4_88k`` and write train/validation Parquet.

Right-truncating an agentic trajectory removes its tail, which is exactly where the
supervised final assistant turn lives -- a truncated row contributes little or no loss
signal. This script drops such rows instead, offline, so the training config carries no
length cap and the collator pads only to the longest sample in each batch.

Run before training::

    export HF_TOKEN=...
    uv run --no-project --with "transformers>=5" --with datasets \\
        python scripts/prefilter_v4_88k.py --max-seq-len 40960 --out data/v4_88k_filtered
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    """Filter the corpus by rendered token length and write train/val Parquet files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="vuhaian/v4_88k", help="HF dataset id.")
    parser.add_argument("--split", default="train", help="HF split expression.")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", help="Tokenizer / chat template source.")
    parser.add_argument("--max-seq-len", type=int, default=40960, help="Rows longer than this are dropped.")
    parser.add_argument("--holdout", type=int, default=512, help="Rows reserved for validation.")
    parser.add_argument("--seed", type=int, default=1234, help="Shuffle seed applied before slicing.")
    parser.add_argument("--out", default="data/v4_88k_filtered", help="Output directory.")
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset)

    def _length(batch: dict) -> dict:
        return {
            "n_tokens": [
                len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"])
                for messages in batch["messages"]
            ]
        }

    dataset = dataset.map(_length, batched=True, batch_size=32, num_proc=args.num_proc)
    kept = dataset.filter(lambda n: n <= args.max_seq_len, input_columns="n_tokens", num_proc=args.num_proc)

    dropped = total - len(kept)
    print(f"rows in:      {total:,}")
    print(f"rows kept:    {len(kept):,} ({len(kept) / total:.2%}) at max_seq_len={args.max_seq_len:,}")
    print(f"rows dropped: {dropped:,} ({dropped / total:.2%})")

    # Shuffle before slicing: the source Parquet is ordered by source/repo/stratum, so a
    # head slice would draw the validation set from a handful of repositories.
    kept = kept.shuffle(seed=args.seed)
    holdout = min(args.holdout, len(kept) // 10)
    validation = kept.select(range(holdout))
    train = kept.select(range(holdout, len(kept)))

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.parquet")
    val_path = os.path.join(args.out, "val.parquet")
    train.to_parquet(train_path)
    validation.to_parquet(val_path)

    print(f"train: {len(train):,} rows -> {train_path}")
    print(f"val:   {len(validation):,} rows -> {val_path}")


if __name__ == "__main__":
    main()
