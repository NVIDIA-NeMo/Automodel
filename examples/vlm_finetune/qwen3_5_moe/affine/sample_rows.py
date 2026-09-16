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

"""Draw a uniform random row subset of a pre-filtered train split.

Rows are sampled without replacement and written in their original order, with every
column (including ``n_tokens`` for the length-grouped sampler) kept. Run from the repo root:

    python examples/vlm_finetune/qwen3_5_moe/affine/sample_rows.py [-n 10000] [--seed 1234]
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    """Sample the requested number of rows, write them to the destination and print their token stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/v5_130k_filtered/train.parquet")
    parser.add_argument("--dst", default="data/v5_10k/train.parquet")
    parser.add_argument("-n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    table = pq.read_table(args.src)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(table.num_rows, size=args.n, replace=False))
    subset = table.take(indices)

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    pq.write_table(subset, args.dst)

    tokens = subset.column("n_tokens").to_numpy()
    print(
        f"{args.dst}: {subset.num_rows} of {table.num_rows} rows (seed {args.seed}), "
        f"{tokens.sum() / 1e6:.1f}M tokens, mean {tokens.mean():.0f}, "
        f"p50 {np.median(tokens):.0f}, max {tokens.max()}"
    )


if __name__ == "__main__":
    main()
