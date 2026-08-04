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
# ========================================================================
# test_gaussian_packer.py
# ========================================================================
import numpy as np
import pytest

# ------------------------------------------------------------------------
# System-under-test
# ------------------------------------------------------------------------
from nemo_automodel.components.datasets.llm.mock_packed import (
    build_packed_dataset,  # main API
    gen_sentence_ids,  # helper (for distribution test)
    make_vocab,  # helper
)


# ------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------
def _check_block(example, eos_id: int):
    """Verify internal consistency of one packed block."""
    L = len(example["input_ids"])

    # 1) All sequence-like columns have same length (=block_size)
    for k in ("attention_mask", "labels", "position_ids"):
        assert len(example[k]) == L

    # 2) attention_mask is all ones, labels == input_ids
    assert all(x == 1 for x in example["attention_mask"])
    assert example["labels"] == example["input_ids"]

    # 3) position_ids reset to 0 after every <eos>
    pos = 0
    for tid, pid in zip(example["input_ids"], example["position_ids"]):
        assert pid == pos
        pos = 0 if tid == eos_id else pos + 1


# ------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 123])
def test_basic_packing(seed):
    """Dataset has correct size and every block passes consistency check."""
    NUM_BLOCKS = 4
    BLOCK_SIZE = 32
    VOCAB_SIZE = 100

    ds = build_packed_dataset(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        mean_len=10,
        std_len=3,
        seed=seed,
        vocab_size=VOCAB_SIZE,
    )

    vocab = make_vocab(VOCAB_SIZE)

    # correct number of rows
    assert len(ds) == NUM_BLOCKS

    eos_id = vocab["<eos>"]
    for ex in ds:
        # each block is exactly BLOCK_SIZE long
        assert len(ex["input_ids"]) == BLOCK_SIZE
        _check_block(ex, eos_id)


def test_gaussian_length_distribution():
    """Sentence-length sampler should roughly match requested μ, σ."""
    MEAN, STD = 20, 5
    TRUNC = 64
    N = 10_000

    vocab = make_vocab(50)
    # collect raw sentence lengths (exclude the trailing <eos>)
    lengths = [len(gen_sentence_ids(vocab, MEAN, STD, TRUNC)) - 1 for _ in range(N)]

    mu_emp, sigma_emp = np.mean(lengths), np.std(lengths)

    # empirical mean / std should be close (within ≈⅓ σ)
    assert abs(mu_emp - MEAN) < STD / 3
    assert abs(sigma_emp - STD) < STD / 3
