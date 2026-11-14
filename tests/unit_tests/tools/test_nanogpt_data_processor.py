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

import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.nanogpt_dataset import _load_bos_index, load_bin_shard
from tools.nanogpt_data_processor import BinaryDataWriter

BOS = 7


@pytest.mark.parametrize("vocab_size", [100, 70_000], ids=["uint16", "uint32"])
def test_bos_index_holds_token_positions_the_dataset_reads(tmp_path, vocab_size):
    shard = tmp_path / "dataset.bin"
    chunks = [[BOS, 1, 2, BOS, 3], [BOS, 4, 5], [6, BOS]]
    writer = BinaryDataWriter(str(shard), bos_token_id=BOS, vocab_size=vocab_size)
    for chunk in chunks:
        writer.write(chunk)
    del writer  # flushes the header token count and closes both files

    bos_positions = _load_bos_index(shard)

    assert load_bin_shard(shard).tolist() == [tok for chunk in chunks for tok in chunk]
    assert bos_positions.dtype == np.int32
    assert bos_positions.tolist() == [0, 3, 5, 9]


def test_bos_positions_past_2gib_of_tokens_fit_the_int32_index(tmp_path):
    shard = tmp_path / "dataset.bin"
    writer = BinaryDataWriter(str(shard), bos_token_id=BOS, vocab_size=70_000)
    writer.write([1])
    # Pretend the shard already holds 2**31 - 3 uint32 tokens (8 GiB) without writing them.
    writer.bytes_written = (2**31 - 3) * writer.dtype.itemsize
    writer.write([BOS, 1])
    del writer

    assert np.fromfile(shard.with_suffix(".bos.idx"), dtype=np.int32).tolist() == [2**31 - 3]


def test_write_rejects_shards_past_the_int32_token_range(tmp_path):
    writer = BinaryDataWriter(str(tmp_path / "dataset.bin"), bos_token_id=BOS, vocab_size=100)
    writer.write([1])
    writer.bytes_written = (2**31 - 1) * writer.dtype.itemsize

    with pytest.raises(ValueError, match="int32 range"):
        writer.write([BOS, 1])
    del writer
