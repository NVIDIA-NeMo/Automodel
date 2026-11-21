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

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.datasets.llm.nanogpt_dataset import _load_bos_index, load_bin_shard
from tools.nanogpt_data_processor import BinaryDataWriter, _get_tokenizer, tokenize_chunk

BOS = 7


@pytest.fixture(scope="module")
def local_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("nanogpt-tokenizer")
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]")
    tokenizer.save_pretrained(path)
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a b c d e f", [1, 2, 3, 4, 5]),
        ("a b", [1, 2, 3]),
        ("", []),
        ("a b c d", [1, 2, 3, 4, 5]),
        ("[BOS] a b c d e", [1, 2, 3, 4]),
        ("[BOS] a", [1, 2]),
    ],
)
def test_tokenize_chunk_max_length(local_tokenizer: Path, text: str, expected: list[int]) -> None:
    # max_length bounds the encoded body; the existing optional BOS is added afterwards.
    assert tokenize_chunk([{"text": text}], str(local_tokenizer), max_length=4) == [expected]


def test_tokenize_chunk_multiple_documents(local_tokenizer: Path) -> None:
    chunk = [{"text": "a b c d e f"}, {"text": ""}, {"text": "b a"}]
    assert tokenize_chunk(chunk, str(local_tokenizer), max_length=4) == [[1, 2, 3, 4, 5], [], [1, 3, 2]]


def test_get_tokenizer_reuses_local_artifact(local_tokenizer: Path) -> None:
    first, bos = _get_tokenizer(str(local_tokenizer))
    second, second_bos = _get_tokenizer(str(local_tokenizer))
    assert first is second
    assert bos == second_bos == 1
    assert first.encode("a b c d e f", truncation=True, max_length=4) == [2, 3, 4, 5]


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
