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

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.nanogpt_data_processor import _get_tokenizer, tokenize_chunk


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
