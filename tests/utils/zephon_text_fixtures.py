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

"""Local tokenizer and finite source fixtures for the Zephon adapter."""

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.datasets.zephon import ZephonDataloaderConfig
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonTextSourceConfig,
)


@pytest.fixture
def tokenizer(tmp_path):
    raw = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, **{f"w{i}": i + 2 for i in range(40)}}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(path)
    return PreTrainedTokenizerFast.from_pretrained(path)


@pytest.fixture
def config(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    path = directory / "text.jsonl"
    path.write_text(
        "".join(json.dumps({"text": f"w{i % 30} w{(i + 1) % 30} w{(i + 2) % 30}"}) + "\n" for i in range(36))
    )
    return ZephonDataloaderConfig(
        dataset_config=ZephonTextSourceConfig(tokenize_special_tokens="eos"),
        zephon=ZephonOptions(
            sources=(ZephonSource(name="text", path=str(directory), fmt="jsonl"),),
            packing=ZephonPackingOptions(seq_len=7, shuffle_after_pack=False),
            source=ZephonSourceOptions(
                chunk_size=4, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            runtime=ZephonRuntimeOptions(runner="threads", mtp_mode=False),
        ),
        batch_size=2,
    )
