# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# tests/test_utils.py
import json
import types

import pytest
import torch

from nemo_automodel.components.datasets.vlm import utils


def test_default_stop_tokens_with_tokenizer():
    tokenizer = types.SimpleNamespace(eos_token="<eos>")
    processor = types.SimpleNamespace(tokenizer=tokenizer)

    tokens = utils.default_stop_tokens(processor)

    assert tokens[-1] == "<eos>"
    assert tokens[:-1] == ("<end_of_turn>", "<|im_end|>", "<|eot_id|>")


def test_default_stop_tokens_without_tokenizer():
    processor = object()

    tokens = utils.default_stop_tokens(processor)

    assert tokens == ("<end_of_turn>", "<|im_end|>", "<|eot_id|>")


def test_json2token_basic():
    obj = {"a": 1, "b": 2}
    token = utils.json2token(obj)
    # Keys are iterated in reverse-sorted order by default
    assert token == "<s_b>2</s_b><s_a>1</s_a>"


def test_json2token_sorted_keys():
    obj = {"b": 2, "a": 1}
    token = utils.json2token(obj, sort_json_key=True)
    assert token == "<s_b>2</s_b><s_a>1</s_a>"


def test_json2token_unsorted_keys():
    obj = {"b": 2, "a": 1}
    token = utils.json2token(obj, sort_json_key=False)
    assert token == "<s_b>2</s_b><s_a>1</s_a>"


def test_json2token_primitives():
    assert utils.json2token(None) == "None"
    assert utils.json2token(True) == "True"
    assert utils.json2token(False) == "False"
    assert utils.json2token(123) == "123"
    assert utils.json2token([1, 2]) == "1<sep/>2"
