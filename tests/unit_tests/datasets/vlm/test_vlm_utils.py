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

import pytest
import torch

from nemo_automodel.components.datasets.vlm import utils


def test_json2token_basic():
    obj = {"a": 1, "b": 2}
    token = utils.json2token(obj)
    assert token == "{\n  \"a\": 1,\n  \"b\": 2\n}"


def test_json2token_sorted_keys():
    obj = {"b": 2, "a": 1}
    token = utils.json2token(obj, sort_json_key=True)
    assert token == "{\n  \"a\": 1,\n  \"b\": 2\n}"


def test_json2token_unsorted_keys():
    obj = {"b": 2, "a": 1}
    token = utils.json2token(obj, sort_json_key=False)
    assert token in (
        "{\n  \"b\": 2,\n  \"a\": 1\n}",
        "{\n  \"a\": 1,\n  \"b\": 2\n}",
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (123, "123"),
        ([1, 2], "[\n  1,\n  2\n]"),
    ],
)
def test_json2token_primitives(value, expected):
    assert utils.json2token(value) == expected
