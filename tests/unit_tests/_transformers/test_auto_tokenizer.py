# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
from unittest.mock import patch

from transformers.tokenization_utils_base import BatchEncoding

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer


class _StubHFTokenizer:
    def __init__(self, bos_id=101, eos_id=102):
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id

    def __call__(self, *args, **kwargs):
        return BatchEncoding(
            {
                "input_ids": [[5, 6]],
                "attention_mask": [[1, 1]],
                "assistant_masks": [[0, 1]],
            }
        )

    def encode(self, *args, **kwargs):
        return [5, 6]


class TestNeMoAutoTokenizerFromPretrained:
    def test_patched_adds_bos_eos(self):
        stub = _StubHFTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            out = tok.__call__(["x"], add_special_tokens=False)
            assert isinstance(out, BatchEncoding)
            assert out["input_ids"] == [[stub.bos_token_id, 5, 6, stub.eos_token_id]]
            assert out["attention_mask"] == [[1, 1, 1, 1]]
            assert out["assistant_masks"] == [[1, 0, 1, 1]]

            enc = tok.encode("x")
            assert enc == [stub.bos_token_id, 5, 6, stub.eos_token_id]

    def test_force_hf_passthrough(self):
        stub = _StubHFTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model", force_hf=True)
            # Should be the original stub and unmodified outputs
            out = tok.__call__(["x"])
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]
            assert tok.encode("x") == [5, 6]


