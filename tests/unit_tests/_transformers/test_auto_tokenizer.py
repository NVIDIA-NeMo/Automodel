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
from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import _add_token


class _StubHFTokenizer:
    def __init__(self, bos_id=101, eos_id=102):
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.add_bos_token = True
        self.add_eos_token = True

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


class _StubConfig:
    model_type = "stub"


class TestNeMoAutoTokenizerFromPretrained:
    def test_patched_adds_bos_eos(self):
        stub = _StubHFTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub), \
             patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            out = tok(["x"])
            assert isinstance(out, BatchEncoding)
            assert out["input_ids"] == [[stub.bos_token_id, 5, 6, stub.eos_token_id]]
            assert out["attention_mask"] == [[1, 1, 1, 1]]
            assert out["assistant_masks"] == [[1, 0, 1, 1]]

            out = tok(["x"], add_special_tokens=False)
            assert isinstance(out, BatchEncoding)
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]
            assert out["assistant_masks"] == [[0, 1]]

            enc = tok.encode("x")
            assert enc == [stub.bos_token_id, 5, 6, stub.eos_token_id]

            enc = tok.encode("x", add_special_tokens=False)
            assert enc == [5, 6]

    def test_cls_sep_pattern_fixed_when_tokens_missing(self):
        """Transformers >=5.0 defaults special_tokens_pattern to 'cls_sep'.
        When cls/sep token IDs are None (e.g. Moonlight TikTokenTokenizer),
        the wrapper should reset it to 'none' to avoid None in input_ids."""
        stub = _StubHFTokenizer()
        # Simulate a tokenizer that got the default "cls_sep" pattern but has no CLS/SEP
        stub.special_tokens_pattern = "cls_sep"
        stub.cls_token_id = None
        stub.sep_token_id = None
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub), \
             patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok.special_tokens_pattern == "none"

    def test_cls_sep_pattern_preserved_when_tokens_present(self):
        """When cls/sep token IDs are properly defined, the pattern should stay."""
        stub = _StubHFTokenizer()
        stub.special_tokens_pattern = "cls_sep"
        stub.cls_token_id = 200
        stub.sep_token_id = 201
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub), \
             patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok.special_tokens_pattern == "cls_sep"

    def test_force_hf_passthrough(self):
        stub = _StubHFTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model", force_hf=True)
            # Should be the original stub and unmodified outputs
            out = tok(["x"])
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]
            assert tok.encode("x") == [5, 6]

    def test_add_bos_token_falls_back_on_value_error(self):
        """When tokenizer.add_bos_token setter raises ValueError (e.g. transformers v5
        read-only property), fall back to setting _add_bos_token directly."""

        class _StrictBosTokenizer(_StubHFTokenizer):
            """Tokenizer whose add_bos_token property raises on set."""
            bos_token = "<s>"
            eos_token = "</s>"

            def __init__(self):
                # Skip parent __init__ which would trigger the strict setter
                self.bos_token_id = 101
                self.eos_token_id = 102
                self.add_eos_token = True

            @property
            def add_bos_token(self):
                return getattr(self, "_add_bos_token", False)

            @add_bos_token.setter
            def add_bos_token(self, value):
                raise ValueError("read-only in this tokenizer version")

        stub = _StrictBosTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub), \
             patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok._add_bos_token is True

    def test_add_eos_token_falls_back_on_value_error(self):
        """When tokenizer.add_eos_token setter raises ValueError,
        fall back to setting _add_eos_token directly."""

        class _StrictEosTokenizer(_StubHFTokenizer):
            """Tokenizer whose add_eos_token property raises on set."""
            bos_token = "<s>"
            eos_token = "</s>"

            def __init__(self):
                # Skip parent __init__ which would trigger the strict setter
                self.bos_token_id = 101
                self.eos_token_id = 102
                self.add_bos_token = True

            @property
            def add_eos_token(self):
                return getattr(self, "_add_eos_token", False)

            @add_eos_token.setter
            def add_eos_token(self, value):
                raise ValueError("read-only in this tokenizer version")

        stub = _StrictEosTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub), \
             patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok._add_eos_token is True


class TestAddTokenHelper:
    def test_input_ids_single_sequence_no_duplicates(self):
        enc = BatchEncoding({"input_ids": [5, 6]})
        # prepend bos
        _add_token(enc, 101, 0, "input_ids")
        out = enc
        assert out["input_ids"] == [101, 5, 6]
        # append eos
        _add_token(out, 102, -1, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]
        # no duplicate prepend
        _add_token(out, 101, 0, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]
        # no duplicate append
        _add_token(out, 102, -1, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]

    def test_masks_batched_always_extend(self):
        enc = BatchEncoding({"attention_mask": [[1, 1], [1]]})
        # always add on prepend
        _add_token(enc, 1, 0, "attention_mask")
        out = enc
        assert out["attention_mask"] == [[1, 1, 1], [1, 1]]
        # always add on append
        _add_token(out, 1, -1, "attention_mask")
        assert out["attention_mask"] == [[1, 1, 1, 1], [1, 1, 1]]

    def test_empty_sequences(self):
        # input_ids empty
        enc_ids = BatchEncoding({"input_ids": []})
        _add_token(enc_ids, 101, 0, "input_ids")
        out_ids = enc_ids
        assert out_ids["input_ids"] == [101]
        _add_token(out_ids, 102, -1, "input_ids")
        assert out_ids["input_ids"] == [101, 102]
        # masks empty batched
        enc_mask = BatchEncoding({"assistant_masks": [[]]})
        _add_token(enc_mask, 1, 0, "assistant_masks")
        out_mask = enc_mask
        assert out_mask["assistant_masks"] == [[1]]
        _add_token(out_mask, 1, -1, "assistant_masks")
        assert out_mask["assistant_masks"] == [[1, 1]]

    def test_invalid_position_raises(self):
        enc = BatchEncoding({"input_ids": [5, 6]})
        with pytest.raises(ValueError):
            _add_token(enc, 999, 1, "input_ids")

    def test_invalid_sequence_type_raises(self):
        enc = BatchEncoding({"input_ids": "not-a-list"})
        with pytest.raises(ValueError):
            _add_token(enc, 101, 0, "input_ids")
