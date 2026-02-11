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

"""
Tests for MistralCommonBackend pad_token / pad_token_id handling.

Regression test for the bug where accessing `pad_token` on a
MistralCommonBackend tokenizer whose underlying SentencePiece model
has no pad token (pad_id == -1) raised:

    IndexError: piece id is out of range.

The fix makes:
  - pad_token_id return None when the underlying pad_id is invalid (<0).
  - pad_token return None when pad_token_id is None or out of range.
  - Both properties settable so _add_pad_token() can override them.
"""

import pytest
from unittest.mock import MagicMock, patch

from nemo_automodel._transformers.tokenization.tokenization_mistral_common import (
    MistralCommonBackend,
    MistralTokenizerType,
)
from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token


# ---------------------------------------------------------------------------
# Helpers: lightweight stub that mimics MistralCommonBackend without needing
# a real SentencePiece file on disk.
# ---------------------------------------------------------------------------

def _make_stub_tokenizer(pad_id=-1, eos_id=2, bos_id=1, unk_id=0, vocab_size=32000):
    """
    Build a MistralCommonBackend instance whose internals are mocked so we
    don't need a real tokenizer file.
    """
    # Mock the SentencePiece inner tokenizer
    inner_tok = MagicMock()
    inner_tok.pad_id = pad_id
    inner_tok.eos_id = eos_id
    inner_tok.bos_id = bos_id
    inner_tok.unk_id = unk_id
    inner_tok.n_words = vocab_size
    # Control tokens set (used by _is_control_token for spm type)
    inner_tok._control_tokens = {bos_id, eos_id}

    def _id_to_piece(token_id):
        if token_id < 0 or token_id >= vocab_size:
            raise IndexError("piece id is out of range.")
        return f"<tok_{token_id}>"

    inner_tok.id_to_piece = _id_to_piece

    # instruct_tokenizer wraps inner_tok
    instruct_tok = MagicMock()
    instruct_tok.tokenizer = inner_tok

    # MistralTokenizer wraps instruct_tok
    mistral_tok = MagicMock()
    mistral_tok.instruct_tokenizer = instruct_tok

    # Bypass __init__ by constructing a bare object and injecting fields
    with patch.object(MistralCommonBackend, "__init__", lambda self, *a, **kw: None):
        backend = MistralCommonBackend.__new__(MistralCommonBackend)

    # Manually assign the attributes that __init__ would create
    backend.tokenizer = mistral_tok
    backend._pad_token_id_override = None
    backend._pad_token_override = None
    backend._all_special_tokens_ids = {bos_id, eos_id}
    backend._tokenizer_type = MistralTokenizerType.spm

    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPadTokenIdProperty:
    """pad_token_id should return None for invalid underlying pad_id."""

    def test_returns_none_when_pad_id_is_negative(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        assert tok.pad_token_id is None

    def test_returns_none_when_pad_id_is_none(self):
        tok = _make_stub_tokenizer(pad_id=None)
        assert tok.pad_token_id is None

    def test_returns_valid_pad_id(self):
        tok = _make_stub_tokenizer(pad_id=3)
        assert tok.pad_token_id == 3

    def test_setter_overrides_pad_token_id(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        assert tok.pad_token_id is None
        tok.pad_token_id = 42
        assert tok.pad_token_id == 42

    def test_setter_override_takes_precedence(self):
        tok = _make_stub_tokenizer(pad_id=3)
        assert tok.pad_token_id == 3
        tok.pad_token_id = 99
        assert tok.pad_token_id == 99


class TestPadTokenProperty:
    """pad_token should return None when pad_token_id is None / out of range."""

    def test_returns_none_when_no_pad_token(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        # This was the original crash: IndexError: piece id is out of range.
        assert tok.pad_token is None

    def test_returns_string_when_pad_id_valid(self):
        tok = _make_stub_tokenizer(pad_id=5)
        assert tok.pad_token == "<tok_5>"

    def test_setter_overrides_pad_token(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        assert tok.pad_token is None
        tok.pad_token = "<pad>"
        assert tok.pad_token == "<pad>"

    def test_returns_none_for_out_of_range_pad_id(self):
        """Even if pad_id is >= 0, if it's >= vocab_size, return None."""
        tok = _make_stub_tokenizer(pad_id=99999, vocab_size=32000)
        assert tok.pad_token is None


class TestGetattr:
    """
    getattr(tokenizer, "pad_token", None) must not raise when the
    underlying tokenizer has no pad token.  This is the pattern used
    by _add_pad_token and many HF utilities.
    """

    def test_getattr_pad_token_returns_none(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        result = getattr(tok, "pad_token", None)
        assert result is None

    def test_getattr_pad_token_id_returns_none(self):
        tok = _make_stub_tokenizer(pad_id=-1)
        result = getattr(tok, "pad_token_id", None)
        assert result is None


class TestAddPadTokenIntegration:
    """
    _add_pad_token(tokenizer) should not crash and should correctly
    fall back to eos_token_id / eos_token when pad is missing.
    """

    def test_add_pad_token_with_missing_pad(self):
        tok = _make_stub_tokenizer(pad_id=-1, eos_id=2)
        # Before: no pad
        assert tok.pad_token_id is None
        assert tok.pad_token is None

        pad_id = _add_pad_token(tok)

        # _add_pad_token returns None when it had to set pad_token_id itself
        assert pad_id is None
        # But the tokenizer now has pad_token_id set to eos_token_id
        assert tok.pad_token_id == 2
        # And pad_token is set to eos_token string
        assert tok.pad_token is not None

    def test_add_pad_token_with_existing_pad(self):
        tok = _make_stub_tokenizer(pad_id=3, eos_id=2)
        assert tok.pad_token_id == 3
        assert tok.pad_token == "<tok_3>"

        pad_id = _add_pad_token(tok)

        # Should return the existing pad_token_id
        assert pad_id == 3
        # pad_token_id should remain unchanged
        assert tok.pad_token_id == 3
