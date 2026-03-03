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

"""Functional tests for NeMoAutoTokenizer with Mistral models.

Verifies that NeMoAutoTokenizer correctly dispatches to MistralCommonBackend
for Mistral model types and that basic tokenization operations work.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.tokenization_mistral_common import MistralCommonBackend

_TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/home/TestData/automodel")
_TOKENIZER_BASE = Path(_TEST_DATA_DIR) / "tokenizers"
MISTRAL_7B_INSTRUCT_PATH = _TOKENIZER_BASE / "Mistral-7B-Instruct-v0.1"


@pytest.fixture
def mistral_tokenizer_path():
    assert MISTRAL_7B_INSTRUCT_PATH.exists(), "path not exists"
    return str(MISTRAL_7B_INSTRUCT_PATH)


@pytest.fixture
def simple_conversation():
    return [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And of Germany?"},
    ]


class TestMistralTokenizerDispatch:
    """Verify NeMoAutoTokenizer dispatches to MistralCommonBackend for Mistral models."""

    def test_from_pretrained_returns_mistral_common_backend(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert isinstance(tokenizer, MistralCommonBackend)

    def test_force_default_returns_hf_tokenizer(self, mistral_tokenizer_path):
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path, force_default=True)
        assert isinstance(tokenizer, NeMoAutoTokenizerWithBosEosEnforced)
        assert not isinstance(tokenizer, MistralCommonBackend)

    def test_force_hf_returns_raw_hf_tokenizer(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path, force_hf=True)
        assert not isinstance(tokenizer, MistralCommonBackend)


class TestMistralCommonBackendTokenization:
    """Verify basic encode/decode round-trip and special token handling."""

    def test_encode_decode_roundtrip(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        text = "Hello, how are you doing today?"
        token_ids = tokenizer.encode(text)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)
        assert len(token_ids) > 0

        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert text in decoded

    def test_vocab_size_positive(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert tokenizer.vocab_size > 0

    def test_special_tokens_defined(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

    def test_apply_chat_template(self, mistral_tokenizer_path, simple_conversation):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        result = tokenizer.apply_chat_template(
            simple_conversation,
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = result["input_ids"] if isinstance(result, Mapping) else result
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_apply_chat_template_no_tokenize(self, mistral_tokenizer_path, simple_conversation):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        result = tokenizer.apply_chat_template(
            simple_conversation,
            tokenize=False,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_encode(self, mistral_tokenizer_path):
        tokenizer = NeMoAutoTokenizer.from_pretrained(mistral_tokenizer_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        texts = ["Hello world", "How are you?"]
        result = tokenizer(texts, padding=True)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2
