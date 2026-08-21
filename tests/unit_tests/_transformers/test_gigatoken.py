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

"""Unit tests for the optional gigatoken tokenizer backend (issue #3177)."""

import pytest

from nemo_automodel._transformers.tokenization import gigatoken as gt_backend


def test_is_available_returns_bool():
    assert isinstance(gt_backend.is_available(), bool)


def test_build_returns_none_when_unavailable(monkeypatch):
    # When gigatoken isn't installed, the builder returns None so callers fall back to HF.
    monkeypatch.setattr(gt_backend, "HAVE_GIGATOKEN", False)
    assert gt_backend.build_gigatoken_tokenizer(object()) is None


def test_build_wraps_hf_tokenizer_when_available(monkeypatch):
    # When available, the builder must call gigatoken.Tokenizer(hf).as_hf() and return it.
    class _FakeTokenizer:
        def __init__(self, hf):
            self.hf = hf

        def as_hf(self):
            return ("wrapped", self.hf)

    class _FakeGigatoken:
        Tokenizer = _FakeTokenizer

    monkeypatch.setattr(gt_backend, "HAVE_GIGATOKEN", True)
    monkeypatch.setattr(gt_backend, "gigatoken", _FakeGigatoken)

    hf = object()
    assert gt_backend.build_gigatoken_tokenizer(hf) == ("wrapped", hf)


def test_parity_with_hf_tokenizer():
    # If gigatoken is installed, its token IDs must match the HF tokenizer exactly.
    if not gt_backend.is_available():
        pytest.skip("gigatoken not installed")
    from transformers import AutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("gpt2 tokenizer unavailable (offline)")
    gt = gt_backend.build_gigatoken_tokenizer(hf)
    assert gt is not None
    text = "Hello, world! gigatoken parity check 123."
    assert list(gt.encode(text)) == list(hf.encode(text))
