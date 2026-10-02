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


def test_wrapper_routes_encode_and_call_to_gigatoken():
    # encode/__call__ go to gigatoken; every other attribute falls through to HF.
    class _GT:
        def encode(self, *a, **k):
            return "GT_ENCODE"

        def __call__(self, *a, **k):
            return "GT_CALL"

    class _HF:
        some_attr = "HF_ATTR"

        def encode(self, *a, **k):
            return "HF_ENCODE"

        def decode(self, *a, **k):
            return "HF_DECODE"

    wrapped = gt_backend.GigatokenTokenizer(_HF(), _GT())
    assert wrapped.encode("x") == "GT_ENCODE"
    assert wrapped("x") == "GT_CALL"
    assert wrapped.some_attr == "HF_ATTR"
    assert wrapped.decode([1]) == "HF_DECODE"


def test_encode_delegates_to_hf_when_bos_eos_enforced():
    # When the HF tokenizer enforces BOS/EOS, gigatoken (which can't) must be bypassed.
    class _GT:
        def encode(self, *a, **k):
            return [1, 2, 3]

        def __call__(self, *a, **k):
            return {"input_ids": [1, 2, 3]}

    class _HF:
        add_bos_token = True
        add_eos_token = False

        def encode(self, *a, **k):
            return [999, 1, 2, 3]  # HF prepends the enforced BOS

        def __call__(self, *a, **k):
            return {"input_ids": [999, 1, 2, 3]}

    wrapped = gt_backend.GigatokenTokenizer(_HF(), _GT())
    assert wrapped.encode("x") == [999, 1, 2, 3]
    assert wrapped("x") == {"input_ids": [999, 1, 2, 3]}


def test_encode_uses_gigatoken_without_enforcement():
    class _GT:
        def encode(self, *a, **k):
            return [1, 2, 3]

        def __call__(self, *a, **k):
            return "GT_CALL"

    class _HF:
        add_bos_token = False
        add_eos_token = False

        def encode(self, *a, **k):
            return [999]

    wrapped = gt_backend.GigatokenTokenizer(_HF(), _GT())
    assert wrapped.encode("x") == [1, 2, 3]
    assert wrapped("x") == "GT_CALL"


def test_encode_uses_gigatoken_when_add_special_tokens_false():
    # add_special_tokens=False means no BOS/EOS anyway, so gigatoken is safe even if enforced.
    class _GT:
        def encode(self, *a, **k):
            return [1, 2, 3]

    class _HF:
        add_bos_token = True

        def encode(self, *a, **k):
            return [999, 1, 2, 3]

    wrapped = gt_backend.GigatokenTokenizer(_HF(), _GT())
    assert wrapped.encode("x", add_special_tokens=False) == [1, 2, 3]


def test_dunders_forward_to_hf():
    # Dunders resolve on the type, so they need explicit forwarding (not __getattr__).
    class _GT:
        pass

    class _HF:
        def __len__(self):
            return 42

        def __contains__(self, item):
            return item == "tok"

        def __getitem__(self, key):
            return "id"

    wrapped = gt_backend.GigatokenTokenizer(_HF(), _GT())
    assert len(wrapped) == 42
    assert "tok" in wrapped
    assert wrapped["anything"] == "id"


def test_maybe_wrap_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(gt_backend, "HAVE_GIGATOKEN", False)
    hf = object()
    assert gt_backend.maybe_wrap_with_gigatoken(hf) is hf


def test_maybe_wrap_falls_back_on_build_error(monkeypatch):
    # An unsupported (non-BPE) tokenizer makes the builder raise; we return HF unchanged.
    monkeypatch.setattr(gt_backend, "HAVE_GIGATOKEN", True)

    def _boom(_):
        raise RuntimeError("unsupported tokenizer")

    monkeypatch.setattr(gt_backend, "build_gigatoken_tokenizer", _boom)
    hf = object()
    assert gt_backend.maybe_wrap_with_gigatoken(hf) is hf


def test_maybe_wrap_wraps_when_available(monkeypatch):
    monkeypatch.setattr(gt_backend, "HAVE_GIGATOKEN", True)
    monkeypatch.setattr(gt_backend, "build_gigatoken_tokenizer", lambda hf: "FAKE_GT")
    hf = object()
    wrapped = gt_backend.maybe_wrap_with_gigatoken(hf)
    assert isinstance(wrapped, gt_backend.GigatokenTokenizer)
    assert wrapped._gt == "FAKE_GT"
    assert wrapped._hf is hf


def test_from_pretrained_use_gigatoken_wires_backend():
    # End-to-end: use_gigatoken=True wraps the tokenizer, encode goes through gigatoken
    # with identical IDs, and non-encode behavior still delegates to HF.
    if not gt_backend.is_available():
        pytest.skip("gigatoken not installed")
    from transformers import AutoTokenizer

    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained("gpt2")
        tok = NeMoAutoTokenizer.from_pretrained("gpt2", use_gigatoken=True)
    except OSError:
        pytest.skip("gpt2 tokenizer unavailable (offline)")
    assert isinstance(tok, gt_backend.GigatokenTokenizer)
    text = "Hello, world! gigatoken wiring 123."
    assert list(tok.encode(text)) == list(hf.encode(text))
    assert tok.decode(tok.encode(text)) == hf.decode(hf.encode(text))
