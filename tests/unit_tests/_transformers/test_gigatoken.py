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


class _TokenizingDataset:
    """A dataset that encodes text with the tokenizer it holds.

    Defined at module level so ``spawn``/``forkserver`` DataLoader workers can pickle it
    (and, with it, the gigatoken-wrapped tokenizer) when shipping it to worker processes.
    """

    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer.encode(self.texts[idx])


def _identity_collate(batch):
    """Return the batch unchanged; module-level so it survives pickling to a worker."""
    return batch


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


# --- P1: pickle support (DataLoader workers with num_workers>0 pickle the tokenizer) ---


def test_pickle_drops_and_rebuilds_gigatoken(monkeypatch):
    # The gigatoken _gt (a Rust object) is unpicklable. __getstate__ must drop it and
    # __setstate__ must rebuild it from _hf, so pickling never touches _gt.
    import pickle

    class _Unpicklable:
        def __reduce__(self):
            raise TypeError("cannot pickle gigatoken Rust backend")

    monkeypatch.setattr(gt_backend, "build_gigatoken_tokenizer", lambda hf: ("REBUILT", hf))
    hf = {"name": "hf-tokenizer"}  # a picklable stand-in for the HF tokenizer
    wrapped = gt_backend.GigatokenTokenizer(hf, _Unpicklable())

    restored = pickle.loads(pickle.dumps(wrapped))  # noqa: S301 - round-tripping our own object
    assert isinstance(restored, gt_backend.GigatokenTokenizer)
    assert restored._hf == hf
    assert restored._gt == ("REBUILT", hf)  # rebuilt from _hf, not the original _gt


def test_getstate_excludes_gigatoken_backend():
    wrapped = gt_backend.GigatokenTokenizer({"name": "hf"}, object())
    state = wrapped.__getstate__()
    assert "_gt" not in state
    assert state["_hf"] == {"name": "hf"}


def test_pickle_roundtrip_with_real_backend():
    # With gigatoken installed, a real tokenizer must survive a pickle round-trip and
    # still encode identically (this is exactly what a spawn/forkserver DataLoader does).
    if not gt_backend.is_available():
        pytest.skip("gigatoken not installed")
    import pickle

    from transformers import AutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("gpt2 tokenizer unavailable (offline)")
    wrapped = gt_backend.maybe_wrap_with_gigatoken(hf)
    assert isinstance(wrapped, gt_backend.GigatokenTokenizer)

    restored = pickle.loads(pickle.dumps(wrapped))  # noqa: S301 - round-tripping our own object
    text = "Hello, world! gigatoken pickle 123."
    assert list(restored.encode(text)) == list(wrapped.encode(text))


@pytest.mark.timeout(120)
def test_dataloader_worker_consumes_sample_with_real_backend():
    # akoumpa's exact repro: a DataLoader worker (forkserver) must pickle the
    # gigatoken-wrapped tokenizer, rebuild it in the worker, and consume a sample
    # without "cannot pickle 'BPETokenizer'". IDs must still match the HF tokenizer.
    if not gt_backend.is_available():
        pytest.skip("gigatoken not installed")
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    try:
        hf = AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("gpt2 tokenizer unavailable (offline)")
    wrapped = gt_backend.maybe_wrap_with_gigatoken(hf)
    assert isinstance(wrapped, gt_backend.GigatokenTokenizer)

    texts = ["Hello, world!", "gigatoken worker check 123."]
    loader = DataLoader(
        _TokenizingDataset(wrapped, texts),
        batch_size=1,
        num_workers=1,
        multiprocessing_context="forkserver",  # forces pickling (fork would not)
        collate_fn=_identity_collate,
    )

    got = [sample for batch in loader for sample in batch]
    expected = [hf.encode(t) for t in texts]
    assert got == expected


# --- P2: attribute writes must forward to both backends (not shadow on the wrapper) ---


def test_setattr_forwards_to_both_backends():
    # tok.pad_token = ... / tok.padding_side = ... must reach _hf AND _gt, or the
    # backends silently diverge from the wrapper.
    class _Backend:
        pass

    hf, gt = _Backend(), _Backend()
    wrapped = gt_backend.GigatokenTokenizer(hf, gt)

    wrapped.padding_side = "left"
    assert hf.padding_side == "left"
    assert gt.padding_side == "left"

    wrapped.pad_token = "<pad>"
    assert hf.pad_token == "<pad>"
    assert gt.pad_token == "<pad>"


def test_setattr_does_not_shadow_on_wrapper():
    # The write must not create a wrapper-level attribute that masks the backends.
    class _Backend:
        padding_side = "right"

    hf, gt = _Backend(), _Backend()
    wrapped = gt_backend.GigatokenTokenizer(hf, gt)
    wrapped.padding_side = "left"

    assert "padding_side" not in wrapped.__dict__  # not stored on the wrapper
    assert wrapped.padding_side == "left"  # read-back (via __getattr__) reflects the backend


def test_own_slots_stay_on_wrapper():
    # _hf/_gt are the wrapper's own attributes and must NOT be forwarded (they'd recurse).
    hf, gt = object(), object()
    wrapped = gt_backend.GigatokenTokenizer(hf, gt)
    assert wrapped.__dict__["_hf"] is hf
    assert wrapped.__dict__["_gt"] is gt


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_parity_after_setattr_with_real_backend(padding_side):
    # akoumpa's repro: after setting pad_token and padding_side on the wrapper, a padded
    # batch must match the HF tokenizer exactly (left AND right). Before the __setattr__
    # fix the wrapper produced [[31373, 0], ...] while HF produced [[50256, 31373], ...].
    if not gt_backend.is_available():
        pytest.skip("gigatoken not installed")
    from transformers import AutoTokenizer

    try:
        hf_ref = AutoTokenizer.from_pretrained("gpt2")
        hf_for_wrap = AutoTokenizer.from_pretrained("gpt2")
    except OSError:
        pytest.skip("gpt2 tokenizer unavailable (offline)")

    wrapped = gt_backend.maybe_wrap_with_gigatoken(hf_for_wrap)
    assert isinstance(wrapped, gt_backend.GigatokenTokenizer)

    # Configure the reference HF tokenizer directly.
    hf_ref.pad_token = hf_ref.eos_token
    hf_ref.padding_side = padding_side
    # Same settings through the wrapper -- these writes must reach both backends.
    wrapped.pad_token = wrapped.eos_token
    wrapped.padding_side = padding_side

    batch = ["hello", "hello world"]
    expected = hf_ref(batch, padding=True)["input_ids"]
    got = wrapped(batch, padding=True)["input_ids"]
    assert got == expected
