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

import sys

import numpy as _np
import pytest
import torch


@pytest.fixture()
def audio_collate_mod():
    # Return the live module (no reload): tests mutate it only via monkeypatch,
    # which restores state on teardown. Reloading would mint new function objects
    # and break ``is`` identity against the vlm ``COLLATE_FNS`` registry, which
    # references the original audio.collate_fns singletons.
    import nemo_automodel.components.datasets.audio.collate_fns as _m

    return _m


@pytest.fixture()
def vlm_collate_mod():
    import nemo_automodel.components.datasets.vlm.collate_fns as _m

    return _m


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return None  # Return None to trigger default fallback

    def decode(self, token):
        if isinstance(token, torch.Tensor):
            token = token.item()
        return str(token)


_ASR_ASSISTANT_TEXT = "你好"


def _asr_conversation(transcript=_ASR_ASSISTANT_TEXT):
    waveform = _np.zeros(800, dtype=_np.float32)
    return [
        {"role": "system", "content": "Transcribe."},
        {"role": "user", "content": [{"type": "audio", "audio": waveform}]},
        {"role": "assistant", "content": [{"type": "text", "text": transcript}]},
    ]


class DummyQwen3OmniAsrProcessor:
    """Mock that mimics Qwen3OmniMoeProcessor.__call__ for ASR usage.

    Returns deterministic ``input_ids`` and audio-feature tensors keyed by the
    audio kwarg the new collate is required to pass. Tracks every call so tests
    can assert what was forwarded.
    """

    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)
        self.call_kwargs = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt is False, "ASR collate must call apply_chat_template(add_generation_prompt=False)"
        assert tokenize is False
        return "chat"

    def __call__(self, *, text, return_tensors, padding, audio=None, **kwargs):
        assert return_tensors == "pt"
        assert padding is True
        self.call_kwargs.append({"text": list(text), "audio": audio, "padding_side": kwargs.get("padding_side")})
        batch_size = len(text)
        input_ids = torch.arange(1, 7).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones_like(input_ids)
        num_audios = 0 if audio is None else len(audio)
        if num_audios == 0:
            num_audios = batch_size
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "input_features": torch.zeros(num_audios, 128, 32, dtype=torch.float32),
            "feature_attention_mask": torch.ones(num_audios, 32, dtype=torch.long),
        }


def test_qwen3_omni_asr_registry_guard(vlm_collate_mod, audio_collate_mod):
    """The global registry must still point at the original (non-ASR) Omni collate."""
    assert vlm_collate_mod.COLLATE_FNS["Qwen3OmniMoeProcessor"] is vlm_collate_mod.qwen3_omni_collate_fn
    assert vlm_collate_mod.COLLATE_FNS["Qwen3OmniMoeProcessor"] is not audio_collate_mod.qwen3_omni_asr_collate_fn


def test_qwen2_5_omni_asr_registry_points_at_audio_collate(vlm_collate_mod, audio_collate_mod):
    """The registry's Qwen2.5-Omni entry must resolve to the moved ASR collate."""
    assert vlm_collate_mod.COLLATE_FNS["Qwen2_5OmniProcessor"] is audio_collate_mod.qwen2_5_omni_asr_collate_fn


def test_qwen3_omni_asr_extract_audios_from_conversation(audio_collate_mod):
    """Audio payloads must be pulled from any user-turn ``{type:audio,audio:...}`` item."""
    conv = _asr_conversation()
    audios = audio_collate_mod._extract_audios_from_conversation(conv)
    assert len(audios) == 1
    assert isinstance(audios[0], _np.ndarray)


def test_qwen3_omni_asr_collate_shapes_and_kwargs(audio_collate_mod, monkeypatch):
    """The collate must produce shifted labels, slice same-shape tensors, and pass audio=."""
    labels_stub = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=torch.long)

    def fake_build_labels(input_ids, conversations, processor_arg):
        assert input_ids.shape == (2, 6)
        assert len(conversations) == 2
        return labels_stub

    monkeypatch.setattr(audio_collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    processor = DummyQwen3OmniAsrProcessor()
    batch = audio_collate_mod.qwen3_omni_asr_collate_fn(
        [{"conversation": _asr_conversation()} for _ in range(2)],
        processor,
    )

    # The audio kwarg must have been forwarded with one waveform per sample.
    assert len(processor.call_kwargs) == 1
    assert processor.call_kwargs[0]["audio"] is not None
    assert len(processor.call_kwargs[0]["audio"]) == 2
    # The collate must pin padding_side="right" to align with recipe token accounting.
    assert processor.call_kwargs[0]["padding_side"] == "right"

    # Pre-shifted labels and same-shape tensors sliced to [:, :-1] (length 5).
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    # Same-shape labels[:, 1:] of length-6 labels_stub == values at columns 1..5.
    assert torch.equal(batch["labels"], labels_stub[:, 1:])

    # Audio feature tensors must NOT be sliced (their shape differs from input_ids).
    assert batch["input_features"].shape == (2, 128, 32)
    assert batch["feature_attention_mask"].shape == (2, 32)


def test_qwen3_omni_asr_raises_when_no_assistant_turn(audio_collate_mod):
    """An assistant-less conversation must error rather than yield NaN loss."""
    processor = DummyQwen3OmniAsrProcessor()
    bad = [
        {"role": "system", "content": "Transcribe."},
        {"role": "user", "content": [{"type": "audio", "audio": _np.zeros(800, dtype=_np.float32)}]},
    ]
    with pytest.raises(ValueError, match="assistant"):
        audio_collate_mod.qwen3_omni_asr_collate_fn([{"conversation": bad}], processor)


def test_qwen3_omni_asr_raises_when_assistant_text_empty(audio_collate_mod):
    """A whitespace-only assistant turn must error."""
    processor = DummyQwen3OmniAsrProcessor()
    bad_conv = _asr_conversation(transcript="  ")
    with pytest.raises(ValueError, match="assistant"):
        audio_collate_mod.qwen3_omni_asr_collate_fn([{"conversation": bad_conv}], processor)


def test_qwen3_omni_asr_works_when_qwen_omni_utils_missing(audio_collate_mod, monkeypatch):
    """The collate must NOT depend on qwen_omni_utils, even when it is missing/poisoned.

    ``audio.collate_fns`` never imports ``qwen_omni_utils`` at all, so poisoning the
    module slot must not affect the collate.
    """
    # Poison the qwen_omni_utils slot so any accidental import would fail.
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", None)

    labels_stub = torch.tensor([[100, 101, 102, 103, 104, 105]], dtype=torch.long)
    monkeypatch.setattr(
        audio_collate_mod,
        "build_labels_from_template",
        lambda input_ids, conversations, processor_arg: labels_stub,
        raising=True,
    )

    processor = DummyQwen3OmniAsrProcessor()
    batch = audio_collate_mod.qwen3_omni_asr_collate_fn([{"conversation": _asr_conversation()}], processor)
    assert batch["input_ids"].shape == (1, 5)
    assert torch.equal(batch["labels"], labels_stub[:, 1:])


def test_qwen3_omni_asr_function_body_does_not_import_qwen_omni_utils(audio_collate_mod):
    """Static guard: the ASR collate must not import qwen_omni_utils or call process_mm_info.

    Parses the function body via AST so the explanatory docstring (which intentionally
    explains *why* this collate avoids qwen_omni_utils) does not trip the check.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(audio_collate_mod.qwen3_omni_asr_collate_fn))
    tree = ast.parse(src)
    func_def = tree.body[0]
    assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))

    body = list(func_def.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    body_src = "\n".join(ast.unparse(node) for node in body)
    assert "qwen_omni_utils" not in body_src
    assert "process_mm_info" not in body_src


def test_qwen3_omni_asr_validate_helper_coerces_float64_to_float32(audio_collate_mod):
    """A 1-D float64 waveform must be coerced to float32 (no raise)."""
    waveform = _np.zeros(400, dtype=_np.float64)
    coerced = audio_collate_mod._validate_and_coerce_audio_payload(waveform, sample_index=0)
    assert isinstance(coerced, _np.ndarray)
    assert coerced.dtype == _np.float32
    assert coerced.ndim == 1
    assert coerced.shape == (400,)


def test_qwen3_omni_asr_validate_helper_rejects_2d_audio(audio_collate_mod):
    """A non-1-D audio payload must raise ValueError naming sample index and shape/dtype."""
    waveform_2d = _np.zeros((2, 400), dtype=_np.float32)
    with pytest.raises(ValueError, match=r"sample\[3\] audio payload must be 1-D"):
        audio_collate_mod._validate_and_coerce_audio_payload(waveform_2d, sample_index=3)


def test_qwen3_omni_asr_collate_coerces_float64_inputs(audio_collate_mod, monkeypatch):
    """End-to-end: the collate must accept a float64 waveform and forward it as float32."""

    labels_stub = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.long)
    monkeypatch.setattr(
        audio_collate_mod,
        "build_labels_from_template",
        lambda input_ids, conversations, processor_arg: labels_stub,
        raising=True,
    )

    processor = DummyQwen3OmniAsrProcessor()
    conv = [
        {"role": "system", "content": "Transcribe."},
        {"role": "user", "content": [{"type": "audio", "audio": _np.zeros(400, dtype=_np.float64)}]},
        {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
    ]
    audio_collate_mod.qwen3_omni_asr_collate_fn([{"conversation": conv}], processor)

    # The collate must have coerced the float64 waveform to float32 BEFORE passing to the processor.
    forwarded_audio = processor.call_kwargs[0]["audio"]
    assert forwarded_audio is not None and len(forwarded_audio) == 1
    assert forwarded_audio[0].dtype == _np.float32
    assert forwarded_audio[0].ndim == 1


def test_qwen3_omni_asr_collate_rejects_2d_audio(audio_collate_mod):
    """End-to-end: a 2-D waveform inside the conversation must raise during collation."""
    processor = DummyQwen3OmniAsrProcessor()
    conv = [
        {"role": "system", "content": "Transcribe."},
        {"role": "user", "content": [{"type": "audio", "audio": _np.zeros((2, 400), dtype=_np.float32)}]},
        {"role": "assistant", "content": [{"type": "text", "text": "你好"}]},
    ]
    with pytest.raises(ValueError, match=r"sample\[0\].*1-D"):
        audio_collate_mod.qwen3_omni_asr_collate_fn([{"conversation": conv}], processor)
