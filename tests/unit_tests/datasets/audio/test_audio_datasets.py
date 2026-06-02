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

import io as _io
import sys as _sys

import numpy as _np
import pytest
import soundfile as _sf

import nemo_automodel.components.datasets.audio.datasets as ds


def test_make_cv17_dataset(monkeypatch):
    """End-to-end sanity check for `make_cv17_dataset`."""

    # Mock dataset with audio data and extra columns to test column removal
    class MockDataset:
        def __init__(self, data):
            self.data = data
            self.column_names = ["audio", "transcription", "extra_col1", "extra_col2", "unwanted_col"]

        def remove_columns(self, columns_to_remove):
            # Simulate column removal
            expected_removed = ["extra_col1", "extra_col2", "unwanted_col"]
            assert set(columns_to_remove) == set(expected_removed)
            return self.data

        def __iter__(self):
            return iter(self.data)

    fake_audio_data = [
        {
            "audio": {"array": [0.1, 0.2, 0.3, -0.1, -0.2], "sampling_rate": 16000},
            "transcription": "Merhaba, nasılsınız?",
        },
        {
            "audio": {"array": [0.5, -0.3, 0.8, 0.2, -0.1], "sampling_rate": 16000},
            "transcription": "Bu bir test cümlesidir.",
        },
    ]

    mock_dataset = MockDataset(fake_audio_data)

    # Patch `load_dataset` so no network call is issued
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **k: mock_dataset)

    result = ds.make_cv17_dataset()

    assert len(result) == len(fake_audio_data)
    for sample, src in zip(result, fake_audio_data, strict=True):
        assert set(sample.keys()) == {"conversation", "audio"}

        # Test conversation structure
        conversation = sample["conversation"]
        assert len(conversation) == 2
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[1]["content"] == src["transcription"]

        # Test audio tuple
        audio_array, sampling_rate = sample["audio"]
        assert audio_array == src["audio"]["array"]
        assert sampling_rate == src["audio"]["sampling_rate"]


# ---------------------------------------------------------------------------
# HF audio ASR dataset builder (Qwen-Omni)
# ---------------------------------------------------------------------------


def _make_wav_bytes(sampling_rate=16000, duration_seconds=0.5, frequency_hz=440.0):
    """Generate a short mono WAV blob for synthetic tests."""
    t = _np.linspace(0, duration_seconds, int(sampling_rate * duration_seconds), endpoint=False)
    waveform = 0.1 * _np.sin(2 * _np.pi * frequency_hz * t).astype(_np.float32)
    buf = _io.BytesIO()
    _sf.write(buf, waveform, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _SyntheticHFRows(rows):
    """Build a real HF ``Dataset`` from a list of ``{audio: {bytes,path}, text}`` rows.

    The audio column is stored as a plain ``struct<bytes, path>`` (no ``Audio``
    feature), because HF's ``Audio.encode_example`` requires ``torchcodec`` even
    for ``decode=False`` and that is absent from this env. The builder's
    downstream ``cast_column(audio, Audio(decode=False))`` is monkey-patched
    away below; the storage layout already matches what the builder's lazy
    ``with_transform`` callback consumes, so the cast is functionally a no-op.
    """
    from datasets import Dataset as _Dataset

    columns = sorted({k for row in rows for k in row.keys()}) if rows else ["audio", "text"]
    data = {col: [r.get(col) for r in rows] for col in columns}
    dataset = _Dataset.from_dict(data)
    # Replace cast_column on the produced instance so the builder's call
    # ``cast_column(audio, Audio(decode=False))`` does not pull torchcodec.
    dataset.cast_column = lambda column_name, _feature: dataset  # type: ignore[assignment]
    return dataset


def test_make_hf_audio_asr_dataset_bytes_branch(monkeypatch):
    """The bytes branch decodes via soundfile and emits the Qwen-Omni schema."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": wav, "path": None}, "text": "你好"},
            {"audio": {"bytes": wav, "path": None}, "text": "侬好"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        split="train",
        sampling_rate=16000,
    )

    assert len(rows) == 2
    for row, src in zip(rows, fake_rows):
        assert list(row.keys()) == ["conversation"]
        conv = row["conversation"]
        # Default system_prompt is None → no system turn.
        assert [t["role"] for t in conv] == ["user", "assistant"]

        # user turn carries the decoded audio
        user_content = conv[0]["content"]
        assert isinstance(user_content, list) and len(user_content) == 1
        audio_item = user_content[0]
        assert audio_item["type"] == "audio"
        waveform = audio_item["audio"]
        assert isinstance(waveform, _np.ndarray)
        assert waveform.dtype == _np.float32
        assert waveform.ndim == 1

        # assistant turn carries the transcript
        assistant_content = conv[1]["content"]
        assert assistant_content == [{"type": "text", "text": src["text"]}]


def test_make_hf_audio_asr_dataset_path_branch(monkeypatch, tmp_path):
    """The path branch decodes via soundfile when no in-memory bytes are present."""
    wav_path = tmp_path / "sample.wav"
    _sf.write(str(wav_path), _np.zeros(800, dtype=_np.float32), 16000, format="WAV", subtype="PCM_16")
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": None, "path": str(wav_path)}, "text": "侬好"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    assert len(rows) == 1
    # Default system_prompt is None → user turn at index 0.
    waveform = rows[0]["conversation"][0]["content"][0]["audio"]
    assert waveform.dtype == _np.float32
    assert waveform.ndim == 1


def test_make_hf_audio_asr_dataset_raises_when_audio_cell_empty(monkeypatch):
    """Both ``bytes`` and ``path`` missing must raise a clear ValueError.

    With the lazy ``with_transform`` builder this fires at access time, not at
    construction time — so the assertion is anchored on ``rows[0]``.
    """
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": None, "path": None}, "text": "x"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    with pytest.raises(ValueError, match="neither 'bytes' nor 'path'"):
        _ = rows[0]


def test_make_hf_audio_asr_dataset_drops_empty_text(monkeypatch):
    """Default behaviour skips samples whose transcript is empty/whitespace."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": wav, "path": None}, "text": "  "},
            {"audio": {"bytes": wav, "path": None}, "text": "侬好"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    assert len(rows) == 1
    # Default system_prompt is None → assistant turn at index 1.
    assert rows[0]["conversation"][1]["content"][0]["text"] == "侬好"


def test_make_hf_audio_asr_dataset_module_does_not_import_torchcodec():
    """The dataset module must not transitively pull in torchcodec."""
    assert "torchcodec" not in _sys.modules
    # The module under test (already imported at the top of this file as ``ds``)
    # must not import torchcodec at module load time either.
    # ``ds.__dict__`` should not contain a top-level ``torchcodec`` binding.
    assert "torchcodec" not in ds.__dict__


def test_make_hf_audio_asr_dataset_resamples_when_sr_differs(monkeypatch):
    """When source SR != target SR, the waveform is resampled and stays float32 mono."""
    wav = _make_wav_bytes(sampling_rate=8000, duration_seconds=0.25)
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        sampling_rate=16000,
    )
    # Default system_prompt is None → user turn at index 0.
    waveform = rows[0]["conversation"][0]["content"][0]["audio"]
    assert waveform.dtype == _np.float32
    assert waveform.ndim == 1
    # 0.25s at 16 kHz target ≈ 4000 samples (allow ±a few for resample_poly polyphase rounding).
    assert abs(waveform.shape[0] - 4000) <= 8


def test_make_hf_audio_asr_dataset_user_prompt_appears_before_audio(monkeypatch):
    """When ``user_prompt`` is set, it becomes the first text item in the user turn."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        system_prompt="Transcribe.",
        user_prompt="please transcribe",
    )
    assert len(rows) == 1
    conv = rows[0]["conversation"]
    # Explicit system_prompt is set → full three-turn shape.
    assert [t["role"] for t in conv] == ["system", "user", "assistant"]
    user_content = conv[1]["content"]
    # First user item is the text prompt, second is the audio ndarray.
    assert user_content[0] == {"type": "text", "text": "please transcribe"}
    assert user_content[1]["type"] == "audio"
    assert isinstance(user_content[1]["audio"], _np.ndarray)


def test_make_hf_audio_asr_dataset_system_none_drops_system_turn(monkeypatch):
    """``system_prompt=None`` (or empty) must drop the system turn entirely."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        system_prompt=None,
        user_prompt="please transcribe",
    )
    assert len(rows) == 1
    conv = rows[0]["conversation"]
    # No system turn.
    assert [t["role"] for t in conv] == ["user", "assistant"]
    user_content = conv[0]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "audio"


def test_make_hf_audio_asr_dataset_both_prompts_none(monkeypatch):
    """When both prompts are None the user turn carries only the audio item."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        system_prompt=None,
        user_prompt=None,
    )
    conv = rows[0]["conversation"]
    assert [t["role"] for t in conv] == ["user", "assistant"]
    assert len(conv[0]["content"]) == 1
    assert conv[0]["content"][0]["type"] == "audio"


def test_make_hf_audio_asr_dataset_blank_prompts_drop(monkeypatch):
    """Whitespace-only prompts are treated as absent."""
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        system_prompt="   ",
        user_prompt="\t\n",
    )
    conv = rows[0]["conversation"]
    assert [t["role"] for t in conv] == ["user", "assistant"]
    assert len(conv[0]["content"]) == 1
    assert conv[0]["content"][0]["type"] == "audio"


def test_make_hf_audio_asr_dataset_is_lazy_no_decode_at_construction(monkeypatch):
    """Builder must NOT call the audio decoder at construction time.

    The decode helper is recorded on each call; constructing the dataset must
    not trigger any decode. Only ``rows[0]`` (the lazy transform access) should.
    """
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": wav, "path": None}, "text": "你好"},
            {"audio": {"bytes": wav, "path": None}, "text": "侬好"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    decode_calls = []
    original_decode = ds._decode_audio_cell_to_mono_float32

    def _spy(audio_cell, target_sampling_rate):
        decode_calls.append(target_sampling_rate)
        return original_decode(audio_cell, target_sampling_rate)

    monkeypatch.setattr(ds, "_decode_audio_cell_to_mono_float32", _spy)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    # Construction must not have decoded any audio.
    assert decode_calls == [], f"decode ran at construction time: {len(decode_calls)} calls"
    # Length is O(1) (Arrow row count); does not iterate.
    assert len(rows) == 2
    assert decode_calls == []
    # First __getitem__ triggers exactly one decode.
    _ = rows[0]
    assert len(decode_calls) == 1
    # Second __getitem__ triggers one more (no caching at this layer).
    _ = rows[1]
    assert len(decode_calls) == 2


def test_make_hf_audio_asr_dataset_default_system_prompt_is_none(monkeypatch):
    """Without overriding ``system_prompt``, the builder emits no system turn.

    Pins down the post-rename default: the builder is dataset-agnostic, so its
    default prompt shape is the most neutral one (``user → assistant``).
    """
    wav = _make_wav_bytes()
    fake_rows = _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "你好"}])
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    conv = rows[0]["conversation"]
    assert [t["role"] for t in conv] == ["user", "assistant"]
    # User turn carries only the audio (no text item).
    assert conv[0]["content"][0]["type"] == "audio"


def test_make_hf_audio_asr_dataset_passes_name_to_load_dataset(monkeypatch):
    """``name`` is forwarded to ``datasets.load_dataset`` as the subset/config.

    AMI requires ``name='ihm'`` or ``name='sdm'``; CommonVoice requires the
    language code. The builder must expose this as a first-class parameter so
    YAML files don't have to round-trip through ``**load_kwargs``.
    """
    wav = _make_wav_bytes()
    captured_kwargs = {}

    def _spy_load_dataset(path, *args, **kwargs):
        captured_kwargs["path"] = path
        captured_kwargs.update(kwargs)
        # Return a real fake dataset so the rest of the builder can run.
        return _SyntheticHFRows([{"audio": {"bytes": wav, "path": None}, "text": "x"}])

    monkeypatch.setattr(ds, "load_dataset", _spy_load_dataset)

    ds.make_hf_audio_asr_dataset(
        path_or_dataset="edinburghcstr/ami",
        name="ihm",
        split="train[:1]",
    )
    assert captured_kwargs["path"] == "edinburghcstr/ami"
    assert captured_kwargs["name"] == "ihm"
    assert captured_kwargs["split"] == "train[:1]"


def test_make_hf_audio_asr_dataset_min_duration_filters_short_bytes(monkeypatch):
    """``min_audio_duration_seconds`` drops sub-threshold samples in the bytes branch.

    The HF Qwen-Omni Whisper feature extractor crashes on sub-second clips
    due to an off-by-one between ``input_features`` and
    ``feature_attention_mask``; the builder exposes this filter to keep
    AMI / CommonVoice-style corpora trainable.
    """
    short_wav = _make_wav_bytes(duration_seconds=0.25)
    long_wav = _make_wav_bytes(duration_seconds=1.5)
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": short_wav, "path": None}, "text": "short"},
            {"audio": {"bytes": long_wav, "path": None}, "text": "long"},
            {"audio": {"bytes": short_wav, "path": None}, "text": "short2"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        min_audio_duration_seconds=1.0,
    )
    assert len(rows) == 1
    assert rows[0]["conversation"][1]["content"][0]["text"] == "long"


def test_make_hf_audio_asr_dataset_min_duration_filters_short_paths(monkeypatch, tmp_path):
    """``min_audio_duration_seconds`` also covers the path branch via sf.info."""
    short_path = tmp_path / "short.wav"
    long_path = tmp_path / "long.wav"
    _sf.write(str(short_path), _np.zeros(800, dtype=_np.float32), 16000, format="WAV", subtype="PCM_16")
    _sf.write(str(long_path), _np.zeros(24000, dtype=_np.float32), 16000, format="WAV", subtype="PCM_16")
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": None, "path": str(short_path)}, "text": "short"},
            {"audio": {"bytes": None, "path": str(long_path)}, "text": "long"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(
        path_or_dataset="ignored",
        min_audio_duration_seconds=1.0,
    )
    assert len(rows) == 1
    assert rows[0]["conversation"][1]["content"][0]["text"] == "long"


def test_make_hf_audio_asr_dataset_min_duration_none_keeps_all(monkeypatch):
    """``min_audio_duration_seconds=None`` (default) skips the filter entirely."""
    short_wav = _make_wav_bytes(duration_seconds=0.25)
    fake_rows = _SyntheticHFRows(
        [
            {"audio": {"bytes": short_wav, "path": None}, "text": "a"},
            {"audio": {"bytes": short_wav, "path": None}, "text": "b"},
        ]
    )
    monkeypatch.setattr(ds, "load_dataset", lambda *a, **kw: fake_rows)

    rows = ds.make_hf_audio_asr_dataset(path_or_dataset="ignored")
    assert len(rows) == 2
