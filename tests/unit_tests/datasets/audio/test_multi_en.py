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

import io
import sys

import numpy as np
import soundfile as sf

import nemo_automodel.components.datasets.audio.datasets as ds_mod
import nemo_automodel.components.datasets.audio.multi_en as me
from nemo_automodel.components.datasets.audio.multi_en import Source


def _make_wav_bytes(sampling_rate=16000, duration_seconds=1.5):
    """Generate a short mono WAV blob for synthetic tests."""
    waveform = np.zeros(int(sampling_rate * duration_seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _fake_audio_dataset(rows, text_col):
    """Build a real HF ``Dataset`` with an audio struct + ``text_col`` + an extra column.

    ``cast_column(audio, Audio(decode=False))`` is patched to a no-op so the test
    never pulls torchcodec, mirroring the storage layout the builder consumes.
    """
    from datasets import Dataset as _Dataset

    data = {
        "audio": [r["audio"] for r in rows],
        text_col: [r["text"] for r in rows],
        "extra_meta": ["drop-me"] * len(rows),
    }
    dataset = _Dataset.from_dict(data)
    dataset.cast_column = lambda column_name, _feature: dataset  # type: ignore[assignment]
    return dataset


def _install_fake_load_dataset(monkeypatch, repo_to_rows):
    """Patch ``multi_en.load_dataset`` to return per-repo fake datasets."""

    def _fake_load_dataset(repo, *args, **kwargs):
        text_col, rows = repo_to_rows[repo]
        return _fake_audio_dataset(rows, text_col)

    monkeypatch.setattr(me, "load_dataset", _fake_load_dataset)


def test_normalize_text_uppercases_strips_tags_keeps_apostrophes_and_digits():
    assert me.normalize_text("It's a <COMMA> test_case.") == "IT'S A TEST CASE"
    assert me.normalize_text("route 66") == "ROUTE 66"
    assert me.normalize_text("hello, world!") == "HELLO WORLD"
    # curly apostrophe normalised to straight
    assert me.normalize_text("we’re here") == "WE'RE HERE"
    # tag-only / punctuation-only collapses to empty
    assert me.normalize_text("<MUSIC>") == ""


def test_source_mix_concat_and_source_labels(monkeypatch):
    """The concatenated mix carries audio/text/source columns with correct labels."""
    wav = _make_wav_bytes()
    sources = [
        Source("src_a", "repo/a", None, "train", "text"),
        Source("src_b", "repo/b", None, "train", "sentence"),
    ]
    _install_fake_load_dataset(
        monkeypatch,
        {
            "repo/a": ("text", [{"audio": {"bytes": wav, "path": None}, "text": "alpha one"}]),
            "repo/b": (
                "sentence",
                [
                    {"audio": {"bytes": wav, "path": None}, "text": "beta one"},
                    {"audio": {"bytes": wav, "path": None}, "text": "beta two"},
                ],
            ),
        },
    )

    mixed = me.build_multi_en_source_mix(sources=sources, shuffle_seed=None, min_audio_duration_seconds=None)

    assert set(mixed.column_names) == {"audio", "text", "source"}
    assert sorted(mixed["source"]) == ["src_a", "src_b", "src_b"]
    # Transcripts are normalised (uppercased) and the renamed column is "text".
    assert set(mixed["text"]) == {"ALPHA ONE", "BETA ONE", "BETA TWO"}


def test_missing_text_column_raises(monkeypatch):
    """A source whose declared text column is absent must raise a clear ValueError."""
    wav = _make_wav_bytes()
    sources = [Source("bad", "repo/bad", None, "train", "does_not_exist")]
    _install_fake_load_dataset(
        monkeypatch,
        {"repo/bad": ("text", [{"audio": {"bytes": wav, "path": None}, "text": "x"}])},
    )
    import pytest

    with pytest.raises(ValueError, match="does_not_exist"):
        me.build_multi_en_source_mix(sources=sources, shuffle_seed=None, min_audio_duration_seconds=None)


def test_per_source_limit_honored(monkeypatch):
    """``Source.limit`` caps the number of rows taken from that source."""
    wav = _make_wav_bytes()
    rows = [{"audio": {"bytes": wav, "path": None}, "text": f"row {i}"} for i in range(10)]
    sources = [Source("capped", "repo/c", None, "train", "text", limit=3)]
    _install_fake_load_dataset(monkeypatch, {"repo/c": ("text", rows)})

    mixed = me.build_multi_en_source_mix(sources=sources, shuffle_seed=None, min_audio_duration_seconds=None)
    assert len(mixed) == 3
    assert all(s == "capped" for s in mixed["source"])


def test_empty_after_normalization_is_dropped(monkeypatch):
    """Rows that normalise to an empty transcript (e.g. tag-only) are filtered out."""
    wav = _make_wav_bytes()
    rows = [
        {"audio": {"bytes": wav, "path": None}, "text": "<MUSIC>"},  # -> "" -> dropped
        {"audio": {"bytes": wav, "path": None}, "text": "real words"},
    ]
    sources = [Source("s", "repo/s", None, "train", "text")]
    _install_fake_load_dataset(monkeypatch, {"repo/s": ("text", rows)})

    mixed = me.build_multi_en_source_mix(sources=sources, shuffle_seed=None, min_audio_duration_seconds=None)
    assert len(mixed) == 1
    assert mixed["text"] == ["REAL WORDS"]


def test_shuffle_is_deterministic_for_fixed_seed(monkeypatch):
    """The same shuffle seed yields the same ordering of the mix."""
    wav = _make_wav_bytes()
    rows_a = [{"audio": {"bytes": wav, "path": None}, "text": f"a{i}"} for i in range(5)]
    rows_b = [{"audio": {"bytes": wav, "path": None}, "text": f"b{i}"} for i in range(5)]
    sources = [Source("a", "repo/a", None, "train", "text"), Source("b", "repo/b", None, "train", "text")]
    repo_map = {"repo/a": ("text", rows_a), "repo/b": ("text", rows_b)}

    _install_fake_load_dataset(monkeypatch, repo_map)
    first = me.build_multi_en_source_mix(sources=sources, shuffle_seed=123, min_audio_duration_seconds=None)["text"]
    _install_fake_load_dataset(monkeypatch, repo_map)
    second = me.build_multi_en_source_mix(sources=sources, shuffle_seed=123, min_audio_duration_seconds=None)["text"]
    assert first == second


def test_min_audio_duration_filters_short_clips(monkeypatch):
    """``min_audio_duration_seconds`` drops sub-threshold clips via header-only sf.info."""
    short = _make_wav_bytes(duration_seconds=0.25)
    long = _make_wav_bytes(duration_seconds=1.5)
    rows = [
        {"audio": {"bytes": short, "path": None}, "text": "short clip"},
        {"audio": {"bytes": long, "path": None}, "text": "long clip"},
    ]
    sources = [Source("s", "repo/s", None, "train", "text")]
    _install_fake_load_dataset(monkeypatch, {"repo/s": ("text", rows)})

    mixed = me.build_multi_en_source_mix(sources=sources, shuffle_seed=None, min_audio_duration_seconds=1.0)
    assert mixed["text"] == ["LONG CLIP"]


def test_make_multi_en_asr_dataset_is_lazy_and_torchcodec_free(monkeypatch):
    """Construction decodes no audio; access triggers the soundfile decode; no torchcodec."""
    wav = _make_wav_bytes()
    rows = [
        {"audio": {"bytes": wav, "path": None}, "text": "hello world"},
        {"audio": {"bytes": wav, "path": None}, "text": "second row"},
    ]
    sources = [Source("s", "repo/s", None, "train", "text")]
    _install_fake_load_dataset(monkeypatch, {"repo/s": ("text", rows)})

    decode_calls = []
    original_decode = ds_mod._decode_audio_cell_to_mono_float32

    def _spy(audio_cell, target_sampling_rate):
        decode_calls.append(target_sampling_rate)
        return original_decode(audio_cell, target_sampling_rate)

    monkeypatch.setattr(ds_mod, "_decode_audio_cell_to_mono_float32", _spy)

    dataset = me.make_multi_en_asr_dataset(sources=sources, shuffle_seed=None, min_audio_duration_seconds=None)
    # No decode at construction.
    assert decode_calls == []
    assert len(dataset) == 2
    # First access decodes exactly once and yields the ASR conversation schema.
    row = dataset[0]
    assert list(row.keys()) == ["conversation"]
    assert len(decode_calls) == 1
    # The default user_prompt is present, so the user turn carries text + audio.
    conv = row["conversation"]
    assert [t["role"] for t in conv] == ["user", "assistant"]
    user_content = conv[0]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "audio"
    assert user_content[1]["audio"].dtype == np.float32

    # The multi_en module must not pull in torchcodec.
    assert "torchcodec" not in sys.modules
    assert "torchcodec" not in me.__dict__
    assert "torchcodec" not in ds_mod.__dict__


def test_default_sources_are_the_six_corpus_mix():
    """The default SOURCES list matches the prototype's six-corpus composition."""
    names = [s.name for s in me.SOURCES]
    assert names == ["ami_ihm", "earnings22", "voxpopuli_en", "gigaspeech_s", "spgispeech_s", "librispeech"]
    by_name = {s.name: s for s in me.SOURCES}
    # Data-mix hard numbers (prototype fidelity).
    assert by_name["voxpopuli_en"].limit == 4000
    assert by_name["gigaspeech_s"].trust_remote_code is True


def test_new_audio_modules_have_no_forbidden_audio_cast_literal():
    """AC-2 negative check: new audio modules must not contain the forbidden
    ``Audio(sampling_rate=...)`` / ``Audio(decode=True)`` callable pattern anywhere
    in their source text (only the non-decoding ``Audio(decode=False)`` is allowed)."""
    import re

    forbidden = re.compile(r"Audio\s*\(\s*sampling_rate|Audio\s*\(\s*decode\s*=\s*True")
    for mod in (me, ds_mod):
        with open(mod.__file__, encoding="utf-8") as fh:
            source = fh.read()
        assert not forbidden.search(source), f"forbidden Audio(...) cast pattern found in {mod.__file__}"


def test_coerce_source_accepts_source_and_mapping():
    """`_coerce_source` passes Source through and builds Source from a dict; rejects junk."""
    src = me.Source("a", "repo/a", None, "train", "text")
    assert me._coerce_source(src) is src

    coerced = me._coerce_source(
        {"name": "b", "repo": "repo/b", "config": "cfg", "split": "train", "text_col": "sentence", "limit": 7}
    )
    assert isinstance(coerced, me.Source)
    assert (coerced.name, coerced.repo, coerced.config, coerced.text_col, coerced.limit) == (
        "b",
        "repo/b",
        "cfg",
        "sentence",
        7,
    )

    import pytest

    with pytest.raises(ValueError, match="unknown Source field"):
        me._coerce_source({"name": "c", "repo": "r", "config": None, "split": "train", "text_col": "t", "bogus": 1})
    with pytest.raises(TypeError, match="must be a Source or mapping"):
        me._coerce_source(["not", "a", "mapping"])


def test_dict_sources_override_is_executable(monkeypatch):
    """A YAML/CLI-shaped `dataset.sources` override (list of dicts) builds the mix.

    The recipe config loader passes nested ``dataset.sources`` entries as plain dicts,
    so the builder must accept dict source specs, not just ``Source`` instances.
    """
    wav = _make_wav_bytes()
    # YAML/CLI shape: a list of plain dicts (e.g. to trim the mix to non-gated sources).
    dict_sources = [
        {"name": "ami_ihm", "repo": "edinburghcstr/ami", "config": "ihm", "split": "train[:2]", "text_col": "text"},
    ]
    _install_fake_load_dataset(
        monkeypatch,
        {
            "edinburghcstr/ami": (
                "text",
                [
                    {"audio": {"bytes": wav, "path": None}, "text": "first row"},
                    {"audio": {"bytes": wav, "path": None}, "text": "second row"},
                ],
            )
        },
    )

    mixed = me.build_multi_en_source_mix(sources=dict_sources, shuffle_seed=None, min_audio_duration_seconds=None)
    assert set(mixed.column_names) == {"audio", "text", "source"}
    assert sorted(mixed["source"]) == ["ami_ihm", "ami_ihm"]
    assert set(mixed["text"]) == {"FIRST ROW", "SECOND ROW"}

    # And end-to-end through the public builder (lazy transform attached).
    dataset = me.make_multi_en_asr_dataset(sources=dict_sources, shuffle_seed=None, min_audio_duration_seconds=None)
    assert list(dataset[0].keys()) == ["conversation"]
