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
"""Multi-source English ASR mix builder for Qwen-Omni fine-tuning.

Mixes several public HuggingFace speech corpora into a single
``{audio, text, source}`` training set, normalizing every transcript to a common
uppercase / punctuation-free style. Audio is never decoded at construction time:
each source's audio column is cast with ``Audio(decode=False)`` and decoded
lazily with ``soundfile`` inside the shared
:func:`nemo_automodel.components.datasets.audio.datasets._attach_asr_transform`
(so no ``torchcodec`` dependency is pulled in).

This is the first-class port of the ``result/data/build_train_mix.py`` prototype.
The key difference from the prototype: the prototype resampled with
``Audio(sampling_rate=...)`` (HuggingFace default decoding → ``torchcodec``),
whereas this builder keeps ``Audio(decode=False)`` and lets resampling happen in
the soundfile decode path inside the lazy transform.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from datasets import Audio, concatenate_datasets, load_dataset

from nemo_automodel.components.datasets.audio.datasets import (
    _attach_asr_transform,
    _filter_min_audio_duration,
    _filter_nonempty_text,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLING_RATE = 16_000

# Default user-turn instruction for the multi-domain English mix. Kept generic
# (not "meeting"-specific) because the mix spans meetings, earnings calls,
# parliamentary speech, audiobooks, and web audio.
DEFAULT_USER_PROMPT = (
    "Transcribe the following English audio verbatim. Output only the raw "
    "transcript text with no leading or trailing commentary."
)

# --- text normalisation: collapse every source to the AMI/LibriSpeech style
# (UPPERCASE, no punctuation, apostrophes kept inside words). This also strips
# gigaspeech's bracketed tags (<COMMA>, <PERIOD>, <SIL>, <MUSIC>, <NOISE>, ...).
_TAG_RE = re.compile(r"<[^>]*>")  # bracketed tags -> space
_PUNCT_RE = re.compile(r"[^\w\s']")  # drop anything but word chars / space / apostrophe
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalise a transcript to UPPERCASE, punctuation-free, apostrophes kept.

    Args:
        text: Raw transcript (may be ``None``).

    Returns:
        The normalised transcript. Digits and intra-word apostrophes are kept;
        bracketed tags (e.g. ``<COMMA>``, ``<MUSIC>``) and punctuation are
        dropped; underscores become spaces; whitespace is collapsed.
    """
    s = (text or "").replace("’", "'")  # curly -> straight apostrophe
    s = _TAG_RE.sub(" ", s)  # remove <COMMA>/<PERIOD>/<SIL>/...
    s = s.upper()
    s = _PUNCT_RE.sub(" ", s)  # strip punctuation (keeps digits + ')
    s = s.replace("_", " ")
    return _WS_RE.sub(" ", s).strip()


@dataclass(frozen=True)
class Source:
    """Specification for one corpus in the English ASR mix."""

    name: str  # label written into the "source" column
    repo: str
    config: Optional[str]
    split: str
    text_col: str  # column holding the transcript in this repo
    limit: Optional[int] = None  # keep only the first N examples
    trust_remote_code: bool = False
    known_count: Optional[int] = None  # from metadata, for reporting only


# Composition (matches result/data/build_train_mix.py exactly). gigaspeech and
# spgispeech are gated on the Hub: accept their terms (and, for gigaspeech, allow
# trust_remote_code) before launching, or pass a trimmed ``sources`` list.
SOURCES: List[Source] = [
    Source("ami_ihm", "edinburghcstr/ami", "ihm", "train", "text", known_count=108_502),
    Source("earnings22", "sanchit-gandhi/earnings22_split", None, "train", "sentence", known_count=52_006),
    Source("voxpopuli_en", "facebook/voxpopuli", "en", "train", "normalized_text", limit=4_000, known_count=4_000),
    Source(
        "gigaspeech_s",
        "speechcolab/gigaspeech",
        "s",
        "train",
        "text",
        trust_remote_code=True,
        known_count=230_068,
    ),
    Source("spgispeech_s", "kensho/spgispeech", "S", "train", "transcript", known_count=77_073),
    Source("librispeech", "openslr/librispeech_asr", "clean", "train.100", "text", known_count=28_539),
]


def _load_and_normalize_source(src: Source, *, audio_column: str, text_column: str):
    """Load one source and normalise it to ``{audio (decode=False), text, source}``.

    No audio is decoded: the audio column is cast with ``Audio(decode=False)`` and
    only the transcript column is touched (text-only Arrow ``map``/``filter``).

    Raises:
        ValueError: If the audio column or the source's text column is missing.
    """
    args = (src.repo, src.config) if src.config else (src.repo,)
    ds = load_dataset(*args, split=src.split, trust_remote_code=src.trust_remote_code)

    if audio_column not in ds.column_names:
        raise ValueError(f"audio_column={audio_column!r} not found in {src.name} columns: {ds.column_names}")
    if src.text_col not in ds.column_names:
        raise ValueError(f"text_column={src.text_col!r} not found in {src.name} columns: {ds.column_names}")

    if src.limit is not None:
        ds = ds.select(range(min(src.limit, len(ds))))

    # No torchcodec: keep raw bytes/path; resampling happens in the lazy decode.
    ds = ds.cast_column(audio_column, Audio(decode=False))

    keep = {audio_column, src.text_col}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    if src.text_col != text_column:
        ds = ds.rename_column(src.text_col, text_column)

    # Text-only normalisation (no audio touched).
    ds = ds.map(lambda batch: {text_column: [normalize_text(t) for t in batch[text_column]]}, batched=True)
    # Drop rows that normalised to an empty transcript (e.g. tag-only "<MUSIC>"),
    # which would otherwise raise inside the lazy transform in a dataloader worker.
    ds = _filter_nonempty_text(ds, text_column)

    ds = ds.add_column("source", [src.name] * len(ds))
    return ds


def build_multi_en_source_mix(
    *,
    sources: Optional[List[Source]] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    shuffle_seed: Optional[int] = 42,
    min_audio_duration_seconds: Optional[float] = 1.0,
):
    """Build the concatenated ``{audio, text, source}`` mix (before the ASR transform).

    Exposed separately from :func:`make_multi_en_asr_dataset` so the mix
    composition (source labels, normalisation, filtering) can be inspected/tested
    without the lazy conversation transform that hides all columns but
    ``conversation``.

    Args:
        sources: Source specs to mix. Defaults to the full six-source
            :data:`SOURCES`. Pass a trimmed list for local/smoke runs.
        audio_column: Name of the audio column to standardise on.
        text_column: Name of the transcript column to standardise on.
        shuffle_seed: Global shuffle seed so consecutive examples mix sources.
            ``None`` disables shuffling (single-source blocks).
        min_audio_duration_seconds: Drop clips shorter than this via header-only
            ``soundfile.info`` (no PCM decode). ``None`` disables the filter.

    Returns:
        A map-style HuggingFace ``Dataset`` with columns
        ``{audio (decode=False), text, source}``.
    """
    sources = list(sources) if sources is not None else list(SOURCES)
    parts = [_load_and_normalize_source(s, audio_column=audio_column, text_column=text_column) for s in sources]

    mixed = concatenate_datasets(parts)
    if shuffle_seed is not None:
        mixed = mixed.shuffle(seed=shuffle_seed)

    if min_audio_duration_seconds is not None:
        mixed = _filter_min_audio_duration(mixed, audio_column, min_audio_duration_seconds)

    return mixed


def make_multi_en_asr_dataset(
    *,
    sampling_rate: int = TARGET_SAMPLING_RATE,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = DEFAULT_USER_PROMPT,
    min_audio_duration_seconds: Optional[float] = 1.0,
    shuffle_seed: Optional[int] = 42,
    sources: Optional[List[Source]] = None,
    audio_column: str = "audio",
    text_column: str = "text",
):
    """Build the multi-source English ASR training dataset for Qwen-Omni.

    Mixes the six public corpora in :data:`SOURCES` (or a caller-supplied subset),
    normalises every transcript with :func:`normalize_text`, and attaches the
    shared lazy ASR transform so audio is decoded with ``soundfile`` (and
    resampled to ``sampling_rate``) only on item access — never at construction
    time and never via ``torchcodec``.

    Args:
        sampling_rate: Target sampling rate in Hz for the decoded waveform.
        system_prompt: Optional system-turn instruction. ``None`` omits it.
        user_prompt: User-turn instruction placed before the audio. Defaults to
            a generic English ASR instruction.
        min_audio_duration_seconds: Drop clips shorter than this (header-only
            probe). Defaults to ``1.0`` to dodge the Qwen-Omni Whisper
            feature-extractor sub-second off-by-one.
        shuffle_seed: Global shuffle seed; ``None`` disables shuffling.
        sources: Optional subset of :data:`SOURCES` (e.g. to skip gated corpora
            for a local smoke run). Defaults to all six.
        audio_column: Name of the audio column to standardise on.
        text_column: Name of the transcript column to standardise on.

    Returns:
        A HuggingFace ``Dataset`` whose elements are
        ``{"conversation": <chat-template list>}``, with audio decoded on demand
        in dataloader workers.
    """
    mixed = build_multi_en_source_mix(
        sources=sources,
        audio_column=audio_column,
        text_column=text_column,
        shuffle_seed=shuffle_seed,
        min_audio_duration_seconds=min_audio_duration_seconds,
    )
    return _attach_asr_transform(
        mixed,
        audio_column=audio_column,
        text_column=text_column,
        sampling_rate=sampling_rate,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
