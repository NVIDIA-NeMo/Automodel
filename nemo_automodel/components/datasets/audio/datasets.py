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
"""Audio / ASR dataset builders for NeMo AutoModel.

All audio decoding here is intentionally ``torchcodec``-free: HuggingFace audio
columns are cast with ``Audio(decode=False)`` and decoded on demand with
``soundfile`` (resampling via ``scipy.signal.resample_poly``). This matches the
inference-time pattern in ``result/decode_vllm.py`` and keeps the builders
usable inside environments that do not ship ``torchcodec``.
"""

import io
import logging

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

logger = logging.getLogger(__name__)


def make_cv17_dataset(path_or_dataset="ysdede/commonvoice_17_tr_fixed", split="train", **kwargs):
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    all_columns = dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
    dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|endoftext11|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    ret = [format(example) for example in dataset]
    return ret


def _decode_audio_cell_to_mono_float32(audio_cell, target_sampling_rate):
    """Decode a HuggingFace ``Audio(decode=False)`` cell to a 1-D float32 waveform.

    Avoids ``torchcodec`` by using ``soundfile`` for both byte and path branches,
    matching the pattern in ``result/decode_vllm.py``.

    Args:
        audio_cell: Dict with ``bytes`` and/or ``path`` keys, as returned by
            HuggingFace ``datasets`` when the column has ``Audio(decode=False)``.
        target_sampling_rate: Desired output sampling rate (Hz). If the source
            differs, the waveform is resampled via ``scipy.signal.resample_poly``.

    Returns:
        Tuple of ``(waveform_float32_mono, target_sampling_rate)``.

    Raises:
        ValueError: If both ``bytes`` and ``path`` are missing.
    """
    if not isinstance(audio_cell, dict):
        raise ValueError(f"audio cell must be a dict, got {type(audio_cell).__name__}: {audio_cell!r}")

    raw_bytes = audio_cell.get("bytes")
    raw_path = audio_cell.get("path")

    if raw_bytes is not None:
        waveform, source_sampling_rate = sf.read(io.BytesIO(raw_bytes))
    elif isinstance(raw_path, str) and raw_path:
        waveform, source_sampling_rate = sf.read(raw_path)
    else:
        raise ValueError(f"audio cell has neither 'bytes' nor 'path': {audio_cell!r}")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32, copy=False)

    if source_sampling_rate != target_sampling_rate:
        # Local import to avoid a hard scipy dependency at module load.
        from scipy.signal import resample_poly

        waveform = resample_poly(waveform, target_sampling_rate, source_sampling_rate).astype(np.float32, copy=False)

    return waveform, target_sampling_rate


def _build_asr_conversation(
    waveform,
    transcript,
    *,
    system_prompt,
    user_prompt,
    has_system,
    has_user_text,
):
    """Assemble the Qwen3-Omni ASR chat-template conversation for one sample."""
    conversation = []
    if has_system:
        conversation.append({"role": "system", "content": system_prompt})

    user_content = []
    if has_user_text:
        user_content.append({"type": "text", "text": user_prompt})
    user_content.append({"type": "audio", "audio": waveform})
    conversation.append({"role": "user", "content": user_content})

    conversation.append({"role": "assistant", "content": [{"type": "text", "text": transcript}]})
    return conversation


def _filter_nonempty_text(dataset, text_column):
    """Drop rows whose ``text_column`` is empty or whitespace (Arrow-level, no decode)."""
    return dataset.filter(
        lambda batch: [bool(t) and bool(t.strip()) for t in batch[text_column]],
        batched=True,
    )


def _filter_min_audio_duration(dataset, audio_column, min_audio_duration_seconds):
    """Drop rows shorter than ``min_audio_duration_seconds`` via header-only ``sf.info``.

    Uses ``soundfile.info`` (header read, no PCM decode) on the ``Audio(decode=False)``
    cell. The bytes branch wraps the payload in ``BytesIO``; the path branch passes the
    path directly. Cells that fail to probe are dropped.
    """

    def _duration_at_least(batch):
        keep = []
        for cell in batch[audio_column]:
            try:
                if cell.get("bytes"):
                    info = sf.info(io.BytesIO(cell["bytes"]))
                else:
                    info = sf.info(cell["path"])
                keep.append((info.frames / info.samplerate) >= min_audio_duration_seconds)
            except Exception:
                keep.append(False)
        return keep

    return dataset.filter(_duration_at_least, batched=True)


def _attach_asr_transform(
    dataset,
    *,
    audio_column="audio",
    text_column="text",
    sampling_rate=16000,
    system_prompt=None,
    user_prompt=None,
):
    """Attach the lazy decode-and-build-conversation transform to an ASR dataset.

    The dataset must already expose an ``audio_column`` cast with
    ``Audio(decode=False)`` and a (normalized, non-empty) ``text_column``. The
    returned dataset yields ``{"conversation": <chat-template list>}`` per item;
    audio is decoded with ``soundfile`` (mono mix + ``float32`` + optional
    ``scipy.signal.resample_poly``) only on access, so the transform runs inside
    dataloader workers rather than at construction time.

    The conversation shape follows the prompt-presence matrix:

    - both ``system_prompt`` and ``user_prompt`` set →
      ``system → user(text+audio) → assistant``
    - only ``system_prompt`` set → ``system → user(audio) → assistant``
    - only ``user_prompt`` set → ``user(text+audio) → assistant``  (no system turn)
    - neither set → ``user(audio) → assistant``

    Whitespace-only prompts are treated as absent.

    Args:
        dataset: A HuggingFace ``Dataset`` with ``audio_column`` (decode=False)
            and ``text_column``.
        audio_column: Name of the audio column.
        text_column: Name of the transcript column.
        sampling_rate: Target sampling rate in Hz.
        system_prompt: Optional system-turn instruction.
        user_prompt: Optional user-turn instruction placed before the audio.

    Returns:
        The dataset with a ``with_transform`` callback attached.
    """
    has_system = isinstance(system_prompt, str) and bool(system_prompt.strip())
    has_user_text = isinstance(user_prompt, str) and bool(user_prompt.strip())

    def _format(batch):
        # ``with_transform`` always passes a column-batched dict
        # ({col: [v1, v2, ...]}) regardless of whether the caller did
        # ``ds[i]`` or ``ds[i:j]``; HF unwraps the single-row case afterwards.
        audio_cells = batch[audio_column]
        transcripts = batch[text_column]
        conversations = []
        for audio_cell, transcript in zip(audio_cells, transcripts):
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"empty transcript in {text_column!r}; refusing to emit zero-label sample")
            waveform, _ = _decode_audio_cell_to_mono_float32(audio_cell, sampling_rate)
            conversations.append(
                _build_asr_conversation(
                    waveform,
                    transcript,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    has_system=has_system,
                    has_user_text=has_user_text,
                )
            )
        return {"conversation": conversations}

    return dataset.with_transform(_format)


def make_hf_audio_asr_dataset(
    path_or_dataset,
    split="train",
    name=None,
    sampling_rate=16000,
    system_prompt=None,
    user_prompt=None,
    audio_column="audio",
    text_column="text",
    drop_empty_text=True,
    min_audio_duration_seconds=None,
    **load_kwargs,
):
    """Lazy HuggingFace audio→text dataset builder for Qwen-Omni ASR fine-tuning.

    Loads any HuggingFace ASR dataset that exposes an audio column (``Audio``
    feature with ``bytes`` and/or ``path`` populated after
    ``cast_column(decode=False)``) and a transcript column, and yields the
    Qwen-Omni chat-template conversation expected by the ASR collate functions.
    **No audio is decoded at construction time** — both the soundfile decode
    (mono mix + ``float32`` cast + optional ``scipy.signal.resample_poly``) and
    the conversation assembly run inside a HuggingFace ``with_transform``
    callback, so the only fixed startup cost is the Arrow-level metadata read of
    the parquet shards (and the on-demand download of those shards if they are
    not already in the HF cache). Empty-transcript filtering happens via
    ``dataset.filter`` against the text column only — also Arrow-level — so audio
    bytes are never materialized at startup.

    Defaults are tuned for the common case (``audio`` / ``text`` columns,
    16 kHz, no system turn). Datasets that diverge can override per-field via
    YAML; see :file:`docs/guides/audio/qwen3-omni-asr.md` for an override table.

    Args:
        path_or_dataset: HuggingFace dataset id or local path.
        split: Dataset split to load (e.g. ``"train"``, ``"train[:5000]"``).
        name: Optional dataset configuration / subset. Forwarded to
            ``datasets.load_dataset(path, name=name, ...)``. Required by some
            datasets (e.g. ``edinburghcstr/ami`` needs ``"ihm"`` or ``"sdm"``;
            CommonVoice needs the language code).
        sampling_rate: Target sampling rate in Hz. Audio is resampled inside
            the lazy transform if the source rate differs.
        system_prompt: Instruction placed in a ``system`` turn. Default
            ``None`` skips the system turn entirely; pass a string to emit one.
        user_prompt: Instruction prepended to the audio inside the user turn.
            Pass ``None`` to emit a user turn with only the audio item.
        audio_column: Name of the audio column in the source dataset (default
            ``"audio"`` — works for AMI / LibriSpeech / GigaSpeech /
            WenetSpeech / CommonVoice).
        text_column: Name of the transcript column (default ``"text"`` —
            works for AMI / LibriSpeech / GigaSpeech / WenetSpeech; override
            to ``"sentence"`` for CommonVoice).
        drop_empty_text: If True, samples whose transcript is empty or
            whitespace are dropped via ``dataset.filter`` (Arrow-level, no
            audio decode). If False, an empty transcript triggers a
            ``ValueError`` inside the transform at access time.
        min_audio_duration_seconds: Optional minimum audio duration. Samples
            shorter than this threshold are dropped via ``dataset.filter``
            using ``soundfile.info`` (header-only read, no full decode). The
            HF Qwen-Omni Whisper feature extractor has a known off-by-one
            between ``input_features`` and ``feature_attention_mask`` for
            sub-second clips (~0.27 s manifests as a 27-vs-26 frame
            mismatch); set this to ``1.0`` for AMI / CommonVoice-style
            corpora that contain very short utterances.
        **load_kwargs: Forwarded to ``datasets.load_dataset`` (e.g.
            ``trust_remote_code=True``).

    Returns:
        A HuggingFace ``Dataset`` whose elements are
        ``{"conversation": <chat-template list>}`` and whose audio is decoded
        on demand via dataloader workers.

    Raises:
        ValueError: When ``audio_column`` or ``text_column`` is missing, when
            an audio cell has neither ``bytes`` nor ``path``, or when
            ``drop_empty_text=False`` and a transcript is empty.
    """
    dataset = load_dataset(path_or_dataset, name=name, split=split, **load_kwargs)

    if audio_column not in dataset.column_names:
        raise ValueError(f"audio_column={audio_column!r} not found in dataset columns: {dataset.column_names}")
    if text_column not in dataset.column_names:
        raise ValueError(f"text_column={text_column!r} not found in dataset columns: {dataset.column_names}")

    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    if drop_empty_text:
        # Arrow-level filter on the text column only; no audio decode runs.
        dataset = _filter_nonempty_text(dataset, text_column)

    if min_audio_duration_seconds is not None:
        dataset = _filter_min_audio_duration(dataset, audio_column, min_audio_duration_seconds)

    return _attach_asr_transform(
        dataset,
        audio_column=audio_column,
        text_column=text_column,
        sampling_rate=sampling_rate,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
