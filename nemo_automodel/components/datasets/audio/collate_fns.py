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
"""Collate functions for Qwen-Omni ASR fine-tuning (``torchcodec``-free).

These collates assume audio waveforms are already attached to each conversation
as 1-D ``np.ndarray`` items (see
:func:`nemo_automodel.components.datasets.audio.datasets.make_hf_audio_asr_dataset`),
so they feed the processor's ``audio=`` kwarg directly without going through
``qwen_omni_utils`` / ``torchcodec``. Label masking is delegated to the shared
marker-based :func:`nemo_automodel.components.datasets.label_utils.build_labels_from_template`.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from nemo_automodel.components.datasets.label_utils import build_labels_from_template


def _extract_audios_from_conversation(conversation: Sequence[Dict[str, Any]]) -> List[Any]:
    """Walk a Qwen-Omni-style conversation and collect audio payloads in order.

    The returned list contains the raw audio objects (typically 1-D ``np.ndarray``
    waveforms) attached to ``{"type": "audio", "audio": ...}`` items in any
    message's content list. Used by :func:`qwen3_omni_asr_collate_fn` to feed the
    processor's ``audio=`` kwarg without going through ``qwen_omni_utils``.
    """
    audios: List[Any] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "audio":
                audios.append(item.get("audio"))
    return audios


def _validate_and_coerce_audio_payload(payload: Any, sample_index: int) -> np.ndarray:
    """Coerce an audio payload to a 1-D ``float32`` ``np.ndarray`` or raise.

    The single rule:
      - Convert any numeric ``np.ndarray`` / ``torch.Tensor`` to ``np.float32``.
      - The result must be exactly 1-D after conversion (mono waveform).
      - Anything else raises ``ValueError`` naming the sample index, observed
        shape, and observed dtype so the caller can pinpoint the bad sample.

    Args:
        payload: Audio object pulled from a conversation content item.
        sample_index: Index of the offending sample within the batch (for error
            messages).

    Returns:
        A 1-D ``np.float32`` ``np.ndarray``.

    Raises:
        ValueError: When the payload is not a numeric array or is not 1-D.
    """
    if hasattr(payload, "detach") and hasattr(payload, "cpu") and hasattr(payload, "numpy"):
        # torch.Tensor or similar; move to CPU before NumPy view.
        payload = payload.detach().cpu().numpy()

    if not isinstance(payload, np.ndarray):
        raise ValueError(
            f"sample[{sample_index}] audio payload must be an np.ndarray or torch.Tensor; "
            f"got type={type(payload).__name__}"
        )

    if not np.issubdtype(payload.dtype, np.number):
        raise ValueError(
            f"sample[{sample_index}] audio payload must have a numeric dtype; "
            f"got shape={payload.shape} dtype={payload.dtype}"
        )

    if payload.dtype != np.float32:
        payload = payload.astype(np.float32, copy=False)

    if payload.ndim != 1:
        raise ValueError(
            f"sample[{sample_index}] audio payload must be 1-D (mono waveform); "
            f"got shape={payload.shape} dtype={payload.dtype}"
        )

    return payload


def _conversation_ends_with_assistant_text(conversation: Sequence[Dict[str, Any]]) -> bool:
    """Return True iff the last turn is an ``assistant`` turn with non-empty text content."""
    if not conversation:
        return False
    last = conversation[-1]
    if last.get("role") != "assistant":
        return False
    content = last.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return True
    return False


def qwen3_omni_asr_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
) -> Dict[str, torch.Tensor]:
    """Collate Qwen3-Omni ASR conversations into model inputs without ``qwen_omni_utils``.

    Unlike ``qwen3_omni_collate_fn`` (in ``vlm.collate_fns``), this collate is
    intended for environments that lack ``qwen_omni_utils`` and ``torchcodec``.
    It assumes audio waveforms are already attached to the conversation as 1-D
    ``np.ndarray`` items of the form ``{"type": "audio", "audio": waveform}`` (see
    :func:`nemo_automodel.components.datasets.audio.datasets.make_hf_audio_asr_dataset`)
    and passes them directly to the processor's ``audio=`` kwarg, which routes to
    the bundled ``WhisperFeatureExtractor``.

    Label masking is delegated to :func:`build_labels_from_template`, which uses
    the marker-based fast path that already supports ``Qwen3OmniMoeProcessor``
    via ``_IMSTART_TEMPLATE_PROCESSORS``. The collate produces pre-shifted labels
    (``labels[:, 1:]``) and slices same-shape tensors to ``[:, :-1]`` so the
    downstream loss (``MaskedCrossEntropy``/``FusedLinearCrossEntropy``) consumes
    them without a second internal shift.

    Args:
        examples: Iterable of dicts each containing a ``conversation`` key, where
            the last turn MUST be an ``assistant`` turn with non-empty text.
        processor: A ``Qwen3OmniMoeProcessor`` instance (or compatible mock).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``input_features``,
        ``feature_attention_mask``, and ``labels`` plus any other tensors the
        processor returns, all aligned along the batch dimension.

    Raises:
        ValueError: If any conversation lacks a non-empty assistant turn at the
            end (the marker-based labeler would otherwise produce all-``-100``
            labels and a NaN loss).
    """
    conversations = [example["conversation"] for example in examples]

    for idx, conv in enumerate(conversations):
        if not _conversation_ends_with_assistant_text(conv):
            raise ValueError(
                f"example[{idx}].conversation must end with an assistant turn containing non-empty text; got: {conv!r}"
            )

    texts = [processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False) for conv in conversations]

    all_audios: List[Any] = []
    for idx, conv in enumerate(conversations):
        sample_audios = _extract_audios_from_conversation(conv)
        for payload in sample_audios:
            all_audios.append(_validate_and_coerce_audio_payload(payload, sample_index=idx))

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        # Match qwen3_omni_collate_fn and the recipe's token-accounting helpers
        # (count_tail_padding only strips right-tail padding); transformers 5.5.0
        # defaults this processor's text padding to "left".
        "padding_side": "right",
    }
    if all_audios:
        processor_kwargs["audio"] = all_audios

    batch = processor(**processor_kwargs)

    labels = build_labels_from_template(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    return batch


def qwen2_5_omni_asr_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
) -> Dict[str, torch.Tensor]:
    """Collate Qwen2.5-Omni ASR conversations.

    Thin alias over :func:`qwen3_omni_asr_collate_fn`: the body is processor-
    agnostic (it only depends on the processor exposing ``apply_chat_template``
    and the ``audio=`` kwarg, both of which ``Qwen2_5OmniProcessor`` provides),
    so the entire Qwen3-Omni-ASR path works unchanged here. We expose a
    separate symbol so YAML configs can pick the right collate via
    ``_target_`` without users having to know about the Qwen3-Omni name.
    """
    return qwen3_omni_asr_collate_fn(examples, processor)
