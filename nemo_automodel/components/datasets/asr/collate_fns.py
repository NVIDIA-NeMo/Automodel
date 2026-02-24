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

import logging
from typing import Any, Dict, Sequence

import torch

logger = logging.getLogger(__name__)


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """
    Shift input ids one token to the right for decoder input.

    This is used to create decoder_input_ids from labels for teacher forcing.
    The first token becomes decoder_start_token_id, and the rest are shifted.

    Args:
        input_ids: Token IDs to shift (labels)
        pad_token_id: ID of padding token
        decoder_start_token_id: ID of the decoder start token

    Returns:
        Shifted token IDs suitable for decoder input
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    # Replace -100 with pad_token_id to handle HuggingFace's standard label masking convention
    # (-100 is used to ignore certain tokens in loss computation, but decoder inputs need valid token IDs)
    if -100 in shifted_input_ids:
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def whisper_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: int = 448,
) -> Dict[str, torch.Tensor]:
    """Collate function for Whisper ASR models.

    Processes raw audio samples into mel spectrograms and tokenizes transcriptions.
    Whisper expects audio at 16kHz sampling rate and generates 80-channel mel spectrograms.

    Args:
        examples: Batch of samples with 'audio' and 'text' or 'sentence' fields
        processor: WhisperProcessor for audio and text processing
        max_length: Maximum length for text sequences (Whisper default: 448 tokens)

    Returns:
        Batch dict with:
            - input_features: (batch, 80, 3000) mel spectrograms
            - decoder_input_ids: (batch, text_seq_len) shifted labels for decoder
            - labels: (batch, text_seq_len) tokenized transcriptions for loss
    """
    audios = [ex["audio"]["array"] for ex in examples]
    text_key = "sentence" if "sentence" in examples[0] else "text"
    texts = [ex[text_key] for ex in examples]

    audio_features = processor.feature_extractor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
    )

    text_encodings = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = text_encodings.input_ids

    # Create decoder_input_ids by shifting labels right for teacher forcing
    # Whisper uses <|startoftranscript|> as the decoder start token
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        # Whisper uses <|endoftext|> as pad token if not explicitly set
        pad_token_id = processor.tokenizer.eos_token_id

    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
    )

    # Combine into single batch dict
    # Note: input_features will be converted to model dtype by the model itself
    # but we return float32 here as the default precision
    batch = {
        "input_features": audio_features.input_features,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }

    return batch


def parakeet_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Collate function for Parakeet CTC ASR models.

    Processes raw audio samples into mel spectrograms and tokenizes transcriptions
    for CTC training.

    Args:
        examples: Batch of samples with 'audio' and 'text' or 'sentence' fields
        processor: ParakeetProcessor for audio and text processing
        max_length: Maximum length for audio in seconds (optional)

    Returns:
        Batch dict with:
            - input_features: (batch, feature_dim, time) mel spectrograms
            - attention_mask: (batch, time) attention mask for variable length sequences
            - labels: (batch, text_seq_len) tokenized transcriptions for CTC loss
    """
    # Extract audio arrays and text
    audios = [ex["audio"]["array"] for ex in examples]
    text_key = "sentence" if "sentence" in examples[0] else "text"
    texts = [ex[text_key] for ex in examples]

    # Process audio to mel spectrograms
    processor_kwargs = {
        "sampling_rate": 16000,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }

    if max_length is not None:
        processor_kwargs["padding"] = "max_length"
        processor_kwargs["max_length"] = max_length * 16000  # Convert seconds to samples

    # Process audio and text together (processor handles both)
    batch = processor(audio=audios, text=texts, **processor_kwargs)

    return batch


COLLATE_FNS = {
    "WhisperProcessor": whisper_collate_fn,
    "ParakeetProcessor": parakeet_collate_fn,
    "default": whisper_collate_fn,
}
