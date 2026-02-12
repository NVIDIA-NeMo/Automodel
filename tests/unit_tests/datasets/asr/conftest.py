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

import numpy as np
import pytest
import torch


class DummyWhisperTokenizer:
    """A minimal, working tokenizer for Whisper that implements the required contract.

    Similar to DummyTokenizer in test_utils.py - this is a functional implementation,
    not a mock. Each character is converted to an integer id; special tokens are added.
    """

    pad_token_id = 50256
    eos_token_id = 50257
    bos_token_id = 50258

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert special tokens to their IDs."""
        if token == "<|startoftranscript|>":
            return self.bos_token_id
        return self.pad_token_id

    def _encode_single(self, text: str) -> list[int]:
        """Encode a single text string to token IDs.

        Uses character-based encoding (similar to DummyTokenizer pattern).
        Normal chars start at 10 for readability.
        """
        # Start with BOS token, encode chars, end with EOS
        return [self.bos_token_id] + [ord(c) % 100 + 10 for c in text] + [self.eos_token_id]

    def __call__(
        self,
        text: list[str] | str,
        return_tensors: str | None = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ):
        """Tokenize text (single string or list of strings).

        Behavior: Returns object with .input_ids attribute containing token tensors.
        """
        if isinstance(text, str):
            text = [text]

        # Encode each text
        input_ids_list = [self._encode_single(t) for t in text]

        # Apply max_length truncation if specified
        if max_length is not None and truncation:
            input_ids_list = [ids[:max_length] for ids in input_ids_list]

        # Apply padding if specified
        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            input_ids_list = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]

        # Convert to tensor if requested
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        else:
            input_ids = input_ids_list

        # Return namespace-like object with input_ids attribute
        class TokenizerOutput:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return TokenizerOutput(input_ids)


class DummyWhisperFeatureExtractor:
    """A minimal, working feature extractor for Whisper.

    Behavior: Converts audio arrays to mel spectrogram tensors with correct shape.
    Uses simplified processing (doesn't do real mel spectrogram computation).
    """

    def __call__(
        self,
        audios: list[np.ndarray],
        sampling_rate: int,
        return_tensors: str | None = None,
    ):
        """Extract mel spectrogram features from audio arrays.

        Behavior:
        - Input: list of numpy arrays (audio waveforms)
        - Output: .input_features with shape (batch_size, 80, 3000)
        """
        batch_size = len(audios)

        # Whisper produces 80-channel mel spectrograms with 3000 time steps (30 seconds at 100 fps)
        # We create a simplified version with random values but correct shape
        mel_features = torch.randn(batch_size, 80, 3000, dtype=torch.float32)

        # Return namespace-like object with input_features attribute
        class FeatureExtractorOutput:
            def __init__(self, input_features):
                self.input_features = input_features

        return FeatureExtractorOutput(mel_features)


class DummyWhisperProcessor:
    """A minimal, working WhisperProcessor that implements the required contract.

    Following the DummyTokenizer pattern - this is a functional implementation,
    not a mock. It actually processes audio and text (in a simplified way).
    """

    def __init__(self):
        self.feature_extractor = DummyWhisperFeatureExtractor()
        self.tokenizer = DummyWhisperTokenizer()


class DummyParakeetProcessor:
    """A minimal, working ParakeetProcessor for CTC models.

    Behavior: Processes audio and text together, returns dict with input_features,
    attention_mask, and labels.
    """

    def __call__(
        self,
        audio: list[np.ndarray],
        text: list[str] | None = None,
        sampling_rate: int = 16000,
        return_tensors: str | None = None,
        return_attention_mask: bool = True,
        **kwargs,
    ) -> dict:
        """Process audio and text for CTC training.

        Behavior:
        - Input: list of audio arrays and texts
        - Output: dict with input_features, attention_mask, labels
        """
        batch_size = len(audio)

        # Parakeet uses mel spectrograms (simplified: 80 features, 100 time steps)
        # Real processor would compute actual features and variable lengths
        input_features = torch.randn(batch_size, 80, 100, dtype=torch.float32)

        # Attention mask: 1 for valid positions, 0 for padding
        # For simplicity, we create full masks (no padding)
        attention_mask = torch.ones(batch_size, 100, dtype=torch.long)

        # Tokenize text (simple character-based encoding)
        if text is not None:
            # Simple encoding: each char to an ID
            labels_list = []
            for t in text:
                # Character-based encoding (10-109 range)
                label_ids = [ord(c) % 100 + 10 for c in t]
                labels_list.append(torch.tensor(label_ids, dtype=torch.long))

            # Pad labels to same length
            max_label_len = max(len(l) for l in labels_list)
            labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
            for i, label in enumerate(labels_list):
                labels[i, : len(label)] = label
        else:
            labels = None

        result = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }

        if labels is not None:
            result["labels"] = labels

        return result


@pytest.fixture
def dummy_whisper_processor():
    """Return a fresh DummyWhisperProcessor for each test."""
    return DummyWhisperProcessor()


@pytest.fixture
def dummy_parakeet_processor():
    """Return a fresh DummyParakeetProcessor for each test."""
    return DummyParakeetProcessor()


@pytest.fixture
def dummy_audio_samples():
    """Create a small batch of dummy audio samples for testing.

    Returns:
        List of dicts with 'audio' and 'text' fields, mimicking HuggingFace dataset format.
    """
    return [
        {
            "audio": {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
            "text": "hello world",
        },
        {
            "audio": {"array": np.random.randn(32000).astype(np.float32), "sampling_rate": 16000},
            "text": "the quick brown fox",
        },
        {
            "audio": {"array": np.random.randn(8000).astype(np.float32), "sampling_rate": 16000},
            "text": "test",
        },
    ]


@pytest.fixture
def dummy_audio_samples_with_sentence_field():
    """Create audio samples with 'sentence' field instead of 'text' (Common Voice format)."""
    return [
        {
            "audio": {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
            "sentence": "this is a test",
        },
        {
            "audio": {"array": np.random.randn(24000).astype(np.float32), "sampling_rate": 16000},
            "sentence": "another sentence",
        },
    ]
