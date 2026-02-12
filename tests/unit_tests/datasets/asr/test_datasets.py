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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datasets import Dataset

from nemo_automodel.components.datasets.asr.datasets import (
    make_common_voice_dataset,
    make_custom_asr_dataset,
    make_librispeech_dataset,
)


@pytest.fixture
def mock_librispeech_dataset():
    """Create a mock LibriSpeech dataset with audio and text fields."""
    data = {
        "audio": [
            {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(24000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(32000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(8000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
        ],
        "text": [
            "the quick brown fox",
            "jumps over the lazy dog",
            "hello world",
            "test sample one",
            "test sample two",
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_common_voice_dataset():
    """Create a mock Common Voice dataset with audio and sentence fields."""
    data = {
        "audio": [
            {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(32000).astype(np.float32), "sampling_rate": 16000},
            {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
        ],
        "sentence": [
            "this is a test",
            "another test sentence",
            "common voice example",
        ],
    }
    return Dataset.from_dict(data)


class TestMakeLibrispeechDataset:
    """Test make_librispeech_dataset behavior."""

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_librispeech_dataset_returns_correct_structure(
        self, mock_load_dataset, mock_librispeech_dataset
    ):
        """Verify LibriSpeech loader returns dataset with audio and text fields."""
        mock_load_dataset.return_value = mock_librispeech_dataset

        dataset = make_librispeech_dataset()

        # Should call load_dataset with correct parameters
        mock_load_dataset.assert_called_once_with(
            "librispeech_asr", "clean", split="train.100", streaming=False, trust_remote_code=True
        )

        # Returned dataset must have 'audio' and 'text' columns
        assert "audio" in dataset.column_names
        assert "text" in dataset.column_names

        # Audio column should contain dicts with 'array' and 'sampling_rate'
        audio_sample = dataset[0]["audio"]
        assert "array" in audio_sample
        assert "sampling_rate" in audio_sample

        # Text column should contain strings
        assert isinstance(dataset[0]["text"], str)

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_librispeech_dataset_with_custom_split(self, mock_load_dataset, mock_librispeech_dataset):
        """Verify custom split parameter is passed correctly."""
        mock_load_dataset.return_value = mock_librispeech_dataset

        dataset = make_librispeech_dataset(split="test")

        # Should use custom split in load_dataset call
        mock_load_dataset.assert_called_once_with(
            "librispeech_asr", "clean", split="test", streaming=False, trust_remote_code=True
        )

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_librispeech_dataset_limits_samples_when_specified(
        self, mock_load_dataset, mock_librispeech_dataset
    ):
        """Verify limit_dataset_samples parameter correctly limits dataset size."""
        mock_load_dataset.return_value = mock_librispeech_dataset

        limit = 3
        dataset = make_librispeech_dataset(limit_dataset_samples=limit)

        # If limit=3, returned dataset should have exactly 3 samples
        assert len(dataset) == limit

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_librispeech_dataset_handles_limit_larger_than_dataset(
        self, mock_load_dataset, mock_librispeech_dataset
    ):
        """Verify limit larger than dataset size doesn't cause errors."""
        mock_load_dataset.return_value = mock_librispeech_dataset
        original_len = len(mock_librispeech_dataset)

        # Request more samples than exist
        limit = original_len + 100
        dataset = make_librispeech_dataset(limit_dataset_samples=limit)

        # Should return full dataset, not raise error
        assert len(dataset) == original_len

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_librispeech_dataset_with_streaming(self, mock_load_dataset):
        """Verify streaming mode uses .take() for limiting samples."""
        # Create a mock streaming dataset
        mock_streaming_dataset = MagicMock()
        mock_streaming_dataset.take = MagicMock(return_value=mock_streaming_dataset)
        mock_load_dataset.return_value = mock_streaming_dataset

        limit = 10
        dataset = make_librispeech_dataset(streaming=True, limit_dataset_samples=limit)

        # Should call load_dataset with streaming=True
        mock_load_dataset.assert_called_once_with(
            "librispeech_asr", "clean", split="train.100", streaming=True, trust_remote_code=True
        )

        # Should use .take() for streaming datasets
        mock_streaming_dataset.take.assert_called_once_with(limit)


class TestMakeCommonVoiceDataset:
    """Test make_common_voice_dataset behavior."""

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_common_voice_dataset_returns_correct_structure(
        self, mock_load_dataset, mock_common_voice_dataset
    ):
        """Verify Common Voice loader returns dataset with audio and sentence fields."""
        mock_load_dataset.return_value = mock_common_voice_dataset

        dataset = make_common_voice_dataset()

        # Should call load_dataset with correct parameters
        mock_load_dataset.assert_called_once_with(
            "mozilla-foundation/common_voice_17_0",
            "en",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )

        # Returned dataset must have 'audio' and 'sentence' columns
        assert "audio" in dataset.column_names
        assert "sentence" in dataset.column_names

        # Audio column should contain dicts with 'array' and 'sampling_rate'
        audio_sample = dataset[0]["audio"]
        assert "array" in audio_sample
        assert "sampling_rate" in audio_sample

        # Sentence column should contain strings
        assert isinstance(dataset[0]["sentence"], str)

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_common_voice_dataset_with_custom_language(
        self, mock_load_dataset, mock_common_voice_dataset
    ):
        """Verify language parameter is passed correctly."""
        mock_load_dataset.return_value = mock_common_voice_dataset

        dataset = make_common_voice_dataset(language="es")

        # Should use custom language in load_dataset call
        args, kwargs = mock_load_dataset.call_args
        assert args[1] == "es"

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_common_voice_dataset_limits_samples(
        self, mock_load_dataset, mock_common_voice_dataset
    ):
        """Verify limit_dataset_samples parameter correctly limits dataset size."""
        mock_load_dataset.return_value = mock_common_voice_dataset

        limit = 2
        dataset = make_common_voice_dataset(limit_dataset_samples=limit)

        # Returned dataset should have exactly 2 samples
        assert len(dataset) == limit


class TestMakeCustomAsrDataset:
    """Test make_custom_asr_dataset behavior."""

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_custom_asr_dataset_returns_correct_structure(
        self, mock_load_dataset, mock_librispeech_dataset
    ):
        """Verify custom dataset loader returns dataset with audio and text fields."""
        mock_load_dataset.return_value = mock_librispeech_dataset

        dataset = make_custom_asr_dataset("my_custom_dataset")

        # Should call load_dataset with custom path
        mock_load_dataset.assert_called_once_with(
            "my_custom_dataset", split="train", streaming=False, trust_remote_code=True
        )

        # Should have audio and text columns
        assert "audio" in dataset.column_names
        assert "text" in dataset.column_names

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_custom_asr_dataset_renames_columns(self, mock_load_dataset):
        """Verify custom column names are renamed to standard 'audio' and 'text'."""
        # Create dataset with non-standard column names
        data = {
            "recording": [
                {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
                {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000},
            ],
            "transcription": ["hello world", "test sample"],
        }
        mock_dataset = Dataset.from_dict(data)
        mock_load_dataset.return_value = mock_dataset

        dataset = make_custom_asr_dataset(
            "custom_dataset", audio_column="recording", text_column="transcription"
        )

        # Should rename columns to standard names
        assert "audio" in dataset.column_names
        assert "text" in dataset.column_names

        # Old column names should not exist
        assert "recording" not in dataset.column_names
        assert "transcription" not in dataset.column_names

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_custom_asr_dataset_limits_samples(self, mock_load_dataset, mock_librispeech_dataset):
        """Verify limit_dataset_samples parameter correctly limits dataset size."""
        mock_load_dataset.return_value = mock_librispeech_dataset

        limit = 3
        dataset = make_custom_asr_dataset("custom_dataset", limit_dataset_samples=limit)

        # Returned dataset should have exactly 3 samples
        assert len(dataset) == limit

    @patch("nemo_automodel.components.datasets.asr.datasets.load_dataset")
    def test_make_custom_asr_dataset_with_streaming(self, mock_load_dataset):
        """Verify streaming mode uses .take() for limiting samples."""
        mock_streaming_dataset = MagicMock()
        mock_streaming_dataset.take = MagicMock(return_value=mock_streaming_dataset)
        mock_load_dataset.return_value = mock_streaming_dataset

        limit = 5
        dataset = make_custom_asr_dataset("custom_dataset", streaming=True, limit_dataset_samples=limit)

        # Should call load_dataset with streaming=True
        mock_load_dataset.assert_called_once_with(
            "custom_dataset", split="train", streaming=True, trust_remote_code=True
        )

        # Should use .take() for streaming datasets
        mock_streaming_dataset.take.assert_called_once_with(limit)
