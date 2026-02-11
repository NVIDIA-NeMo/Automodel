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

from typing import Optional

from datasets import load_dataset


def make_common_voice_dataset(
    path_or_dataset: str = "mozilla-foundation/common_voice_17_0",
    language: str = "en",
    split: str = "train",
    streaming: bool = False,
    limit_dataset_samples: Optional[int] = None,
):
    """Load Common Voice dataset for ASR training.

    Common Voice is a multilingual speech corpus with 100+ languages. Each sample
    contains audio data and corresponding transcription text.

    Note:
        As of October 2025, Mozilla Common Voice datasets are no longer hosted on
        HuggingFace. Download the dataset from Mozilla Data Collective
        (https://datacollective.mozillafoundation.org) and provide the local path.
        Alternatively, use LibriSpeech which is readily available via HuggingFace.

    Args:
        path_or_dataset: HuggingFace dataset ID or local path
        language: Language code (e.g., 'en', 'es', 'fr')
        split: Dataset split ('train', 'validation', 'test')
        streaming: Stream dataset instead of downloading entirely
        limit_dataset_samples: Limit to first N samples for debugging

    Returns:
        HuggingFace Dataset with 'audio' and 'sentence' fields
    """
    dataset = load_dataset(path_or_dataset, language, split=split, streaming=streaming, trust_remote_code=True)

    if limit_dataset_samples:
        if streaming:
            dataset = dataset.take(limit_dataset_samples)
        else:
            dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset


def make_librispeech_dataset(
    path_or_dataset: str = "librispeech_asr",
    split: str = "train.100",
    streaming: bool = False,
    limit_dataset_samples: Optional[int] = None,
):
    """Load LibriSpeech dataset for ASR training.

    LibriSpeech is a 1000-hour English speech corpus derived from audiobooks.
    It provides high-quality recordings with accurate transcriptions.

    Args:
        path_or_dataset: HuggingFace dataset ID or local path
        split: Dataset split (e.g., 'train.100', 'train.clean.360', 'test')
        streaming: Stream dataset instead of downloading entirely
        limit_dataset_samples: Limit to first N samples for debugging

    Returns:
        HuggingFace Dataset with 'audio' and 'text' fields
    """
    dataset = load_dataset(path_or_dataset, "clean", split=split, streaming=streaming, trust_remote_code=True)

    if limit_dataset_samples:
        if streaming:
            dataset = dataset.take(limit_dataset_samples)
        else:
            dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset


def make_custom_asr_dataset(
    path_or_dataset: str,
    split: str = "train",
    audio_column: str = "audio",
    text_column: str = "text",
    streaming: bool = False,
    limit_dataset_samples: Optional[int] = None,
):
    """Load custom ASR dataset from HuggingFace or local files.

    Generic loader for any HuggingFace audio dataset that follows the standard
    structure with audio and text columns. Supports JSON, JSONL, Parquet, etc.

    Args:
        path_or_dataset: HuggingFace dataset ID or local path to dataset files
        split: Dataset split name
        audio_column: Name of column containing audio data
        text_column: Name of column containing transcription text
        streaming: Stream dataset instead of downloading entirely
        limit_dataset_samples: Limit to first N samples for debugging

    Returns:
        HuggingFace Dataset with audio and text fields
    """
    dataset = load_dataset(path_or_dataset, split=split, streaming=streaming, trust_remote_code=True)

    if audio_column != "audio" or text_column != "text":
        dataset = dataset.rename_column(audio_column, "audio")
        dataset = dataset.rename_column(text_column, "text")

    if limit_dataset_samples:
        if streaming:
            dataset = dataset.take(limit_dataset_samples)
        else:
            dataset = dataset.select(range(min(limit_dataset_samples, len(dataset))))

    return dataset
