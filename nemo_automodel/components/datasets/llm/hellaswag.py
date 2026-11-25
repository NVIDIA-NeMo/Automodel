# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from datasets import load_dataset

from nemo_automodel.components.datasets.llm.sample_limit import resolve_sample_limit

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from nemo_automodel.components.datasets.utils import SFTSingleTurnPreprocessor


@dataclass
class HellaSwagConfig:
    """Construction-time configuration for :class:`HellaSwag`."""

    accepts_tokenizer: ClassVar[bool] = True

    path_or_dataset: str
    """Path to the dataset or a HuggingFace dataset id."""
    split: str = "train"
    """Dataset split to use (e.g. ``train``, ``validation``)."""
    num_samples_limit: int | None = None
    """Deprecated alias for ``limit_dataset_samples``."""
    pad_to_max_length: bool = True
    """Whether to pad sequences to the dataset max length."""

    limit_dataset_samples: int | None = None
    """If set, limit the dataset to this many samples via slicing."""

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | None") -> "HellaSwag":
        """Build a :class:`HellaSwag` dataset from this :class:`HellaSwagConfig` and a runtime tokenizer."""
        if tokenizer is None:
            raise ValueError("HellaSwag requires a tokenizer")
        return HellaSwag(
            path_or_dataset=self.path_or_dataset,
            tokenizer=tokenizer,
            split=self.split,
            num_samples_limit=self.num_samples_limit,
            limit_dataset_samples=self.limit_dataset_samples,
            pad_to_max_length=self.pad_to_max_length,
        )


class HellaSwag:
    """A dataset wrapper for the HellaSwag benchmark, tailored for single-turn supervised fine-tuning (SFT).

    This class loads and preprocesses the HellaSwag dataset using a tokenizer and a custom preprocessing
    pipeline for language model fine-tuning. The dataset consists of context and multiple-choice endings,
    where the goal is to choose the most plausible continuation.

    Attributes:
        dataset (Dataset): The processed dataset ready for model training or evaluation.
    """

    def __init__(
        self,
        path_or_dataset: str,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        num_samples_limit: int | None = None,
        pad_to_max_length: bool = True,
        *,
        limit_dataset_samples: int | None = None,
    ) -> None:
        """Initialize the HellaSwag dataset wrapper.

        Args:
            path_or_dataset (str or Dataset): Path to the dataset or a HuggingFace Dataset object.
            tokenizer (PreTrainedTokenizer): The tokenizer used to process text.
            split (str, optional): Dataset split to use (e.g., 'train', 'validation'). Defaults to 'train'.
            num_samples_limit: Deprecated alias for ``limit_dataset_samples``.
            limit_dataset_samples: Maximum number of samples to load. Defaults to None.
            pad_to_max_length (bool, optional): Whether to pad sequences to max length in the dataset.
                If False, sequences will have variable lengths and padding will be handled by the collate function.
                Defaults to True.

        Notes:
            If limit_dataset_samples is an integer, it limits the dataset size using slicing.
        """
        limit_dataset_samples = resolve_sample_limit(limit_dataset_samples, num_samples_limit)
        if isinstance(limit_dataset_samples, int):
            split = f"{split}[:{limit_dataset_samples}]"
        raw_datasets = load_dataset(path_or_dataset, split=split)
        processor = SFTSingleTurnPreprocessor(tokenizer)
        processor.pad_to_max_length = pad_to_max_length
        self.dataset = processor.process(raw_datasets, self)

    def get_context(self, examples):
        """Extracts the context part of each example.

        Args:
            examples (dict): A dictionary containing example data with a "ctx" key.

        Returns:
            list[str]: List of context strings.
        """
        return examples["ctx"]

    def get_target(self, examples):
        """Extracts the correct ending based on the label.

        Args:
            examples (dict): A dictionary with "endings" (list of strings) and "label" (index of correct ending).

        Returns:
            list[str]: The gold target strings based on the label index.
        """
        return [endings[int(lbl)] for endings, lbl in zip(examples["endings"], examples["label"], strict=False)]

    def __getitem__(self, index):
        """Get a processed example by index.

        Args:
            index (int): Index of the example.

        Returns:
            dict: A tokenized and preprocessed example.
        """
        ans = self.dataset[index]
        return ans

    def __len__(self):
        """Get the number of examples in the dataset.

        Returns:
            int: Length of the processed dataset.
        """
        return len(self.dataset)
