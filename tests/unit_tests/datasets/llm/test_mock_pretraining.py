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

import pytest
import torch

from nemo_automodel.components.datasets.llm.mock_pretraining import MockPretrainingDatasetConfig


def test_mock_pretraining_dataset_builds_deterministic_shifted_samples() -> None:
    config = MockPretrainingDatasetConfig(
        vocab_size=32,
        seq_length=8,
        num_samples=4,
    )
    first_dataset = config.build()
    second_dataset = config.build()

    for index in range(len(first_dataset)):
        first = first_dataset[index]
        second = second_dataset[index]
        torch.testing.assert_close(first["input_ids"][1:], first["labels"][:-1])
        torch.testing.assert_close(first["input_ids"], second["input_ids"])
        torch.testing.assert_close(first["labels"], second["labels"])
        torch.testing.assert_close(first["position_ids"], torch.arange(8))


def test_mock_pretraining_dataset_uses_distinct_splits() -> None:
    common = {
        "vocab_size": 32,
        "seq_length": 8,
        "num_samples": 4,
    }
    train = MockPretrainingDatasetConfig(**common, split="train").build()
    validation = MockPretrainingDatasetConfig(**common, split="validation").build()

    assert not torch.equal(train[0]["input_ids"], validation[0]["input_ids"])


def test_mock_pretraining_dataset_inserts_end_of_document_tokens() -> None:
    dataset = MockPretrainingDatasetConfig(
        vocab_size=5000,
        eod_token_id=4998,
        seq_length=512,
        num_samples=32,
    ).build()

    assert any(torch.any(dataset[index]["labels"] == 4998) for index in range(len(dataset)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vocab_size", 1, "vocab_size must be at least 2"),
        ("eod_token_id", 16, "eod_token_id must be in"),
        ("seq_length", 0, "seq_length must be positive"),
        ("num_samples", 0, "num_samples must be positive"),
        ("split", "dev", "split must be train, validation, or test"),
    ],
)
def test_mock_pretraining_dataset_rejects_invalid_config(field: str, value: object, message: str) -> None:
    kwargs = {"vocab_size": 16, "seq_length": 4, "num_samples": 2, field: value}
    with pytest.raises(ValueError, match=message):
        MockPretrainingDatasetConfig(**kwargs)


def test_mock_pretraining_dataset_rejects_excess_samples() -> None:
    config = MockPretrainingDatasetConfig(
        vocab_size=16,
        seq_length=512,
        num_samples=200_000,
    )

    with pytest.raises(ValueError, match="exceeds the .* complete samples"):
        config.build()
