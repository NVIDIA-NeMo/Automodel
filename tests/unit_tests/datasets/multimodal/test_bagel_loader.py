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

"""Tests for typed BAGEL dataset and dataloader construction."""

from unittest.mock import ANY, patch

import pytest
from torch.utils.data import IterableDataset

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.multimodal.datasets import BagelDatasetConfig
from nemo_automodel.components.datasets.multimodal.loader import BagelDataloaderConfig
from nemo_automodel.recipes._typed_config import RecipeConfig


class FakePackedDataset(IterableDataset):
    builds = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.epochs = []
        type(self).builds.append(self)
        kwargs["data_config"].grouped_datasets["group"]["weight"] = 99

    def set_epoch(self, seed):
        self.epochs.append(seed)

    def __iter__(self):
        yield {"sample": 1}


def _bagel_dataset_config():
    return BagelDatasetConfig(
        grouped_datasets={"group": {"weight": 1}},
        dataset_info={"group": {"source": {"data_dir": "/tmp"}}},
        data_seed=17,
    )


def test_bagel_dataset_config_build_is_repeatable_without_mutating_declarative_state():
    FakePackedDataset.builds = []
    config = _bagel_dataset_config()

    with patch("nemo_automodel.components.datasets.multimodal.packing.PackedDataset", FakePackedDataset):
        first = config.build(
            tokenizer=object(),
            special_tokens={},
            rank=1,
            world_size=2,
            num_workers=3,
            global_seed=23,
        )
        second = config.build(
            tokenizer=object(),
            special_tokens={},
            rank=1,
            world_size=2,
            num_workers=3,
            global_seed=23,
        )

    assert config.grouped_datasets == {"group": {"weight": 1}}
    assert first is not second
    assert first.epochs == [17]
    assert second.epochs == [17]
    assert first.kwargs["global_seed"] == 23


def test_bagel_dataloader_config_rejects_automatic_batching_of_packed_rows():
    config = BagelDataloaderConfig(dataset_config=_bagel_dataset_config(), num_workers=0)

    with pytest.raises(ValueError, match="local_batch_size=1"):
        config.build(
            tokenizer=object(),
            special_tokens={},
            rank=0,
            world_size=1,
            batch_size=2,
            global_seed=1,
        )


def test_bagel_dataloader_config_builds_and_iterates_on_cpu():
    config = BagelDataloaderConfig(dataset_config=_bagel_dataset_config(), num_workers=0, pin_memory=False)
    dataset = FakePackedDataset(data_config=_bagel_dataset_config())

    with (
        patch.object(config.dataset_config, "build", return_value=dataset) as build_dataset,
        patch(
            "nemo_automodel.components.datasets.multimodal.loader.bagel_packed_collate_fn",
            new=lambda batch: batch[0],
        ),
    ):
        result = config.build(
            tokenizer=object(),
            special_tokens={"image_start_token": 1},
            rank=0,
            world_size=1,
            batch_size=1,
            global_seed=5,
        )

    assert next(iter(result.dataloader)) == {"sample": 1}
    assert result.dataset is dataset
    build_dataset.assert_called_once_with(
        tokenizer=ANY,
        special_tokens={"image_start_token": 1},
        rank=0,
        world_size=1,
        num_workers=0,
        global_seed=5,
    )


def test_recipe_config_resolves_bagel_dataset_and_dataloader_fields():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "grouped_datasets": {"group": {"weight": 1}},
                    "dataset_info": {"group": {"source": {"data_dir": "/tmp"}}},
                    "expected_num_tokens": 128,
                    "data_seed": 9,
                },
                "dataloader": {"num_workers": 2, "pin_memory": False, "prefetch_factor": 4},
            }
        )
    ).bagel_dataloader

    assert config.dataset_config.expected_num_tokens == 128
    assert config.dataset_config.data_seed == 9
    assert config.num_workers == 2
    assert config.pin_memory is False
    assert config.prefetch_factor == 4
