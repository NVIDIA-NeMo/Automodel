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

from dataclasses import dataclass

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.vlm.loader import (
    VlmCollatorConfig,
    VlmDataloaderConfig,
)
from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig
from nemo_automodel.recipes._typed_config import RecipeConfig


class DummyProcessor:
    def __init__(self):
        self.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()


@dataclass
class StaticDatasetConfig:
    events: list[str]

    def build(self):
        self.events.append("dataset")
        return ["one", "two"]


class BuildContext:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")

    def __exit__(self, *_):
        self.events.append("exit")


def test_recipe_config_separates_vlm_dataset_wrapper_and_packing_fields():
    config = RecipeConfig(
        ConfigNode(
            {
                "model": {
                    "_target_": "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained",
                    "pretrained_model_name_or_path": "org/model",
                },
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset",
                    "num_samples": 4,
                    "max_length": 128,
                    "pretokenize": True,
                    "inject_fake_images": False,
                },
                "packed_sequence": {
                    "pretokenize": True,
                    "max_length": 128,
                    "pack_size": 128,
                    "packing_ratio": 0.9,
                    "collate_max_length": 128,
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).dataloader.train

    assert config is not None
    assert isinstance(config.dataset_config, MockVlmDatasetConfig)
    assert config.dataset_config.max_length == 128
    assert config.pretokenization.max_length == 128
    assert config.pretokenization.inject_fake_images is False
    assert config.packing.pack_size == 128
    assert config.packing.packing_ratio == 0.9
    assert config.packing.collate_max_length == 128


def test_vlm_dataloader_uses_runtime_tokenizer_and_builds_dataset_inside_context():
    events = []
    processor = DummyProcessor()

    def collate(examples, *, processor, prefix):
        return [prefix + example for example in examples], processor

    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig(events),
        collator=VlmCollatorConfig(factory=collate, kwargs={"prefix": "item:"}),
        shuffle=False,
        num_workers=0,
    )

    dataloader = config.build(
        tokenizer=processor,
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        dataset_build_context=BuildContext(events),
    )
    batch, batch_processor = next(iter(dataloader))

    assert events == ["enter", "dataset", "exit"]
    assert batch == ["item:one", "item:two"]
    assert batch_processor is processor
