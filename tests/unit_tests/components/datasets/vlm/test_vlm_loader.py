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

import pytest

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.vlm.collate_fns import (
    neat_packed_vlm_collater,
    packed_sequence_thd_vlm_collater,
)
from nemo_automodel.components.datasets.vlm.datasets import CordV2DatasetConfig, PreTokenizedDatasetWrapperConfig
from nemo_automodel.components.datasets.vlm.loader import (
    VlmCollatorConfig,
    VlmDataloaderConfig,
    VlmProcessorConfig,
    VlmVideoProcessorConfig,
)
from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig
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
                    "packing_format": "thd",
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader

    assert config is not None
    assert isinstance(config.dataset_config, MockVlmDatasetConfig)
    assert config.dataset_config.max_length == 128
    assert config.pretokenization.max_length == 128
    assert config.pretokenization.inject_fake_images is False
    assert config.packing.pack_size == 128
    assert config.packing.packing_ratio == 0.9
    assert config.packing.collate_max_length == 128
    assert config.packing.packing_format == "thd"


def test_recipe_config_rejects_unknown_vlm_packing_format():
    raw = ConfigNode(
        {
            "dataset": {"_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset"},
            "packed_sequence": {"pack_size": 128, "packing_format": "unknown"},
        }
    )

    with pytest.raises(ValueError, match="Unsupported VLM packing_format"):
        _ = RecipeConfig(raw).vlm_dataloader


def test_recipe_config_accepts_cord_v2_sample_limit():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset",
                    "limit_dataset_samples": 100,
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader

    assert isinstance(config.dataset_config, CordV2DatasetConfig)
    assert config.dataset_config.limit_dataset_samples == 100


def test_vlm_dataloader_builds_processor_and_dataset_inside_context_then_iterates():
    events = []
    processor = DummyProcessor()

    def build_processor():
        events.append("processor")
        return processor

    def collate(examples, *, processor, prefix):
        return [prefix + example for example in examples], processor

    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig(events),
        processor_config=VlmProcessorConfig(factory=build_processor),
        collator=VlmCollatorConfig(factory=collate, kwargs={"prefix": "item:"}),
        shuffle=False,
        num_workers=0,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        dataset_build_context=BuildContext(events),
    )
    batch, batch_processor = next(iter(result.dataloader))

    assert events == ["enter", "processor", "dataset", "exit"]
    assert result.processor is processor
    assert batch == ["item:one", "item:two"]
    assert batch_processor is processor


def test_vlm_processor_builds_independently_configured_video_processor():
    video_processor = object()
    calls = []

    def build_video_processor(*, pretrained_model_name_or_path, size, fps, max_frames):
        calls.append(("video", pretrained_model_name_or_path, size, fps, max_frames))
        return video_processor

    def build_processor(*, model_id, video_processor):
        calls.append(("processor", model_id, video_processor))
        return DummyProcessor()

    config = VlmProcessorConfig(
        factory=build_processor,
        kwargs={"model_id": "outer-model"},
        video_processor=VlmVideoProcessorConfig(
            factory=build_video_processor,
            kwargs={
                "size": {"shortest_edge": 1024, "longest_edge": 524288},
                "fps": 2,
                "max_frames": 8,
            },
        ),
    )

    result = config.build(pretrained_model_name_or_path="runtime-model")

    assert isinstance(result, DummyProcessor)
    assert calls == [
        ("video", "runtime-model", {"shortest_edge": 1024, "longest_edge": 524288}, 2, 8),
        ("processor", "outer-model", video_processor),
    ]


def test_recipe_config_resolves_nested_vlm_video_processor():
    def build_video_processor(**kwargs):
        return kwargs

    def build_processor(**kwargs):
        return kwargs

    config = RecipeConfig(
        ConfigNode(
            {
                "processor": {
                    "_target_": build_processor,
                    "pretrained_model_name_or_path": "outer-model",
                    "video_processor": {
                        "_target_": build_video_processor,
                        "size": {"shortest_edge": 1024, "longest_edge": 524288},
                        "fps": 2,
                        "max_frames": 8,
                    },
                },
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset",
                    "num_samples": 1,
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader.processor_config

    assert config.factory is build_processor
    assert config.kwargs == {"pretrained_model_name_or_path": "outer-model"}
    assert config.video_processor is not None
    assert config.video_processor.factory is build_video_processor
    assert config.video_processor.kwargs == {
        "size": {"shortest_edge": 1024, "longest_edge": 524288},
        "fps": 2,
        "max_frames": 8,
    }


def test_vlm_dataloader_selects_thd_collater(monkeypatch):
    processor = DummyProcessor()
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(packing_format="thd"),
        shuffle=False,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
    )

    assert result.dataloader.collate_fn.func is packed_sequence_thd_vlm_collater
    assert result.dataloader.collate_fn.keywords == {"padding_idx": 0, "max_length": None}


def test_vlm_dataloader_skips_dense_neat_packing_mask_under_cp(monkeypatch):
    processor = DummyProcessor()
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(),
        shuffle=False,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        packing_attn_implementation="sdpa",
        cp_size=32,
    )

    assert result.dataloader.collate_fn.func is neat_packed_vlm_collater
    assert result.dataloader.collate_fn.keywords["attn_implementation"] == "sdpa"
    assert result.dataloader.collate_fn.keywords["materialize_4d_mask"] is False
