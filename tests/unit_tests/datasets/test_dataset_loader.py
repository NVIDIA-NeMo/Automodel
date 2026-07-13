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
import torch
from torch.utils.data import IterableDataset

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.megatron.sampler import MegatronSamplerConfig
from nemo_automodel.components.datasets.llm.retrieval_dataset import RetrievalDatasetConfig
from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import InlineRetrievalDatasetConfig
from nemo_automodel.components.datasets.loader import (
    CollatorConfig,
    DataloaderConfig,
    ParallelAwareDataloader,
    ThdPackingConfig,
    make_collate_fn,
    make_packing_config,
)
from nemo_automodel.recipes._typed_config import RecipeConfig


class TokenizerAwareCollator:
    init_count = 0

    def __init__(self, tokenizer: object, prefix: str) -> None:
        type(self).init_count += 1
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __call__(self, batch: list[str]) -> list[str]:
        return [self.prefix + item for item in batch]


@dataclass
class StaticDatasetConfig:
    values: object

    def build(self) -> object:
        return self.values


def test_collator_config_builds_class_once_with_runtime_tokenizer():
    TokenizerAwareCollator.init_count = 0
    tokenizer = object()
    config = make_collate_fn(TokenizerAwareCollator, {"prefix": "query: "})

    assert isinstance(config, CollatorConfig)
    collator = config.build(tokenizer=tokenizer)

    assert TokenizerAwareCollator.init_count == 1
    assert collator.tokenizer is tokenizer
    assert collator(["one"]) == ["query: one"]
    assert collator(["two"]) == ["query: two"]
    assert TokenizerAwareCollator.init_count == 1


def test_dataloader_build_instantiates_collator_once():
    TokenizerAwareCollator.init_count = 0
    config = DataloaderConfig(
        dataset_config=StaticDatasetConfig(["one", "two"]),
        batch_size=1,
        collate_fn=CollatorConfig(TokenizerAwareCollator, {"prefix": "query: "}),
        num_workers=0,
    )

    loader = config.build(tokenizer=object(), dp_rank=0, dp_world_size=1)

    assert list(loader) == [["query: one"], ["query: two"]]
    assert TokenizerAwareCollator.init_count == 1


def test_recipe_config_resolves_top_level_retrieval_dataset_and_collator():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": ("nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig"),
                    "data_dir_list": ["train.jsonl"],
                    "n_passages": 3,
                },
                "dataloader": {
                    "collate_fn": {
                        "_target_": "tests.unit_tests.datasets.test_dataset_loader.TokenizerAwareCollator",
                        "prefix": "passage: ",
                    },
                    "batch_size": 2,
                    "num_workers": 0,
                },
            }
        )
    ).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, RetrievalDatasetConfig)
    assert config.dataset_config.n_passages == 3
    assert isinstance(config.collate_fn, CollatorConfig)
    assert config.batch_size == 2


def test_recipe_config_supports_legacy_nested_dataset():
    raw = ConfigNode(
        {
            "dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["train.jsonl"],
                    "n_passages": 3,
                },
                "batch_size": 2,
            }
        }
    )

    with pytest.warns(FutureWarning, match="dataloader.dataset.*top-level `dataset`"):
        config = RecipeConfig(raw).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, RetrievalDatasetConfig)
    assert config.dataset_config.n_passages == 3
    assert config.batch_size == 2


def test_recipe_config_prefers_top_level_dataset_over_legacy_nested_dataset():
    raw = ConfigNode(
        {
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                "data_dir_list": ["top-level.jsonl"],
                "n_passages": 5,
            },
            "dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["nested.jsonl"],
                    "n_passages": 2,
                }
            },
        }
    )

    with pytest.warns(FutureWarning, match="dataloader.dataset.*top-level `dataset`"):
        config = RecipeConfig(raw).dataloader

    assert config is not None
    assert config.dataset_config.data_dir_list == ["top-level.jsonl"]
    assert config.dataset_config.n_passages == 5


def test_recipe_config_supports_legacy_nested_validation_dataset():
    raw = ConfigNode(
        {
            "validation_dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["validation.jsonl"],
                    "n_passages": 4,
                },
                "batch_size": 2,
            }
        }
    )

    with pytest.warns(FutureWarning, match="validation_dataloader.dataset.*top-level `validation_dataset`"):
        configs = RecipeConfig(raw).validation_dataloaders

    assert set(configs) == {"default"}
    assert configs["default"].dataset_config.n_passages == 4
    assert configs["default"].batch_size == 2


def test_legacy_inline_retrieval_target_uses_exact_typed_config():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": (
                        "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset"
                    ),
                    "data_dir_list": ["train.jsonl"],
                }
            }
        )
    ).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, InlineRetrievalDatasetConfig)


def test_recipe_config_rejects_unknown_dataset_field():
    raw = ConfigNode(
        {
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                "data_dir_list": ["train.jsonl"],
                "n_passsages": 3,
            }
        }
    )

    with pytest.raises(TypeError, match="n_passsages"):
        _ = RecipeConfig(raw).dataloader


def test_packing_config_rejects_unknown_field():
    with pytest.raises(TypeError, match="packed_sequnce_size"):
        make_packing_config("thd", {"packed_sequence_size": 8, "packed_sequnce_size": 8})


def test_packing_config_warns_for_ignored_legacy_field():
    with pytest.warns(FutureWarning, match="split_across_pack"):
        config = make_packing_config("thd", {"packed_sequence_size": 8, "split_across_pack": False})

    assert isinstance(config, ThdPackingConfig)


def test_megatron_loader_preserves_schedule_and_sampler_config():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining",
                    "paths": ["train"],
                    "splits_to_build": "train",
                },
                "dataloader": {"dataloader_type": "single"},
                "step_scheduler": {
                    "local_batch_size": 2,
                    "global_batch_size": 16,
                    "max_steps": 50,
                    "val_every_steps": 5,
                },
            }
        )
    ).dataloader

    assert config is not None
    assert config.dataset_builds_on_all_ranks is True
    assert config.dataset_build_schedule.local_batch_size == 2
    assert config.dataset_build_schedule.global_batch_size == 16
    assert config.dataset_build_schedule.max_steps == 50
    assert config.dataset_build_schedule.val_check_interval == 5
    assert isinstance(config.batch_sampler_config, MegatronSamplerConfig)
    assert next(iter(config.batch_sampler_config.build(dataset_len=32, rank=1, world_size=2))) == [2, 3]


def test_iterable_shuffle_failure_is_not_silently_ignored():
    class BrokenShuffleDataset(IterableDataset):
        def __iter__(self):
            yield "sample"

        def shard(self, *_):
            return self

        def shuffle(self, **_):
            raise ValueError("invalid shuffle configuration")

    with pytest.raises(ValueError, match="invalid shuffle configuration"):
        ParallelAwareDataloader(
            BrokenShuffleDataset(),
            dp_rank=0,
            dp_world_size=1,
            shuffle=True,
            num_workers=0,
        )


def test_prebatched_iterable_disables_automatic_batching():
    class PrebatchedDataset(IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.ones((2, 4), dtype=torch.long)}

        def shard(self, *_):
            return self

    config = DataloaderConfig(dataset_config=StaticDatasetConfig(PrebatchedDataset()), batch_size=None)

    batch = next(iter(config.build(dp_rank=0, dp_world_size=1)))

    assert batch["input_ids"].shape == (2, 4)


def test_rank_ordering_context_wraps_only_dataset_construction():
    events = []

    class BuildContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_):
            events.append("exit")

    class RecordingDatasetConfig:
        def build(self):
            events.append("dataset")
            return ["sample"]

    class RecordingPackingConfig:
        prepacked = False

        def build(self, dataset, **_):
            events.append("packing")
            return dataset, None

    config = DataloaderConfig(
        dataset_config=RecordingDatasetConfig(),
        packing=RecordingPackingConfig(),
        batch_size=1,
    )

    config.build(dp_rank=0, dp_world_size=1, dataset_build_context=BuildContext())

    assert events == ["enter", "dataset", "exit", "packing"]
