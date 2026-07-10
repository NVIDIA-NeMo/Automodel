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

from contextlib import nullcontext
from dataclasses import dataclass

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.retrieval_dataset import RetrievalDatasetConfig
from nemo_automodel.components.datasets.loader import CollatorConfig, DataloaderConfig, make_collate_fn
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
    values: list[str]

    def build(self) -> list[str]:
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


def test_dataloader_build_instantiates_collator_once(monkeypatch):
    TokenizerAwareCollator.init_count = 0
    monkeypatch.setattr("nemo_automodel.components.training.rng.ScopedRNG", lambda **kwargs: nullcontext())
    monkeypatch.setattr("nemo_automodel.components.distributed.utils.FirstRankPerNode", lambda: nullcontext())
    config = DataloaderConfig(
        dataset_config=StaticDatasetConfig(["one", "two"]),
        batch_size=1,
        collate_fn=CollatorConfig(TokenizerAwareCollator, {"prefix": "query: "}),
        loader_kwargs={"num_workers": 0},
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
