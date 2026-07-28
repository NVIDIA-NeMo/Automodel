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

"""Typed construction for the packed BAGEL multimodal dataloader."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.datasets.multimodal.collate_fns import bagel_packed_collate_fn
from nemo_automodel.components.datasets.multimodal.datasets import BagelDatasetConfig
from nemo_automodel.components.datasets.multimodal.packing import PackedDataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class BagelDataloaderBuild:
    """Materialized BAGEL dataloader and its packed dataset."""

    dataloader: StatefulDataLoader
    dataset: PackedDataset


@dataclass
class BagelDataloaderConfig:
    """Construction-time configuration for the complete BAGEL input pipeline."""

    dataset_config: BagelDatasetConfig
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 2

    def build(
        self,
        *,
        tokenizer: "PreTrainedTokenizerBase",
        special_tokens: Mapping[str, object],
        rank: int,
        world_size: int,
        batch_size: int,
        global_seed: int,
    ) -> BagelDataloaderBuild:
        """Build the packed BAGEL dataset and its stateful dataloader.

        Args:
            tokenizer: Runtime tokenizer used by the packed dataset.
            special_tokens: Runtime BAGEL special-token mapping.
            rank: Global distributed rank used for source-data sharding and worker seeding.
            world_size: Global distributed world size.
            batch_size: Runtime local batch size; BAGEL packed samples require one row per step.
            global_seed: Global training seed used to initialize dataloader worker RNG state.

        Returns:
            Named result containing the dataloader and its packed dataset.
        """
        if batch_size != 1:
            raise ValueError(f"BAGEL packed dataloaders require local_batch_size=1, got {batch_size}")
        dataset = self.dataset_config.build(
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            rank=rank,
            world_size=world_size,
            num_workers=self.num_workers,
            global_seed=global_seed,
        )
        rank_seed = global_seed * world_size + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rank_seed)

        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=bagel_packed_collate_fn,
            drop_last=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        return BagelDataloaderBuild(dataloader=dataloader, dataset=dataset)


__all__ = ["BagelDataloaderBuild", "BagelDataloaderConfig"]
