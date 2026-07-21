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

"""Typed result and construction contract for diffusion dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch.utils.data import DataLoader, Sampler


@dataclass(frozen=True)
class DiffusionDataloaderBuild:
    """Materialized diffusion dataloader and its sampler."""

    dataloader: DataLoader
    sampler: Sampler[object] | None


class DiffusionDataloaderConfig(Protocol):
    """Typed construction contract used by the diffusion recipe."""

    def build(
        self,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
    ) -> DiffusionDataloaderBuild:
        """Build the configured per-rank diffusion dataloader."""


__all__ = ["DiffusionDataloaderBuild", "DiffusionDataloaderConfig"]
