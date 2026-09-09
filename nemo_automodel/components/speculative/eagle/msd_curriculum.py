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

"""Two-stage data curriculum for Multimodal Speculative Decoding training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class MSDDataSource(str, Enum):
    """Data source selected for one MSD optimizer step."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


class MSDCurriculumPhase(str, Enum):
    """High-level phase of the two-stage MSD training schedule."""

    TEXT_ONLY = "text_only"
    MULTIMODAL_RAMP = "multimodal_ramp"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class MSDCurriculumConfig:
    """Configuration for the two-stage MSD data curriculum.

    The first stage consumes text-only instruction data. In the second stage,
    the probability of selecting a multimodal batch rises linearly to
    ``final_multimodal_ratio``. A final ratio of one makes the post-ramp phase
    entirely multimodal, matching the curriculum described in the MSD paper.

    Args:
        text_only_steps: Number of initial optimizer steps using only text
            batches.
        multimodal_ramp_steps: Number of stage-two steps over which the
            multimodal sampling probability rises to its final value. Set to
            zero to switch immediately after the text-only stage.
        final_multimodal_ratio: Multimodal batch probability after the ramp.
        seed: Seed used for stateless, step-indexed source selection. All
            distributed ranks therefore select the same source for a global
            optimizer step without communicating RNG state.
    """

    text_only_steps: int
    multimodal_ramp_steps: int
    final_multimodal_ratio: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.text_only_steps < 0:
            raise ValueError("text_only_steps must be non-negative.")
        if self.multimodal_ramp_steps < 0:
            raise ValueError("multimodal_ramp_steps must be non-negative.")
        if not 0.0 <= self.final_multimodal_ratio <= 1.0:
            raise ValueError("final_multimodal_ratio must be in [0, 1].")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


class MSDCurriculum:
    """Stateless-by-step two-stage curriculum with checkpointable progress.

    Call :meth:`source_for_step` from a training loop that owns separate text
    and multimodal dataloaders, then draw the selected dataloader's next batch.
    :meth:`next_source` is a convenience for single-process loops and stores
    the next global step in its checkpoint state.
    """

    def __init__(self, config: MSDCurriculumConfig, *, global_step: int = 0) -> None:
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        self.config = config
        self.global_step = global_step

    def multimodal_probability(self, global_step: int) -> float:
        """Return the multimodal batch probability at ``global_step``."""
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        if global_step < self.config.text_only_steps:
            return 0.0
        if self.config.multimodal_ramp_steps == 0:
            return self.config.final_multimodal_ratio

        ramp_step = global_step - self.config.text_only_steps + 1
        ramp_progress = min(ramp_step / self.config.multimodal_ramp_steps, 1.0)
        return self.config.final_multimodal_ratio * ramp_progress

    def phase_for_step(self, global_step: int) -> MSDCurriculumPhase:
        """Return the curriculum phase active at ``global_step``."""
        probability = self.multimodal_probability(global_step)
        if probability == 0.0:
            return MSDCurriculumPhase.TEXT_ONLY
        if probability < self.config.final_multimodal_ratio:
            return MSDCurriculumPhase.MULTIMODAL_RAMP
        return MSDCurriculumPhase.MULTIMODAL

    def source_for_step(self, global_step: int) -> MSDDataSource:
        """Deterministically choose the data source for ``global_step``.

        The random draw is seeded from the global step rather than mutable RNG
        state. Resumed and distributed runs therefore make the same sampling
        decision for a given optimizer step.
        """
        probability = self.multimodal_probability(global_step)
        if probability == 0.0:
            return MSDDataSource.TEXT
        if probability == 1.0:
            return MSDDataSource.MULTIMODAL

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + global_step)
        is_multimodal = torch.rand((), generator=generator).item() < probability
        return MSDDataSource.MULTIMODAL if is_multimodal else MSDDataSource.TEXT

    def next_source(self) -> MSDDataSource:
        """Return the current source selection and advance one optimizer step."""
        source = self.source_for_step(self.global_step)
        self.global_step += 1
        return source

    def state_dict(self) -> dict[str, Any]:
        """Return progress required to resume :meth:`next_source`."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore curriculum progress from a checkpoint state dictionary."""
        global_step = state_dict.get("global_step")
        if not isinstance(global_step, int) or global_step < 0:
            raise ValueError("MSD curriculum checkpoint must contain a non-negative integer global_step.")
        self.global_step = global_step
