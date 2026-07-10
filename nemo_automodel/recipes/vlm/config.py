# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Typed VLM train and validation dataloader composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ProcessorMixin

from nemo_automodel.components.datasets.vlm.loader import VlmDataloaderConfig
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.components.training.rng import ScopedRNG


@dataclass(frozen=True)
class VlmInputPipeline:
    """Materialized VLM training and named validation dataloaders."""

    train: DataLoader
    validation: Mapping[str, DataLoader] = field(default_factory=dict)


@dataclass(frozen=True)
class VlmInputConfig:
    """Declarative VLM training and validation input-pipeline composition."""

    train: VlmDataloaderConfig
    validation: Mapping[str, VlmDataloaderConfig] = field(default_factory=dict)
    batch_size: int = 1
    seed: int = 42
    attn_implementation: str | None = None

    def build(
        self,
        *,
        model: nn.Module | AutoPipeline,
        tokenizer: ProcessorMixin | None,
        dp_rank: int,
        dp_world_size: int,
        pp_enabled: bool = False,
        cp_size: int = 1,
    ) -> VlmInputPipeline:
        """Build VLM training and validation dataloaders.

        Args:
            model: Live model used to resolve multimodal position and pipeline capabilities.
            tokenizer: Runtime multimodal tokenizer or processor used by datasets and collators.
            dp_rank: Rank within the data-parallel group.
            dp_world_size: Size of the data-parallel group.
            pp_enabled: Whether pipeline parallelism is enabled.
            cp_size: Context-parallel world size used by packed multimodal inputs.

        Returns:
            Materialized training and named validation dataloaders.
        """
        model_part = model.parts[0] if isinstance(model, AutoPipeline) else model
        get_rope_index = getattr(model_part, "get_rope_index", None)
        packing_attn_implementation = self.train.resolve_packing_attn_implementation(
            model_attn_implementation=self.attn_implementation,
            cp_size=cp_size,
        )
        if self.train.packing is not None:
            configure_packing(attn_implementation=packing_attn_implementation or "sdpa")

        pp_n_microbatches = None
        prepares_cp_inputs = callable(getattr(model_part, "prepare_model_inputs_for_cp", None))
        if pp_enabled and isinstance(model, AutoPipeline) and not (cp_size > 1 and prepares_cp_inputs):
            pp_n_microbatches = model.pp_batch_size // model.pp_microbatch_size

        def materialize(
            config: VlmDataloaderConfig,
            *,
            packing_backend: str | None = None,
            pipeline_microbatches: int | None = None,
        ) -> DataLoader:
            """Build one VLM dataloader with the shared runtime inputs."""
            with ScopedRNG(seed=self.seed, ranked=True):
                return config.build(
                    tokenizer=tokenizer,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    batch_size=self.batch_size,
                    dataset_build_context=FirstRankPerNode(),
                    get_rope_index=get_rope_index,
                    packing_attn_implementation=packing_backend,
                    pp_n_microbatches=pipeline_microbatches,
                )

        return VlmInputPipeline(
            train=materialize(
                self.train,
                packing_backend=packing_attn_implementation,
                pipeline_microbatches=pp_n_microbatches,
            ),
            validation={name: materialize(config) for name, config in self.validation.items()},
        )


__all__ = ["VlmInputConfig", "VlmInputPipeline"]
