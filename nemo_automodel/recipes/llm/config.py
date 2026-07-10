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

"""Typed construction for LLM recipe input pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch.nn as nn

from nemo_automodel.components.datasets.loader import CollateFn, DataloaderConfig
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.training.rng import ScopedRNG

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmInputPipeline:
    """Materialized training and validation dataloaders."""

    train: "DataLoader"
    validation: dict[str, "DataLoader"]


@dataclass(frozen=True)
class LlmInputConfig:
    """Declarative LLM dataloader composition and model-dependent layout policy.

    The child :class:`DataloaderConfig` objects own dataset and loader
    construction. This config owns the LLM-specific composition policy that
    depends on the live model: packed-validation layout, NEAT model setup, and
    pipeline-parallel causal-mask wrapping.
    """

    train: DataloaderConfig
    validation: Mapping[str, DataloaderConfig]
    attn_implementation: str = "sdpa"

    @staticmethod
    def _uses_thd_collater(config: DataloaderConfig) -> bool:
        from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

        return config.collate_fn is packed_sequence_thd_collater

    @property
    def uses_thd_collater(self) -> bool:
        """Whether training explicitly selects the THD collater."""
        return self._uses_thd_collater(self.train)

    @property
    def requires_pp_thd_microbatch_override(self) -> bool:
        """Whether TE+THD pipeline batches require the single-microbatch layout."""
        return self.attn_implementation == "te" and self.uses_thd_collater

    def _pack_validation(self, config: DataloaderConfig, model: nn.Module) -> bool:
        if self.train.packing is None:
            return False
        if self._uses_thd_collater(config):
            return True

        model_policy = getattr(model, "should_pack_validation_with_training", None)
        model_requires_training_layout = callable(model_policy) and bool(model_policy())
        backend_requires_training_layout = self.attn_implementation in {"magi", "te"}
        return (model_requires_training_layout or backend_requires_training_layout) and self.uses_thd_collater

    @staticmethod
    def _supports_sequence_packing(model: nn.Module) -> bool:
        supports = getattr(model, "supports_sequence_packing", None)
        if supports is not None:
            return bool(supports)

        # Models built through NeMoAutoModel expose the forwarded capability
        # property. Custom builders can use the same public capability owner.
        from nemo_automodel._transformers.capabilities import ModelSupports

        return ModelSupports(model).supports_sequence_packing

    @staticmethod
    def _pp_collate_wrapper(model: nn.Module, pp_enabled: bool) -> Callable[[CollateFn], CollateFn] | None:
        if not pp_enabled:
            return None

        model_config = getattr(model, "config", None)
        if model_config is None:
            logger.warning(
                "The live model has no config for causal-mask precomputation. "
                "Pipeline parallel mask precomputation will be skipped."
            )
            return None
        if getattr(model_config, "model_type", None) == "deepseek_v4":
            logger.info("Skipping pipeline parallel causal mask precomputation for model_type=deepseek_v4.")
            return None

        from nemo_automodel.components.datasets.utils import add_causal_masks_to_batch

        def wrapper(base_collate_fn: CollateFn) -> CollateFn:
            def chained_collate_fn(batch: list[object]) -> object:
                """Collate examples and attach causal masks over the resulting token axis.

                Args:
                    batch: Raw examples accepted by ``base_collate_fn``. Token-list fields have per-example
                        shape ``[S_i]`` and pre-batched tensor fields have shape ``[B_i, ...]``.

                Returns:
                    Collated mapping with token tensors shaped ``[B, S]`` and a ``causal_mask_mapping`` over
                    the same ``S`` token axis. Other fields preserve the base collator's documented layout.
                """
                return add_causal_masks_to_batch(base_collate_fn(batch), model_config=model_config)

            return chained_collate_fn

        return wrapper

    def build(
        self,
        *,
        model: nn.Module,
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None",
        dp_rank: int,
        dp_world_size: int,
        pp_enabled: bool = False,
        cp_size: int = 1,
        dataset_build_process_group: "ProcessGroup | None" = None,
    ) -> LlmInputPipeline:
        """Build the training and validation input pipelines.

        Args:
            model: Live model used to resolve input capabilities and PP mask behavior.
            tokenizer: Runtime tokenizer or processor consumed by datasets and collators.
            dp_rank: Rank within the data-parallel group.
            dp_world_size: Size of the data-parallel group.
            pp_enabled: Whether pipeline parallelism is enabled.
            cp_size: Context-parallel world size used by sequence packing.
            dataset_build_process_group: Optional model-local process group used to serialize dataset construction.

        Returns:
            Materialized training and named validation dataloaders.
        """
        if self.train.packing is not None and self.train.packing.requires_model_configuration:
            from nemo_automodel.components.models.common.packing import configure_packing

            configure_packing(attn_implementation=self.attn_implementation)

        collate_wrapper = self._pp_collate_wrapper(model, pp_enabled)
        supports_sequence_packing = self._supports_sequence_packing(model)

        def materialize(config: DataloaderConfig) -> "DataLoader":
            build_context = (
                nullcontext()
                if config.dataset_builds_on_all_ranks
                else FirstRankPerNode(group=dataset_build_process_group)
            )
            with ScopedRNG(seed=config.seed, ranked=True):
                return config.build(
                    tokenizer=tokenizer,
                    dataset_build_context=build_context,
                    dp_rank=dp_rank,
                    dp_world_size=dp_world_size,
                    pp_enabled=pp_enabled,
                    supports_seq_lens=supports_sequence_packing,
                    cp_size=cp_size,
                    attn_implementation=self.attn_implementation,
                    collate_wrapper=collate_wrapper,
                )

        train = materialize(self.train)
        validation = {
            name: materialize(config if self._pack_validation(config, model) else replace(config, packing=None))
            for name, config in self.validation.items()
        }
        return LlmInputPipeline(train=train, validation=validation)


__all__ = ["LlmInputConfig", "LlmInputPipeline"]
