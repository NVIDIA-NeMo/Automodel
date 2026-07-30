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

"""Typed construction for VLM processors, datasets, packing, and dataloaders."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sized
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol, runtime_checkable

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor, ProcessorMixin

from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template
from nemo_automodel.components.datasets.loader import DatasetConfig, TokenizerDatasetConfig
from nemo_automodel.components.datasets.vlm.collate_fns import (
    COLLATE_FNS,
    neat_packed_vlm_collater,
    packed_sequence_thd_vlm_collater,
    pad_collate_fn,
)
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapperConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig
from nemo_automodel.components.datasets.vlm.pp_media import wrap_vlm_collate_for_pp

logger = logging.getLogger(__name__)

VlmCollateFn = Callable[[list[object]], object]


@runtime_checkable
class RankedVlmDatasetConfig(Protocol):
    """Dataset config whose build requires the runtime data-parallel rank."""

    builds_with_data_parallel_rank: bool

    def build(self, *, rank: int, world_size: int) -> object:
        """Build the per-rank dataset shard."""


@dataclass(frozen=True)
class VlmVideoProcessorConfig:
    """Declarative video-processor factory and keyword arguments."""

    factory: Callable[..., object]
    """Configured video-processor factory."""
    kwargs: dict[str, object] = field(default_factory=dict)
    """Declarative keyword arguments for the configured factory."""

    def build(self, *, pretrained_model_name_or_path: str) -> object:
        """Build the configured video processor.

        Args:
            pretrained_model_name_or_path: Runtime model identifier used when the nested config does not override it.

        Returns:
            Video-processor instance accepted by the parent processor factory.
        """
        kwargs = dict(self.kwargs)
        kwargs.setdefault("pretrained_model_name_or_path", pretrained_model_name_or_path)
        return self.factory(**kwargs)


@dataclass(frozen=True)
class VlmProcessorConfig:
    """Declarative processor factory and keyword arguments."""

    factory: Callable[..., ProcessorMixin] | None = None
    """Configured processor factory; ``None`` selects ``AutoProcessor.from_pretrained``."""
    kwargs: dict[str, object] = field(default_factory=dict)
    """Declarative keyword arguments for the configured processor factory."""
    video_processor: VlmVideoProcessorConfig | None = None
    """Optional independently configured video processor."""

    def build(self, *, pretrained_model_name_or_path: str) -> ProcessorMixin | None:
        """Build the configured processor.

        Args:
            pretrained_model_name_or_path: Runtime model identifier used by the default AutoProcessor factory.

        Returns:
            Processor instance, or ``None`` when the model has no compatible AutoProcessor.
        """
        kwargs = dict(self.kwargs)
        if self.video_processor is not None:
            kwargs["video_processor"] = self.video_processor.build(
                pretrained_model_name_or_path=pretrained_model_name_or_path
            )

        if self.factory is not None:
            return self.factory(**kwargs)

        try:
            return AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "num_hidden_layers" in message and ("layer_types" in message or "layer types" in message):
                from nemo_automodel._transformers.v4_patches.layer_types import relax_layer_types_validator

                relax_layer_types_validator()
                try:
                    return AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
                except Exception as retry_exc:
                    logger.warning(
                        "AutoProcessor not available for %s after relaxing layer-type validation: %s",
                        pretrained_model_name_or_path,
                        retry_exc,
                    )
                    return None
            logger.warning("AutoProcessor not available for %s: %s", pretrained_model_name_or_path, exc)
            return None


@dataclass(frozen=True)
class VlmCollatorConfig:
    """Declarative VLM collator factory."""

    factory: Callable[..., object]
    """Function that accepts ``examples`` and the runtime ``processor``."""
    kwargs: dict[str, object] = field(default_factory=dict)
    """Declarative keyword arguments bound once while building the dataloader."""

    def build(self, *, processor: ProcessorMixin | None) -> VlmCollateFn:
        """Bind the runtime processor to the configured collator."""
        return partial(self.factory, processor=processor, **self.kwargs)


@dataclass(frozen=True)
class VlmDataloaderBuild:
    """Materialized VLM dataloader and its processor."""

    dataloader: DataLoader
    processor: ProcessorMixin | None


@dataclass
class VlmDataloaderConfig:
    """Typed construction config for the complete VLM input pipeline."""

    dataset_config: DatasetConfig | RankedVlmDatasetConfig
    processor_config: VlmProcessorConfig = field(default_factory=VlmProcessorConfig)
    pretokenization: PreTokenizedDatasetWrapperConfig | None = None
    packing: NeatPackConfig | None = None
    collator: VlmCollatorConfig | None = None
    chat_template: str | None = None
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False

    def resolve_packing_attn_implementation(
        self,
        *,
        model_attn_implementation: str | None,
        cp_size: int,
    ) -> str | None:
        """Resolve the packed-collator mask backend.

        Args:
            model_attn_implementation: Attention implementation selected by the model config.
            cp_size: Runtime context-parallel world size.

        Returns:
            Attention implementation used to choose the packed attention-mask representation.
        """
        if self.packing is None:
            return None
        override = self.packing.attn_implementation
        if override is not None and cp_size > 1:
            return override
        if override not in (None, model_attn_implementation):
            logger.warning(
                "Ignoring packed_sequence.attn_implementation=%r at cp_size=1: the packed mask format must "
                "match the model attention backend (%r)",
                override,
                model_attn_implementation,
            )
        return model_attn_implementation

    def _build_source(
        self,
        *,
        pretrained_model_name_or_path: str,
        dp_rank: int,
        dp_world_size: int,
        dataset_build_context: AbstractContextManager[object] | None,
    ) -> tuple[object, ProcessorMixin | None]:
        """Build the processor and raw dataset under the caller-owned ordering context."""
        with dataset_build_context or nullcontext():
            processor = self.processor_config.build(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
            )
            if self.chat_template is not None and processor is not None:
                chat_template = _resolve_chat_template(self.chat_template)
                processor.chat_template = chat_template
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError("A processor tokenizer is required to apply dataset.chat_template")
                tokenizer.chat_template = chat_template

            if isinstance(self.dataset_config, RankedVlmDatasetConfig):
                dataset = self.dataset_config.build(rank=dp_rank, world_size=dp_world_size)
            elif isinstance(self.dataset_config, TokenizerDatasetConfig):
                dataset = self.dataset_config.build(tokenizer=processor)
            else:
                dataset = self.dataset_config.build()
        return dataset, processor

    def build(
        self,
        *,
        pretrained_model_name_or_path: str,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        dataset_build_context: AbstractContextManager[object] | None = None,
        get_rope_index: Callable[..., object] | None = None,
        packing_attn_implementation: str | None = None,
        pp_n_microbatches: int | None = None,
        cp_size: int = 1,
    ) -> VlmDataloaderBuild:
        """Build the processor, dataset wrappers, sampler, collator, and dataloader.

        Args:
            pretrained_model_name_or_path: Runtime model identifier used to build the processor.
            dp_rank: Rank within the data-parallel group.
            dp_world_size: Size of the data-parallel group.
            batch_size: Runtime local training batch size.
            dataset_build_context: Optional rank-ordering context used only for processor and source-dataset build.
            get_rope_index: Optional model callback used to create packed multimodal position IDs.
            packing_attn_implementation: Resolved attention backend for packed-mask construction.
            pp_n_microbatches: Optional pipeline microbatch count used to pre-chunk media tensors.
            cp_size: Runtime context-parallel world size. Neat-packed CP uses
                compact document IDs instead of a dense quadratic attention mask.

        Returns:
            Named result containing the stateful dataloader and runtime processor.
        """
        dataset, processor = self._build_source(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            dataset_build_context=dataset_build_context,
        )
        raw_dataset = dataset

        if self.pretokenization is not None:
            if processor is None:
                raise ValueError("VLM pretokenization requires a processor")
            dataset = self.pretokenization.build(dataset=dataset, processor=processor)

        if self.packing is not None:
            if self.pretokenization is None:
                raise ValueError("VLM neat packing requires pretokenization")
            tokenizer = getattr(processor, "tokenizer", None)
            padding_idx = getattr(tokenizer, "pad_token_id", 0) or 0
            dataset = self.packing.build(
                dataset=dataset,
                padding_idx=padding_idx,
                ds_raw=raw_dataset,
                get_rope_index=get_rope_index,
                processor=processor,
            )
            if self.packing.packing_format == "thd":
                logger.info("Configured VLM THD packing (Transformer Engine, qkv_format=thd)")
                collate_fn = partial(
                    packed_sequence_thd_vlm_collater,
                    padding_idx=padding_idx,
                    max_length=self.packing.collate_max_length,
                )
            else:
                materialize_4d_mask = cp_size <= 1
                if not materialize_4d_mask:
                    logger.info(
                        "Skipping the dense packed VLM attention mask at cp_size=%d; "
                        "the CP path rebuilds it from compact document IDs",
                        cp_size,
                    )
                collate_fn = partial(
                    neat_packed_vlm_collater,
                    padding_idx=padding_idx,
                    max_length=self.packing.collate_max_length,
                    attn_implementation=packing_attn_implementation,
                    materialize_4d_mask=materialize_4d_mask,
                )
        elif self.collator is not None:
            collate_fn = self.collator.build(processor=processor)
        elif self.pretokenization is not None:
            collate_fn = partial(pad_collate_fn, processor=processor)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                logger.warning("Using %s with the default VLM collator", processor_type)
                processor_type = "default"
            collate_fn = partial(COLLATE_FNS[processor_type], processor=processor)

        if hasattr(dataset, "robust_collate"):
            collate_fn = dataset.robust_collate(collate_fn)
        if pp_n_microbatches is not None:
            collate_fn = wrap_vlm_collate_for_pp(collate_fn, n_microbatches=pp_n_microbatches)

        if not isinstance(dataset, Sized):
            raise TypeError(f"VLM dataloaders require a sized dataset, got {type(dataset).__name__}")
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=self.shuffle,
        )
        dataloader = StatefulDataLoader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )
        return VlmDataloaderBuild(dataloader=dataloader, processor=processor)


__all__ = [
    "RankedVlmDatasetConfig",
    "VlmCollatorConfig",
    "VlmDataloaderBuild",
    "VlmDataloaderConfig",
    "VlmProcessorConfig",
]
