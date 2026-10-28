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
import os
from collections.abc import Callable, Sized
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.stateful import Stateful
from transformers import AutoProcessor, ProcessorMixin

from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template
from nemo_automodel.components.datasets.loader import DatasetConfig, TokenizerDatasetConfig
from nemo_automodel.components.datasets.packing import (
    DEFAULT_PACKED_SEQUENCE_CONTRACT,
    BatchOutputFormat,
    PackedSequenceContract,
    resolve_packing_contract,
)
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
_VLM_PARENT_DATASET_STATE = "_vlm_parent_dataset_state"


class _VlmStatefulDataloader(StatefulDataLoader):
    """Restore VLM checkpoints across synchronous and worker-based loaders."""

    _parent_dataset: Stateful | None = None

    def _worker_states(self, dataset_state: Any) -> dict[str, Any]:
        """Give each worker an independently owned copy of common map-style state."""
        return {
            f"worker_{worker_id}": {
                "worker_id": worker_id,
                "dataset_state": deepcopy(dataset_state),
                "fetcher_state": None,
            }
            for worker_id in range(self.num_workers)
        }

    def _restore_parent_dataset_state(self, dataset_state: Any) -> Stateful | None:
        """Restore parent state only when it originated from synchronous loading."""
        if dataset_state is not None and isinstance(self.dataset, Stateful):
            self.dataset.load_state_dict(deepcopy(dataset_state))
            return self.dataset
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return native worker snapshots and any migrated synchronous parent state."""
        state = dict(super().state_dict())
        if self._parent_dataset is not None:
            state[_VLM_PARENT_DATASET_STATE] = deepcopy(self._parent_dataset.state_dict())
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore native worker state or migrate legacy map-style synchronous state.

        Worker-count changes require stateless map-style worker snapshots.
        Legacy synchronous snapshots carry sampler/dataset state at the top
        level; give each worker that state and continue the sampler.
        Worker-side random transforms cannot reproduce the old main-process RNG.
        """
        if not state_dict:
            super().load_state_dict(state_dict)
            return
        parent_dataset = None
        if "_snapshot" in state_dict:
            snapshot = state_dict["_snapshot"]
            saved_workers = snapshot["_main_snapshot"]["_num_workers"]
            worker_states = list(snapshot["_worker_snapshots"].values())
            stateless_workers = (
                bool(worker_states)
                and len(worker_states) == saved_workers
                and all(worker["fetcher_state"] is None and worker["dataset_state"] is None for worker in worker_states)
            )
            if saved_workers != self.num_workers:
                if not stateless_workers:
                    raise ValueError(
                        f"Cannot change VLM checkpoint num_workers from {saved_workers} to {self.num_workers}: "
                        "worker snapshots must contain stateless map-style datasets. "
                        f"Restore with num_workers={saved_workers}."
                    )
                state_dict = dict(state_dict)
                snapshot = state_dict["_snapshot"] = dict(snapshot)
                snapshot["_main_snapshot"] = deepcopy(snapshot["_main_snapshot"])
                snapshot["_main_snapshot"]["_num_workers"] = self.num_workers
                snapshot["_last_yielded_worker_id"] = (snapshot["_snapshot_step"] - 1) % self.num_workers
                snapshot["_worker_snapshots"] = self._worker_states(None)
                logger.warning(
                    "Migrating VLM dataloader checkpoint from num_workers=%d to num_workers=%d.",
                    saved_workers,
                    self.num_workers,
                )
            if _VLM_PARENT_DATASET_STATE in state_dict:
                parent_dataset = self._restore_parent_dataset_state(state_dict[_VLM_PARENT_DATASET_STATE])
        elif "_num_yielded" in state_dict:
            if state_dict["fetcher_state"] is not None:
                raise ValueError("VLM workers can migrate only map-style synchronous dataloader checkpoints")
            legacy = deepcopy(state_dict)
            parent_dataset = self._restore_parent_dataset_state(legacy["dataset_state"])
            main_state = {
                key: legacy[key]
                for key in (
                    "_index_sampler_state",
                    "_sampler_iter_state",
                    "_sampler_iter_yielded",
                    "_IterableDataset_len_called",
                    "_shared_seed",
                )
            }
            # Choose a worker seed without advancing the training RNG. Recipes
            # restore training RNG after the dataloader, so a fresh seed may differ.
            generator = torch.Generator()
            generator.set_state(self.generator.get_state() if self.generator is not None else torch.get_rng_state())
            main_state["_base_seed"] = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
            main_state["_num_workers"] = self.num_workers
            state_dict = {
                "_snapshot": {
                    "_snapshot_step": legacy["_num_yielded"],
                    "_last_yielded_worker_id": (legacy["_num_yielded"] - 1) % self.num_workers,
                    "_main_snapshot": main_state,
                    "_worker_snapshots": self._worker_states(legacy["dataset_state"]),
                },
                "_steps_since_snapshot": 0,
                "_iterator_finished": legacy["_iterator_finished"],
            }
            logger.warning("Migrating a synchronous VLM dataloader checkpoint to num_workers=%d.", self.num_workers)
        super().load_state_dict(state_dict)
        self._parent_dataset = parent_dataset


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
    """Worker processes per dataloader; zero selects available CPUs, capped at 4 * visible GPUs when present."""
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False

    @property
    def is_packed(self) -> bool:
        """Whether this loader emits packed multimodal inputs."""
        return self.packing is not None

    @property
    def output_format(self) -> BatchOutputFormat:
        """Selected native batch representation, independent of the packing algorithm."""
        return "thd" if self.packing is not None and self.packing.packing_format == "thd" else "batched"

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
                # These configs are text datasets (e.g. ChatDataset) that consume a tokenizer,
                # not the full processor: a ProcessorMixin does not proxy tokenizer attributes
                # (eos_token_id, pad_token, ...) -- those live on processor.tokenizer. Forward the
                # wrapped tokenizer so both a genuine processor and a bare tokenizer work.
                source_tokenizer = processor.tokenizer if isinstance(processor, ProcessorMixin) else processor
                dataset = self.dataset_config.build(tokenizer=source_tokenizer)
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
        packing_contract: PackedSequenceContract = DEFAULT_PACKED_SEQUENCE_CONTRACT,
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
            packing_attn_implementation: Deprecated attention-backend name
                retained for Python-call compatibility.
            packing_contract: Structural model contract used for packed-mask construction.
                Defaults to block-causal masking without packed-sequence metadata.
            pp_n_microbatches: Optional pipeline microbatch count used to pre-chunk media tensors.
            cp_size: Runtime context-parallel world size. Neat-packed CP uses
                compact document IDs instead of a dense quadratic attention mask;
                THD packing also accounts for per-document CP alignment.

        Returns:
            Named result containing the stateful dataloader and runtime processor.
        """
        if self.num_workers < 0:
            raise ValueError("VLM num_workers must be zero (automatic workers) or a positive worker count.")
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
            legacy_attn_implementation = packing_attn_implementation or self.packing.attn_implementation
            packing_contract = resolve_packing_contract(packing_contract, legacy_attn_implementation)
            tokenizer = getattr(processor, "tokenizer", None)
            padding_idx = getattr(tokenizer, "pad_token_id", 0) or 0
            dataset = self.packing.build(
                dataset=dataset,
                padding_idx=padding_idx,
                ds_raw=raw_dataset,
                get_rope_index=get_rope_index,
                processor=processor,
                cp_size=cp_size,
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
                    packing=packing_contract,
                    materialize_4d_mask=materialize_4d_mask,
                    defer_flash_metadata=pp_n_microbatches is not None,
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
        num_workers = self.num_workers
        if num_workers == 0:
            try:
                available_cpus = len(os.sched_getaffinity(0))
            except (AttributeError, OSError, NotImplementedError):
                available_cpus = None
            num_cpus = available_cpus or os.cpu_count() or 1
            num_gpus = torch.cuda.device_count()
            num_workers = min(4 * num_gpus, num_cpus) if num_gpus > 0 else num_cpus
            logger.warning(
                "VLM dataloader num_workers=0 selects automatic preprocessing and prefetching; "
                "using num_workers=%d with %d visible GPUs and %d available CPUs "
                "(capped at 4 workers per visible GPU when GPUs are available). "
                "This count is per dataloader; set an explicit positive num_workers "
                "to override the automatic selection.",
                num_workers,
                num_gpus,
                num_cpus,
            )
        dataloader = _VlmStatefulDataloader(
            dataset=dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and num_workers > 0,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
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
