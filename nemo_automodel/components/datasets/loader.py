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

"""Typed, data-parallel-aware dataloader construction.

- :class:`ParallelAwareDataloader` is a stateful, data-parallel-aware ``DataLoader``: it shards an
  ``IterableDataset`` (or attaches a distributed / length-grouped sampler to a map-style dataset)
  for ``(dp_rank, dp_world_size)`` and inherits ``StatefulDataLoader`` for per-rank checkpoint resume.
- :class:`DataloaderConfig` owns dataset, sampler, packing, and dataloader construction. Runtime-only objects
  such as tokenizers and the rank-ordering context are explicit :meth:`DataloaderConfig.build` arguments.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable, Sized
from contextlib import AbstractContextManager, nullcontext
from copy import copy
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, runtime_checkable

from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.sampler import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

CollateFn = Callable[[list[object]], object]


class PlainDatasetConfig(Protocol):
    """Dataset config whose build contract has no runtime dependencies."""

    def build(self) -> object:
        """Build the configured dataset."""


@dataclass(frozen=True)
class DatasetBuildSchedule:
    """Training schedule values required while materializing schedule-aware datasets."""

    local_batch_size: int
    global_batch_size: int
    max_steps: int | None
    val_check_interval: int | None


@runtime_checkable
class TokenizerDatasetConfig(Protocol):
    """Dataset config whose build contract accepts a runtime tokenizer or processor."""

    accepts_tokenizer: bool

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None") -> object:
        """Build the configured dataset with a runtime tokenizer or processor."""


@runtime_checkable
class ScheduledDatasetConfig(TokenizerDatasetConfig, Protocol):
    """Dataset config that additionally consumes the recipe training schedule."""

    requires_training_schedule: bool

    def build(
        self,
        *,
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None",
        training_schedule: DatasetBuildSchedule,
    ) -> object:
        """Build the configured dataset with tokenizer and schedule dependencies."""


@runtime_checkable
class AllRanksDatasetConfig(Protocol):
    """Dataset config whose build implementation performs collectives across all ranks."""

    builds_on_all_ranks: bool


class BatchSamplerConfig(Protocol):
    """Typed construction contract for a dataset-specific batch sampler."""

    def build(self, *, dataset_len: int, rank: int, world_size: int) -> Sampler[list[int]]:
        """Build a per-rank batch sampler for a materialized dataset."""


DatasetConfig = PlainDatasetConfig | TokenizerDatasetConfig | ScheduledDatasetConfig


def _set_spawn_start_method() -> None:
    """Set the multiprocessing start method to ``spawn`` if not already set."""
    try:
        import torch.multiprocessing as mp

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
    except RuntimeError as exc:
        logger.debug("Multiprocessing start method is already fixed: %s", exc)


def _shard_iterable_dataset(dataset: Any, *, dp_rank: int, dp_world_size: int) -> Any:
    """Shard an ``IterableDataset`` across data-parallel ranks for unique samples.

    Calls the dataset's own ``shard`` or HuggingFace ``split_dataset_by_node``.
    """
    if callable(getattr(dataset, "shard", None)):
        dataset = dataset.shard(dp_world_size, dp_rank)
        logger.info(f"Sharded IterableDataset via dataset.shard: world_size={dp_world_size}, rank={dp_rank}")
    elif hasattr(dataset, "dataset"):
        from datasets.distributed import split_dataset_by_node

        dataset = copy(dataset)
        dataset.dataset = split_dataset_by_node(dataset.dataset, world_size=dp_world_size, rank=dp_rank)
        logger.info(f"Sharded dataset via split_dataset_by_node: world_size={dp_world_size}")
    else:
        logger.warning("IterableDataset does not support sharding; Data may be duplicated across ranks.")
    return dataset


@dataclass
class PackingConfig:
    """Base config for sequence packing; ``None`` (no config) means no packing.

    Subclasses (:class:`ThdPackingConfig` / :class:`NeatPackingConfig`) pick the packing strategy and the
    matching collater. :meth:`build` returns ``(dataset, collate_fn)`` — the packed dataset and the
    packing-specific collater. Construction-time knobs are the fields; runtime / model-derived values
    (``split`` / ``seed`` / ``supports_seq_lens`` / ``pad_token_id`` / ``cp_size`` / ``attn_implementation``)
    are :meth:`build` args.
    """

    packed_sequence_size: int
    max_packs: int | None = None
    prepacked: bool = False
    """Whether the dataset already contains packed samples and must not be repacked."""
    requires_model_configuration: ClassVar[bool] = False
    """Whether model-side packing hooks must be configured before building the dataset."""

    def build(
        self,
        dataset: object,
        *,
        split: str | list[str] | None = None,
        seed: int = 42,
        supports_seq_lens: bool = True,
        pad_token_id: int = 0,
        cp_size: int = 1,
        attn_implementation: str | None = None,
    ) -> tuple[object, CollateFn | None]:
        """Pack ``dataset`` and return ``(dataset, collate_fn)``."""
        raise NotImplementedError


@dataclass
class ThdPackingConfig(PackingConfig):
    """THD (flattened, ``seq_lens``-based) packing; pairs with ``packed_sequence_thd_collater``.

    Requires a model whose forward accepts ``seq_lens`` — packing is skipped (with a warning) otherwise.
    """

    def build(
        self,
        dataset: object,
        *,
        split: str | list[str] | None = None,
        seed: int = 42,
        supports_seq_lens: bool = True,
        pad_token_id: int = 0,
        cp_size: int = 1,
        attn_implementation: str | None = None,
    ) -> tuple[object, CollateFn | None]:
        """Pack with THD; returns ``(dataset, None)`` if the model does not accept ``seq_lens``."""
        del attn_implementation
        if not supports_seq_lens:
            logger.warning("Packed sequence is not supported without seq_lens; disabling packed sequence")
            return dataset, None
        from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
        from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

        logger.info(f"THD-packing dataset with size: {self.packed_sequence_size}")
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed)
        dataset = pack_dataset(
            dataset,
            split=split,
            packed_sequence_size=self.packed_sequence_size,
            max_packs=self.max_packs,
            padding_idx=pad_token_id,
            cp_size=cp_size,
        )
        return dataset, packed_sequence_thd_collater


@dataclass
class NeatPackingConfig(PackingConfig):
    """NEAT (bin-packed) packing paired with ``neat_packed_collater``."""

    requires_model_configuration: ClassVar[bool] = True
    drop_long_samples: bool = True

    def build(
        self,
        dataset: object,
        *,
        split: str | list[str] | None = None,
        seed: int = 42,
        supports_seq_lens: bool = True,
        pad_token_id: int = 0,
        cp_size: int = 1,
        attn_implementation: str | None = None,
    ) -> tuple[object, CollateFn]:
        """Pack with NEAT and configure the collator for the selected attention implementation."""
        del supports_seq_lens, cp_size
        from nemo_automodel.components.datasets.llm.neat_packing import neat_pack_dataset
        from nemo_automodel.components.datasets.utils import neat_packed_collater

        logger.info(f"NEAT-packing dataset with size: {self.packed_sequence_size}")
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed)
        dataset = neat_pack_dataset(
            dataset,
            split=split,
            pack_size=self.packed_sequence_size,
            max_packs=self.max_packs,
            padding_idx=pad_token_id,
            drop_long_samples=self.drop_long_samples,
        )
        return dataset, partial(neat_packed_collater, attn_implementation=attn_implementation)


_PACKING_CONFIGS: dict[str, type[PackingConfig]] = {
    "thd": ThdPackingConfig,
    "neat": NeatPackingConfig,
}
_LEGACY_PACKING_FIELDS = {"split_across_pack"}


def _resolve_target(target: Any, registry: dict[str, Any]) -> Any:
    """Resolve ``target`` via ``registry``, a dotted import path, or pass it through unchanged.

    ``target`` may be a key in ``registry`` (whose value is the object itself or a dotted path to it), a
    dotted import path (``"pkg.mod.attr"``), or an already-resolved object (returned as-is). This backs the
    ``make_*`` resolvers below so packing configs, collate fns and sampler factories share one lookup.
    """
    if not isinstance(target, str):
        return target
    entry = registry.get(target, target)
    if not isinstance(entry, str):
        return entry
    if "." not in entry:
        raise ValueError(f"Unknown target {target!r}; expected one of {sorted(registry)} or a dotted path")
    import importlib

    module_path, _, attr = entry.rpartition(".")
    return getattr(importlib.import_module(module_path), attr)


def make_packing_config(target: str | None, kwargs: dict[str, object] | None = None) -> PackingConfig | None:
    """Resolve a packing-config ``target`` and construct it from ``kwargs`` (``target=None`` → no packing).

    ``target`` is either a built-in strategy key (``"thd"`` / ``"neat"``) or a dotted import path to
    a :class:`PackingConfig` subclass (e.g. ``"my_pkg.MyPackingConfig"``). Strategy-specific fields from the
    union-shaped ``packed_sequence`` YAML block are accepted and filtered for the selected strategy; unknown
    fields are rejected instead of silently disappearing.
    """
    if not target:
        return None
    cls = _resolve_target(target, _PACKING_CONFIGS)
    kwargs = kwargs or {}
    valid = {f.name for f in fields(cls)}
    union_fields = {f.name for config_cls in _PACKING_CONFIGS.values() for f in fields(config_cls)}
    unknown = sorted(set(kwargs) - valid - union_fields - _LEGACY_PACKING_FIELDS)
    if unknown:
        raise TypeError(f"{cls.__name__} got unexpected packing config field(s): {', '.join(unknown)}")
    legacy = sorted(set(kwargs) & _LEGACY_PACKING_FIELDS)
    if legacy:
        warnings.warn(
            f"Packing config field(s) {', '.join(legacy)} are no longer supported and have no effect",
            FutureWarning,
            stacklevel=2,
        )
    return cls(**{k: v for k, v in kwargs.items() if k in valid})


_COLLATE_FNS: dict[str, str] = {
    "default": "nemo_automodel.components.datasets.utils.default_collater",
}


@dataclass
class CollatorConfig:
    """Construction-time configuration for a tokenizer-aware collator class."""

    factory: Callable[..., CollateFn]
    kwargs: dict[str, object] = field(default_factory=dict)

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | ProcessorMixin") -> CollateFn:
        """Instantiate the collator once with its runtime tokenizer or processor.

        Args:
            tokenizer: Runtime tokenizer or multimodal processor used by the collator.

        Returns:
            Callable collator passed to the dataloader.
        """
        collator = self.factory(tokenizer=tokenizer, **self.kwargs)
        if not callable(collator):
            raise TypeError(f"Collator factory {self.factory!r} returned non-callable {type(collator).__name__}")
        return collator


def make_collate_fn(target: object, kwargs: dict[str, object] | None = None) -> CollateFn | CollatorConfig | None:
    """Resolve a collate target into a batch callable or tokenizer-aware config.

    ``target`` is a built-in collator key (for example ``"default"``), a dotted import path, or an already
    resolved callable. Collator classes become :class:`CollatorConfig` so :meth:`DataloaderConfig.build`
    instantiates them once with the runtime tokenizer. Function kwargs remain partial-bound per batch.
    """
    if target is None:
        return None
    fn = _resolve_target(target, _COLLATE_FNS)
    if inspect.isclass(fn):
        return CollatorConfig(factory=fn, kwargs=kwargs or {})
    return partial(fn, **kwargs) if kwargs else fn


@dataclass
class _LegacyDatasetConfig:
    """Fallback shim for a dataset ``_target_`` that has no typed ``<Name>Config``.

    :func:`make_dataset_config` maps known datasets onto their typed config; the few
    targets with no config yet (a handful of VLM ``make_*`` factories) land here instead. Wraps the target so
    it satisfies the typed dataset build contract. Config resolution records whether the external target accepts
    a tokenizer, and :meth:`build` forwards only that explicit runtime dependency.
    """

    factory: Callable[..., object]
    kwargs: dict[str, object]
    accepts_tokenizer: bool = False

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None" = None) -> object:
        """Call the wrapped dataset target with its declared YAML arguments.

        Args:
            tokenizer: Runtime tokenizer or processor. It is forwarded only when the target explicitly declares
                a ``tokenizer`` parameter; that compatibility decision is recorded once at config resolution.

        Returns:
            Dataset object returned by the configured target.
        """
        kwargs = dict(self.kwargs)
        if self.accepts_tokenizer:
            kwargs["tokenizer"] = tokenizer
        return self.factory(**kwargs)


_DATASETS = "nemo_automodel.components.datasets"
_DATASET_CONFIGS: dict[str, str] = {
    # Exact legacy dataset targets -> typed config targets. Exact paths avoid class-name collisions while
    # preserving current YAMLs that still name the pre-config class or factory. Targets that already name a
    # config fall through to direct import; unknown external targets use the compatibility adapter below.
    f"{_DATASETS}.llm.megatron_dataset.MegatronPretraining": (
        f"{_DATASETS}.llm.megatron_dataset.MegatronPretrainingConfig"
    ),
    f"{_DATASETS}.llm.squad.make_squad_dataset": f"{_DATASETS}.llm.squad.SquadConfig",
    f"{_DATASETS}.llm.hellaswag.HellaSwag": f"{_DATASETS}.llm.hellaswag.HellaSwagConfig",
    f"{_DATASETS}.llm.chat_dataset.ChatDataset": f"{_DATASETS}.llm.chat_dataset.ChatDatasetConfig",
    f"{_DATASETS}.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset": (
        f"{_DATASETS}.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDatasetConfig"
    ),
    f"{_DATASETS}.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset": (
        f"{_DATASETS}.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDatasetConfig"
    ),
    f"{_DATASETS}.llm.delta_lake_dataset.DeltaLakeDataset": f"{_DATASETS}.llm.delta_lake_dataset.DeltaLakeDatasetConfig",
    f"{_DATASETS}.llm.mock_iterable_dataset.MockIterableDataset": (
        f"{_DATASETS}.llm.mock_iterable_dataset.MockIterableDatasetConfig"
    ),
    f"{_DATASETS}.llm.mock_seq_cls.MockSequenceClassificationDataset": (
        f"{_DATASETS}.llm.mock_seq_cls.MockSequenceClassificationDatasetConfig"
    ),
    f"{_DATASETS}.llm.nanogpt_dataset.NanogptDataset": f"{_DATASETS}.llm.nanogpt_dataset.NanogptDatasetConfig",
    f"{_DATASETS}.llm.agent_chat.make_agent_chat_dataset": f"{_DATASETS}.llm.agent_chat.AgentChatConfig",
    f"{_DATASETS}.llm.xlam.make_xlam_dataset": f"{_DATASETS}.llm.xlam.XlamConfig",
    f"{_DATASETS}.llm.seq_cls.GLUE_MRPC": f"{_DATASETS}.llm.seq_cls.GLUE_MRPCConfig",
    f"{_DATASETS}.llm.mock.build_unpacked_dataset": f"{_DATASETS}.llm.mock.MockUnpackedDatasetConfig",
    f"{_DATASETS}.llm.mock_packed.build_packed_dataset": f"{_DATASETS}.llm.mock_packed.MockPackedDatasetConfig",
    f"{_DATASETS}.llm.retrieval_dataset.make_retrieval_dataset": (
        f"{_DATASETS}.llm.retrieval_dataset.RetrievalDatasetConfig"
    ),
    f"{_DATASETS}.llm.retrieval_dataset_inline.make_retrieval_dataset": (
        f"{_DATASETS}.llm.retrieval_dataset_inline.InlineRetrievalDatasetConfig"
    ),
    f"{_DATASETS}.vlm.datasets.make_rdr_dataset": f"{_DATASETS}.vlm.datasets.RdrDatasetConfig",
    f"{_DATASETS}.vlm.datasets.make_cord_v2_dataset": f"{_DATASETS}.vlm.datasets.CordV2DatasetConfig",
    f"{_DATASETS}.vlm.datasets.make_medpix_dataset": f"{_DATASETS}.vlm.datasets.MedPixDatasetConfig",
    f"{_DATASETS}.vlm.datasets.make_llava_onevision_dataset": (f"{_DATASETS}.vlm.datasets.LlavaOnevisionDatasetConfig"),
    f"{_DATASETS}.vlm.datasets.make_tulu3_magicoder_text_mix_dataset": (
        f"{_DATASETS}.vlm.datasets.Tulu3MagicoderTextMixDatasetConfig"
    ),
    f"{_DATASETS}.vlm.datasets.make_unimm_chat_dataset": f"{_DATASETS}.vlm.datasets.UnimmChatDatasetConfig",
    f"{_DATASETS}.vlm.datasets.make_meta_dataset": f"{_DATASETS}.vlm.datasets.MetaDatasetConfig",
    f"{_DATASETS}.vlm.mock.build_mock_vlm_dataset": f"{_DATASETS}.vlm.mock.MockVlmDatasetConfig",
    f"{_DATASETS}.audio.datasets.make_cv17_dataset": f"{_DATASETS}.audio.datasets.Cv17DatasetConfig",
}


def make_dataset_config(target: object, kwargs: dict[str, object] | None = None) -> DatasetConfig:
    """Resolve a dataset ``_target_`` to an object exposing a typed ``build`` method.

    Exact legacy registrations map a pre-config dataset class or ``make_*`` factory onto its typed
    ``<Name>Config`` without class-name dispatch. Misses fall back to the ``_target_`` itself, so a target that
    already names a config imports directly. Typed config fields are validated before construction; external
    factories use :class:`_LegacyDatasetConfig`.
    """
    kwargs = kwargs or {}
    if isinstance(target, str):
        target_path = target
    else:
        module = getattr(target, "__module__", None)
        name = getattr(target, "__qualname__", getattr(target, "__name__", None))
        target_path = f"{module}.{name}" if module and name else None
    config_target = _DATASET_CONFIGS.get(target_path, target)
    obj = _resolve_target(config_target, {})
    if is_dataclass(obj) and hasattr(obj, "build"):
        valid = {f.name for f in fields(obj)}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            raise TypeError(f"{obj.__name__} got unexpected dataset config field(s): {', '.join(unknown)}")
        return cast(DatasetConfig, obj(**kwargs))
    parameters = inspect.signature(obj).parameters
    accepts_tokenizer = "tokenizer" in parameters
    return _LegacyDatasetConfig(obj, kwargs, accepts_tokenizer=accepts_tokenizer)


def _make_sampler(
    dataset: Any,
    *,
    dp_rank: int,
    dp_world_size: int,
    seed: int,
    shuffle: bool,
    group_by_length: bool,
    batch_size: int,
) -> Sampler:
    """Build the default map-style sampler (distributed, or length-grouped)."""
    if group_by_length:
        from nemo_automodel.components.datasets.llm.length_grouped_sampler import LengthGroupedSampler

        return LengthGroupedSampler(
            dataset=dataset, batch_size=batch_size, seed=seed, num_replicas=dp_world_size, rank=dp_rank
        )
    return StatefulDistributedSampler(
        dataset, seed=seed, drop_last=True, num_replicas=dp_world_size, rank=dp_rank, shuffle=shuffle
    )


class ParallelAwareDataloader(StatefulDataLoader):
    """Stateful, data-parallel-aware ``DataLoader``.

    Routes a dataset to ``(dp_rank, dp_world_size)``: an ``IterableDataset`` is sharded (and optionally
    buffer-shuffled); a map-style dataset gets the default distributed (or length-grouped) sampler with
    ``batch_size`` and ``drop_last`` under PP. Inherits ``StatefulDataLoader`` for per-rank checkpoint resume.
    """

    def __init__(
        self,
        dataset: object,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int | None = 1,
        collate_fn: CollateFn | None = None,
        batch_sampler: Sampler[list[int]] | None = None,
        seed: int = 42,
        shuffle: bool | None = None,
        group_by_length: bool = False,
        pp_enabled: bool = False,
        shuffle_buffer_size: int = 10000,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        drop_last: bool = False,
    ) -> None:
        """Shard / sample ``dataset`` for the given ranks, then construct the ``StatefulDataLoader``."""
        loader_kwargs: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        elif isinstance(dataset, IterableDataset):
            dataset = _shard_iterable_dataset(dataset, dp_rank=dp_rank, dp_world_size=dp_world_size)
            if shuffle and hasattr(dataset, "shuffle"):
                dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
                logger.info("Shuffling IterableDataset with buffer_size=%s", shuffle_buffer_size)
            loader_kwargs["batch_size"] = batch_size
        else:
            if batch_size is None:
                raise ValueError("batch_size=None is only supported for iterable or explicitly batch-sampled datasets")
            loader_kwargs["sampler"] = _make_sampler(
                dataset,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                seed=seed,
                shuffle=True if shuffle is None else shuffle,
                group_by_length=group_by_length,
                batch_size=batch_size,
            )
            loader_kwargs["batch_size"] = batch_size
            loader_kwargs["drop_last"] = pp_enabled or drop_last

        _set_spawn_start_method()
        super().__init__(dataset, collate_fn=collate_fn, **loader_kwargs)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size


@dataclass
class DataloaderConfig:
    """A typed dataset config + loader settings; :meth:`build` produces a :class:`ParallelAwareDataloader`.

    ``dataset_config`` is a typed per-dataset config (for example ``ChatDatasetConfig`` or
    ``GLUE_MRPCConfig``); the dataset's declarative arguments live there. Runtime dependencies are explicit
    :meth:`build` arguments, while supported ``DataLoader`` settings are named fields on this config.
    """

    dataset_config: DatasetConfig
    packing: PackingConfig | None = None
    batch_sampler_config: BatchSamplerConfig | None = None
    dataset_build_schedule: DatasetBuildSchedule | None = None
    shuffle: bool | None = None
    group_by_length: bool = False
    shuffle_buffer_size: int = 10000
    batch_size: int | None = 1
    seed: int = 42
    collate_fn: CollateFn | CollatorConfig | None = None
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False

    @property
    def dataset_builds_on_all_ranks(self) -> bool:
        """Whether dataset construction must bypass rank-zero-first ordering."""
        return isinstance(self.dataset_config, AllRanksDatasetConfig)

    def _build_dataset(
        self,
        *,
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None",
        dataset_build_context: AbstractContextManager[object] | None,
    ) -> object:
        """Materialize only the dataset inside the caller-provided ordering context."""
        with dataset_build_context or nullcontext():
            if isinstance(self.dataset_config, ScheduledDatasetConfig):
                if self.dataset_build_schedule is None:
                    raise ValueError(
                        f"{type(self.dataset_config).__name__} requires step-scheduler values to build the dataset"
                    )
                return self.dataset_config.build(
                    tokenizer=tokenizer,
                    training_schedule=self.dataset_build_schedule,
                )
            if isinstance(self.dataset_config, TokenizerDatasetConfig):
                return self.dataset_config.build(tokenizer=tokenizer)
            return self.dataset_config.build()

    def build(
        self,
        *,
        dp_rank: int,
        dp_world_size: int,
        pp_enabled: bool = False,
        tokenizer: "PreTrainedTokenizerBase | ProcessorMixin | None" = None,
        dataset_build_context: AbstractContextManager[object] | None = None,
        supports_seq_lens: bool = True,
        cp_size: int = 1,
        attn_implementation: str | None = None,
        collate_wrapper: Callable[[CollateFn], CollateFn] | None = None,
    ) -> DataLoader:
        """Build the configured dataset, packing, sampler, collator, and stateful dataloader.

        Args:
            dp_rank: Rank within the data-parallel group.
            dp_world_size: Size of the data-parallel group.
            pp_enabled: Whether pipeline parallelism requires dropping incomplete batches.
            tokenizer: Runtime tokenizer or multimodal processor for tokenizer-aware datasets and collators.
            dataset_build_context: Optional caller-owned rank-ordering context used only while materializing the
                dataset. Collective dataset builders should pass ``None``.
            supports_seq_lens: Whether the model forward contract accepts THD ``seq_lens`` metadata.
            cp_size: Context-parallel world size used for packed-sequence divisibility.
            attn_implementation: Attention backend used by NEAT packing.
            collate_wrapper: Optional recipe-owned wrapper around the resolved collator.

        Returns:
            Stateful, data-parallel-aware dataloader.
        """
        dataset = self._build_dataset(tokenizer=tokenizer, dataset_build_context=dataset_build_context)

        collate_override = None
        if self.packing is not None and not self.packing.prepacked:
            dataset, collate_override = self.packing.build(
                dataset,
                split=getattr(self.dataset_config, "split", None),
                seed=self.seed,
                supports_seq_lens=supports_seq_lens,
                pad_token_id=getattr(tokenizer, "pad_token_id", 0),
                cp_size=cp_size,
                attn_implementation=attn_implementation,
            )
        elif self.packing is not None:
            logger.info(
                "Using prepacked sequence dataset with size %s; skipping loader-side packing",
                self.packing.packed_sequence_size,
            )

        # Collate: packing override -> custom collate_fn -> shared default; then the optional PP wrapper.
        collate_fn = collate_override or self.collate_fn
        if isinstance(collate_fn, CollatorConfig):
            if tokenizer is None:
                raise ValueError("A tokenizer or processor is required to build the configured collator")
            collate_fn = collate_fn.build(tokenizer=tokenizer)
        if collate_fn is None:
            if self.batch_size is None:
                from torch.utils.data import default_convert

                collate_fn = default_convert
            else:
                from nemo_automodel.components.datasets.utils import default_collater

                collate_fn = default_collater
        if collate_wrapper is not None:
            collate_fn = collate_wrapper(collate_fn)

        batch_sampler = None
        if self.batch_sampler_config is not None:
            if not isinstance(dataset, Sized):
                raise TypeError(
                    f"{type(self.batch_sampler_config).__name__} requires a sized dataset, got {type(dataset).__name__}"
                )
            batch_sampler = self.batch_sampler_config.build(
                dataset_len=len(dataset), rank=dp_rank, world_size=dp_world_size
            )

        return ParallelAwareDataloader(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            seed=self.seed,
            shuffle=self.shuffle,
            group_by_length=self.group_by_length,
            pp_enabled=pp_enabled,
            shuffle_buffer_size=self.shuffle_buffer_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=self.drop_last,
        )


__all__ = [
    "CollatorConfig",
    "DatasetBuildSchedule",
    "DataloaderConfig",
    "NeatPackingConfig",
    "PackingConfig",
    "ParallelAwareDataloader",
    "ThdPackingConfig",
    "make_collate_fn",
    "make_dataset_config",
    "make_packing_config",
]
