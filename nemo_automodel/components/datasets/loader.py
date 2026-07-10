# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Config-driven dataloader construction — the replacement for the recipe's ``build_dataloader``.

- :class:`ParallelAwareDataloader` is a stateful, data-parallel-aware ``DataLoader``: it shards an
  ``IterableDataset`` (or attaches a distributed / length-grouped sampler to a map-style dataset)
  for ``(dp_rank, dp_world_size)`` and inherits ``StatefulDataLoader`` for per-rank checkpoint resume.
- :class:`DataloaderConfig` holds a dataset **factory + kwargs** (like ``OptimizerConfig`` holds its
  ``factory`` + ``kwargs``) plus loader/packing settings. :meth:`DataloaderConfig.build` seeds construction
  (``ScopedRNG``), materializes the dataset rank-0-first (``FirstRankPerNode``), forwarding runtime objects
  (``tokenizer`` / ``processor`` / ...) that the dataset's constructor accepts, packs, resolves the collate
  fn, and returns a :class:`ParallelAwareDataloader`. ``build_dataloader_config`` assembles the config from a
  resolved factory + plain dicts (mirrors ``build_optimizer_config``).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

if TYPE_CHECKING:
    from torch.utils.data import Sampler
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)


def _set_spawn_start_method() -> None:
    """Set the multiprocessing start method to ``spawn`` if not already set."""
    try:
        import torch.multiprocessing as mp

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _filter_runtime(fn: Callable, runtime: dict[str, Any]) -> dict[str, Any]:
    """Keep only the ``runtime`` keys that ``fn`` accepts (pass all if ``fn`` takes ``**kwargs``)."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return runtime
    return {k: v for k, v in runtime.items() if k in params}


def _model_supports_seq_lens(model: Any) -> bool:
    """True when ``model.forward()`` accepts a ``seq_lens`` kwarg (or ``**kwargs``).

    Inlined from ``_transformers.capabilities`` so the ``datasets`` component does not import another
    component (the import linter forbids cross-component dependencies, and importing ``capabilities``
    transitively pulls in ``components.distributed``).
    """
    fwd = getattr(model, "forward", None)
    if not callable(fwd):
        return False
    try:
        params = inspect.signature(fwd).parameters
        if "seq_lens" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False


def _shard_iterable_dataset(dataset: Any, *, dp_rank: int, dp_world_size: int) -> Any:
    """Shard an ``IterableDataset`` across data-parallel ranks for unique samples.

    Calls the dataset's own ``shard`` or HuggingFace ``split_dataset_by_node``.
    """
    if callable(getattr(dataset, "shard", None)):
        dataset = dataset.shard(dp_world_size, dp_rank)
        logger.info(f"Sharded IterableDataset via dataset.shard: world_size={dp_world_size}, rank={dp_rank}")
    elif hasattr(dataset, "dataset"):
        from datasets.distributed import split_dataset_by_node

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

    def build(
        self,
        dataset: Any,
        *,
        split: Any = None,
        seed: int = 42,
        supports_seq_lens: bool = True,
        pad_token_id: int = 0,
        cp_size: int = 1,
        attn_implementation: str | None = None,
    ) -> tuple[Any, Callable | None]:
        """Pack ``dataset`` and return ``(dataset, collate_fn)``."""
        raise NotImplementedError


@dataclass
class ThdPackingConfig(PackingConfig):
    """THD (flattened, ``seq_lens``-based) packing; pairs with ``packed_sequence_thd_collater``.

    Requires a model whose forward accepts ``seq_lens`` — packing is skipped (with a warning) otherwise.
    """

    def build(self, dataset, *, split=None, seed=42, supports_seq_lens=True, pad_token_id=0, cp_size=1, **_):
        """Pack with THD; returns ``(dataset, None)`` if the model does not accept ``seq_lens``."""
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
    """NEAT (bin-packed) packing; pairs with ``neat_packed_collater`` and configures the model's attn impl."""

    drop_long_samples: bool = True

    def build(self, dataset, *, split=None, seed=42, pad_token_id=0, attn_implementation=None, **_):
        """Pack with NEAT and tell the model side which attention impl the collater targets."""
        from nemo_automodel.components.datasets.llm.neat_packing import neat_pack_dataset
        from nemo_automodel.components.datasets.utils import neat_packed_collater
        from nemo_automodel.components.models.common.packing import configure_packing

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
        if attn_implementation is not None:
            configure_packing(attn_implementation=attn_implementation)
        return dataset, partial(neat_packed_collater, attn_implementation=attn_implementation)


PACKING_CONFIGS: dict[str, type[PackingConfig]] = {
    "thd": ThdPackingConfig,
    "neat": NeatPackingConfig,
}


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


def make_packing_config(target: str | None, kwargs: dict[str, Any] | None = None) -> PackingConfig | None:
    """Resolve a packing-config ``target`` and construct it from ``kwargs`` (``target=None`` → no packing).

    ``target`` is either a key in :data:`PACKING_CONFIGS` (``"thd"`` / ``"neat"``) or a dotted import path to
    a :class:`PackingConfig` subclass (e.g. ``"my_pkg.MyPackingConfig"``). ``kwargs`` is filtered to the
    resolved class's fields, so the union-of-strategies ``packed_sequence`` YAML block can be passed as-is.
    """
    if not target:
        return None
    cls = _resolve_target(target, PACKING_CONFIGS)
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (kwargs or {}).items() if k in valid})


COLLATE_FNS: dict[str, str] = {
    "default": "nemo_automodel.components.datasets.utils.default_collater",
}


@dataclass
class CollatorConfig:
    """Construction-time configuration for a tokenizer-aware collator class."""

    factory: Callable[..., Callable[..., Any]]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | ProcessorMixin") -> Callable[..., Any]:
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


def make_collate_fn(target: Any, kwargs: dict[str, Any] | None = None) -> Callable[..., Any] | CollatorConfig | None:
    """Resolve a collate target into a batch callable or tokenizer-aware config.

    ``target`` is a key in :data:`COLLATE_FNS` (e.g. ``"default"``), a dotted import path, or an already
    resolved callable. Collator classes become :class:`CollatorConfig` so :meth:`DataloaderConfig.build`
    instantiates them once with the runtime tokenizer. Function kwargs remain partial-bound per batch.
    """
    if target is None:
        return None
    fn = _resolve_target(target, COLLATE_FNS)
    if inspect.isclass(fn):
        return CollatorConfig(factory=fn, kwargs=kwargs or {})
    return partial(fn, **kwargs) if kwargs else fn


@dataclass
class _LegacyDatasetConfig:
    """Fallback shim for a dataset ``_target_`` that has no typed ``<Name>Config``.

    :func:`make_dataset_config` maps known datasets onto their config via :data:`DATASET_CONFIGS`; the few
    targets with no config yet (a handful of VLM ``make_*`` factories) land here instead. Wraps the target so
    it satisfies the ``dataset_config.build(**runtime) -> Dataset`` interface: :meth:`build` calls it with the
    YAML kwargs plus any runtime objects (``tokenizer`` / ``processor`` / ...) its signature accepts.
    """

    factory: Any
    kwargs: dict[str, Any]

    def build(self, **runtime: Any) -> Any:
        """Call the wrapped dataset class / factory with the YAML kwargs + the runtime args it accepts."""
        return self.factory(**self.kwargs, **_filter_runtime(self.factory, runtime))


_DATASETS = "nemo_automodel.components.datasets"
DATASET_CONFIGS: dict[str, str] = {
    # ``dataset._target_.split(".")[-1].lower()`` -> the typed ``<Name>Config`` (lazy dotted). Maps each
    # pre-Config dataset class / ``make_*`` factory onto its config, so an old YAML that points ``_target_`` at
    # the dataset itself constructs the typed config. A ``_target_`` already naming a ``<Name>Config`` is not a
    # key here, so it falls through to a direct import. A target with no config (a few VLM factories) hits the
    # legacy shim. Most entries are keyed by bare name (module-independent), while exact dotted-path entries
    # disambiguate factories that share a name but implement different dataset formats.
    "megatronpretraining": f"{_DATASETS}.llm.megatron_dataset.MegatronPretrainingConfig",
    "make_squad_dataset": f"{_DATASETS}.llm.squad.SquadConfig",
    "hellaswag": f"{_DATASETS}.llm.hellaswag.HellaSwagConfig",
    "chatdataset": f"{_DATASETS}.llm.chat_dataset.ChatDatasetConfig",
    "columnmappedtextinstructiondataset": (
        f"{_DATASETS}.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDatasetConfig"
    ),
    "mockiterabledataset": f"{_DATASETS}.llm.mock_iterable_dataset.MockIterableDatasetConfig",
    "mocksequenceclassificationdataset": f"{_DATASETS}.llm.mock_seq_cls.MockSequenceClassificationDatasetConfig",
    "nanogptdataset": f"{_DATASETS}.llm.nanogpt_dataset.NanogptDatasetConfig",
    "make_agent_chat_dataset": f"{_DATASETS}.llm.agent_chat.AgentChatConfig",
    "make_xlam_dataset": f"{_DATASETS}.llm.xlam.XlamConfig",
    "glue_mrpc": f"{_DATASETS}.llm.seq_cls.GLUE_MRPCConfig",
    "build_unpacked_dataset": f"{_DATASETS}.llm.mock.MockUnpackedDatasetConfig",
    "make_retrieval_dataset": f"{_DATASETS}.llm.retrieval_dataset.RetrievalDatasetConfig",
    f"{_DATASETS}.llm.retrieval_dataset_inline.make_retrieval_dataset": (
        f"{_DATASETS}.llm.retrieval_dataset_inline.InlineRetrievalDatasetConfig"
    ),
    "make_rdr_dataset": f"{_DATASETS}.vlm.datasets.RdrDatasetConfig",
    "make_cord_v2_dataset": f"{_DATASETS}.vlm.datasets.CordV2DatasetConfig",
    "make_unimm_chat_dataset": f"{_DATASETS}.vlm.datasets.UnimmChatDatasetConfig",
    "make_meta_dataset": f"{_DATASETS}.vlm.datasets.MetaDatasetConfig",
    "build_mock_vlm_dataset": f"{_DATASETS}.vlm.mock.MockVlmDatasetConfig",
}


def make_dataset_config(target: Any, kwargs: dict[str, Any] | None = None) -> Any:
    """Resolve a dataset ``_target_`` to an object exposing ``build(**runtime) -> Dataset``.

    Exact dotted-path registrations are checked first, then the ``_target_``'s bare name
    (``split(".")[-1].lower()``). This maps a pre-Config dataset class / ``make_*`` factory onto its typed
    ``<Name>Config`` while allowing same-named factories with different formats. Misses fall back to the
    ``_target_`` itself, so a target that already names a config imports directly. A resolved typed config is
    constructed from ``kwargs`` filtered to its fields; other factories use :class:`_LegacyDatasetConfig`.
    """
    kwargs = kwargs or {}
    if not isinstance(target, str):
        target = f"{target.__module__}.{getattr(target, '__qualname__', getattr(target, '__name__', ''))}"
    config_target = DATASET_CONFIGS.get(target, DATASET_CONFIGS.get(target.split(".")[-1].lower(), target))
    obj = _resolve_target(config_target, {})
    if is_dataclass(obj) and hasattr(obj, "build"):
        valid = {f.name for f in fields(obj)}
        return obj(**{k: v for k, v in kwargs.items() if k in valid})
    return _LegacyDatasetConfig(obj, kwargs)


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
        dataset: Any,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int = 1,
        collate_fn: Callable | None = None,
        seed: int = 42,
        shuffle: bool | None = None,
        group_by_length: bool = False,
        pp_enabled: bool = False,
        shuffle_buffer_size: int = 10000,
        **loader_kwargs: Any,
    ) -> None:
        """Shard / sample ``dataset`` for the given ranks, then construct the ``StatefulDataLoader``."""
        if isinstance(dataset, IterableDataset):
            dataset = _shard_iterable_dataset(dataset, dp_rank=dp_rank, dp_world_size=dp_world_size)
            if shuffle and hasattr(dataset, "shuffle"):
                try:
                    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
                    logger.info(f"Shuffling IterableDataset with buffer_size={shuffle_buffer_size}")
                except Exception as e:
                    logger.warning(f"IterableDataset shuffle skipped due to error: {e}")
        else:
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
            if pp_enabled:
                loader_kwargs["drop_last"] = True

        _set_spawn_start_method()
        super().__init__(dataset, collate_fn=collate_fn, **loader_kwargs)
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size


@dataclass
class DataloaderConfig:
    """A typed dataset config + loader settings; :meth:`build` produces a :class:`ParallelAwareDataloader`.

    ``dataset_config`` is a typed per-dataset config (a ``<Name>Config`` with ``build(**runtime) -> Dataset``,
    e.g. ``ChatDatasetConfig`` / ``GLUE_MRPCConfig``); the dataset's own args live there. Runtime objects
    (``tokenizer`` / ``processor`` / ...) are forwarded to ``dataset_config.build`` by :meth:`build` (only the
    keys its signature accepts). ``loader_kwargs`` is a thin ``torch`` ``DataLoader`` passthrough.
    """

    dataset_config: Any
    packing: PackingConfig | None = None
    shuffle: bool | None = None
    group_by_length: bool = False
    shuffle_buffer_size: int = 10000
    batch_size: int = 1
    seed: int = 42
    collate_fn: Callable | CollatorConfig | None = None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        *,
        dp_rank: int,
        dp_world_size: int,
        pp_enabled: bool = False,
        model: Any = None,
        cp_size: int = 1,
        attn_implementation: str | None = None,
        collate_wrapper: Callable | None = None,
        **runtime: Any,
    ) -> DataLoader:
        """Build the dataset (forwarding ``runtime`` objects) and wrap it in a :class:`ParallelAwareDataloader`.

        ``runtime`` carries the dataset's runtime build args (``tokenizer=`` for LLM, ``processor=`` for VLM,
        ...); only the keys ``dataset_config.build`` accepts are forwarded.
        """
        import contextlib

        from nemo_automodel.components.distributed.utils import FirstRankPerNode
        from nemo_automodel.components.training.rng import ScopedRNG

        # ``FirstRankPerNode`` materializes the dataset on node-rank-0 first, gating the other ranks behind a
        # Gloo barrier. That is correct for datasets that are built independently per rank, but it deadlocks
        # builders that run their own collective synchronization across *all* ranks (e.g. Megatron's
        # ``BlendedMegatronDatasetBuilder`` issues ``torch.distributed.barrier()`` internally): rank-0 would
        # wait inside that NCCL barrier while the peers sit in the Gloo barrier here. Such configs opt out via
        # ``manages_own_distributed_build`` and are built without the wrapper.
        build_ctx = (
            contextlib.nullcontext()
            if getattr(self.dataset_config, "manages_own_distributed_build", False)
            else FirstRankPerNode()
        )
        with ScopedRNG(seed=self.seed, ranked=True):
            with build_ctx:
                dataset = self.dataset_config.build(**_filter_runtime(self.dataset_config.build, runtime))

            collate_override = None
            if self.packing is not None and not self.packing.prepacked:
                dataset, collate_override = self.packing.build(
                    dataset,
                    split=getattr(self.dataset_config, "split", None),
                    seed=self.seed,
                    supports_seq_lens=(model is None or _model_supports_seq_lens(model)),
                    pad_token_id=getattr(runtime.get("tokenizer"), "pad_token_id", 0),
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
                tokenizer = runtime.get("tokenizer")
                if tokenizer is None:
                    raise ValueError("A tokenizer or processor is required to build the configured collator")
                collate_fn = collate_fn.build(tokenizer=tokenizer)
            if collate_fn is None:
                from nemo_automodel.components.datasets.utils import default_collater

                collate_fn = default_collater
            if collate_wrapper is not None:
                collate_fn = collate_wrapper(collate_fn)

            return ParallelAwareDataloader(
                dataset,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                seed=self.seed,
                shuffle=self.shuffle,
                group_by_length=self.group_by_length,
                pp_enabled=pp_enabled,
                shuffle_buffer_size=self.shuffle_buffer_size,
                **self.loader_kwargs,
            )


def build_dataloader_config(
    dataset_config: Any,
    *,
    dataloader: dict[str, Any] | None = None,
    packing: PackingConfig | None = None,
    collate_fn: Callable | CollatorConfig | None = None,
    seed: int = 42,
    local_batch_size: int = 1,
) -> DataloaderConfig:
    """Assemble a :class:`DataloaderConfig` from already-typed pieces + plain loader settings.

    Mirrors ``build_optimizer_config``: the recipe boundary instantiates the typed pieces — ``dataset_config``
    (via ``_callable_and_kwargs``), ``packing`` (via :func:`make_packing_config`) and the ``collate_fn``
    callable — and resolves the ``dataloader`` block to a plain dict, so this layer never sees a
    ``ConfigNode``. Loader-level keys that are :class:`DataloaderConfig` fields are lifted out of
    ``dataloader``; the remainder becomes ``loader_kwargs`` (``num_workers`` ...).
    """
    dl = dict(dataloader or {})
    dl.pop(
        "dataloader_type", None
    )  # legacy Megatron key; ignored (megatron uses the default sampler), not a DataLoader kwarg
    dl.pop("_target_", None)  # YAML dataloader target (loader is always ParallelAwareDataloader now); not a kwarg
    return DataloaderConfig(
        dataset_config=dataset_config,
        packing=packing,
        shuffle=dl.pop("shuffle", None),
        group_by_length=dl.pop("group_by_length", False),
        shuffle_buffer_size=dl.pop("shuffle_buffer_size", 10000),
        # batch_size is passed explicitly by ParallelAwareDataloader.build; pop any YAML
        # dataloader.batch_size so it doesn't also land in loader_kwargs and collide.
        batch_size=dl.pop("batch_size", local_batch_size),
        seed=seed,
        collate_fn=collate_fn,
        loader_kwargs=dl,
    )


__all__ = [
    "COLLATE_FNS",
    "CollatorConfig",
    "DATASET_CONFIGS",
    "PACKING_CONFIGS",
    "DataloaderConfig",
    "NeatPackingConfig",
    "PackingConfig",
    "ParallelAwareDataloader",
    "ThdPackingConfig",
    "build_dataloader_config",
    "make_collate_fn",
    "make_dataset_config",
    "make_packing_config",
]
