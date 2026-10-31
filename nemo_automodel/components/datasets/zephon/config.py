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

"""Declarative Zephon settings; construction uses the resolved runtime tokenizer."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from nemo_automodel.components.datasets.packing import (
    DEFAULT_PACKED_SEQUENCE_CONTRACT,
    BatchOutputFormat,
    PackedSequenceContract,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from zephon import Pipeline

    from nemo_automodel.components.datasets.loader import CollateFn
    from nemo_automodel.components.datasets.zephon.llm import TokenIds
    from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader

TextSpecialTokens = Literal["bos_eos", "bos", "eos", "none", "tokenizer_default"]


def _validate_insertion_overrides(
    policy: str | None,
    *,
    bos_token_id: int | None,
    eos_token_id: int | None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> None:
    """Validate explicit bracketing IDs without changing tokenizer state.

    Config construction rejects unused overrides. Runtime construction also
    checks vocabulary membership once the effective tokenizer is available.
    """
    if bos_token_id is None and eos_token_id is None:
        return
    valid_ids = set(tokenizer.get_vocab().values()) if tokenizer is not None else None
    for name, value, insertion in (("bos_token_id", bos_token_id, "bos"), ("eos_token_id", eos_token_id, "eos")):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer or None")
        if policy not in (insertion, "bos_eos"):
            raise ValueError(f"{name} requires an explicit {insertion} or bos_eos special-token policy")
        if valid_ids is not None and value not in valid_ids:
            raise ValueError(f"{name}={value} is not an existing tokenizer vocabulary ID")


@dataclass(frozen=True, kw_only=True)
class ZephonSource:
    """A named local/cloud dataset descriptor using Zephon's URI grammar."""

    name: str
    path: str
    fmt: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("Every Zephon source needs a nonempty name and path")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Zephon source weights must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class ZephonTextSourceConfig:
    """Select text tokenization, with optional explicit BOS/EOS insertion IDs.

    Overrides apply only to the corresponding ``tokenize_special_tokens``
    insertion policy and otherwise raise. They must identify existing model
    vocabulary tokens; tokenizer postprocessing and special-token settings
    are preserved. Omitted overrides use the effective tokenizer's IDs.
    """

    text_field: str = "text"
    tokenize_special_tokens: TextSpecialTokens = "bos_eos"
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if not self.text_field:
            raise ValueError("text_field must be nonempty")
        if self.tokenize_special_tokens not in ("bos_eos", "bos", "eos", "none", "tokenizer_default"):
            raise ValueError("Unknown raw-text tokenize_special_tokens policy")
        _validate_insertion_overrides(
            self.tokenize_special_tokens, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

    def build(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        options: ZephonOptions,
        ids: TokenIds,
        dp_rank: int,
        dp_world_size: int,
    ) -> Pipeline:
        """Build raw-text encoding on the shared source pipeline."""
        from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline

        _validate_insertion_overrides(
            self.tokenize_special_tokens,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tokenizer=tokenizer,
        )
        bos_token_id = self.bos_token_id if self.bos_token_id is not None else ids.bos
        eos_token_id = self.eos_token_id if self.eos_token_id is not None else ids.eos
        wrapping = options.packing.pack_algorithm in ("wrap", "best_fit_wrap")
        pipeline = create_source_pipeline(options, dp_rank=dp_rank, dp_world_size=dp_world_size)
        return pipeline.tokenize(
            tokenizer=tokenizer,
            field=self.text_field,
            add_attention_mask=False,
            max_length=options.packing.seq_len + 1 if wrapping and options.packing.seq_len is not None else None,
            split_long_samples=wrapping,
            special_tokens=self.tokenize_special_tokens,
            bos_token_id=bos_token_id if self.tokenize_special_tokens in ("bos", "bos_eos") else None,
            eos_token_id=eos_token_id if self.tokenize_special_tokens in ("eos", "bos_eos") else None,
            parallelism=options.workers.tokenize_parallelism,
        )


@dataclass(frozen=True, kw_only=True)
class ZephonPretokenizedTextSourceConfig:
    """Unshifted, unpadded tokens; an optional aligned loss mask selects targets.

    The adapter preserves supplied IDs and masks, adding an all-ones loss mask
    only when absent. It does not tokenize or filter documents. Zephon owns
    label shifting. With online packing disabled, optional ``positions`` retain
    subdocument boundaries; otherwise each supplied sequence is one document.
    Shifted labels and framework-specific cache metadata are rejected.
    """

    tokens_field: str = "input_ids"
    loss_mask_field: str | None = "loss_mask"

    def __post_init__(self) -> None:
        if not self.tokens_field or self.tokens_field == self.loss_mask_field:
            raise ValueError("Pretokenized text needs distinct nonempty token and loss-mask fields")

    def build(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        options: ZephonOptions,
        ids: TokenIds,
        dp_rank: int,
        dp_world_size: int,
    ) -> Pipeline:
        """Validate pretokenized fields and calibrate from their original token counts."""
        import numpy as np

        from nemo_automodel.components.datasets.zephon.llm import (
            measure_pretokenized_document,
            normalize_pretokenized_document,
        )
        from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline

        del ids
        valid_ids = np.asarray(sorted(set(tokenizer.get_vocab().values())), dtype=np.int64)
        pipeline = create_source_pipeline(
            options,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            measure=partial(measure_pretokenized_document, tokens_field=self.tokens_field),
        )
        return pipeline.map_transform(
            partial(
                normalize_pretokenized_document,
                config=self,
                valid_token_ids=valid_ids,
                preserve_positions=options.packing.pack_algorithm is None,
            ),
            parallelism=options.workers.map_parallelism,
        )


class TokenEstimationSettings(TypedDict, total=False):
    """Existing Zephon TokenEstimation fields accepted at the recipe boundary."""

    primer: Literal["measure"] | dict[str, float] | float
    calibration_samples: int
    calibration_shards_min: int
    calibration_shards_max: int
    fallback_tokens_per_byte: float


@dataclass(frozen=True, kw_only=True)
class ZephonSourceOptions:
    """Source traversal, token-aware allocation and end-of-data policy."""

    chunk_size: int = 16384
    shuffle_shards: bool = True
    shuffle_within_shard: bool = True
    shuffle_block_size: int | str | None = "auto"
    exhausted_policy: str | dict[str, str] | None = None
    reshuffle_on_repeat: bool | dict[str, bool] = True
    max_repeats: int | dict[str, int | None] | None = None
    stop_after_passes: int | None = None
    token_estimation: bool | TokenEstimationSettings | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("Zephon chunk_size must be positive")
        if not isinstance(self.token_estimation, (bool, dict, type(None))):
            raise ValueError("token_estimation must be a boolean or existing TokenEstimation settings")
        if isinstance(self.token_estimation, dict):
            unknown = self.token_estimation.keys() - TokenEstimationSettings.__annotations__.keys()
            if unknown:
                raise ValueError(f"Unknown TokenEstimation settings: {sorted(unknown)}")


@dataclass(frozen=True, kw_only=True)
class ZephonPackingOptions:
    """Optional online packing; seq_len bounds shifted model inputs.

    Omitted lengths use the model's input-token limit. ``pack_algorithm=None``
    keeps supplied documents separate and lets to_training pad shorter inputs.
    """

    seq_len: int | None = None
    pack_algorithm: Literal["wrap", "first_fit", "best_fit", "best_fit_wrap"] | None = "wrap"
    pack_num_bins: int | None = None
    max_num_documents: int | None = None
    max_num_documents_scope: Literal["batch", "bin"] = "batch"
    drop_oversized: bool | None = None
    pad_token_id: int | None = None
    shuffle_after_pack: bool | None = None
    shuffle_buffer_size: int | None = None

    def __post_init__(self) -> None:
        if self.seq_len is not None and (
            isinstance(self.seq_len, bool) or not isinstance(self.seq_len, int) or self.seq_len <= 0
        ):
            raise ValueError("Zephon seq_len must be positive")
        if self.pack_algorithm not in (None, "wrap", "first_fit", "best_fit", "best_fit_wrap"):
            raise ValueError(
                "Use pack_algorithm=null or an existing Zephon algorithm: wrap, first_fit, best_fit, best_fit_wrap"
            )
        packing_enabled = self.pack_algorithm is not None
        if not packing_enabled and (
            self.pack_num_bins is not None or self.max_num_documents is not None or self.drop_oversized
        ):
            raise ValueError("pack_num_bins, max_num_documents and drop_oversized require online packing")
        wrapping = self.pack_algorithm in ("wrap", "best_fit_wrap")
        if self.drop_oversized is None:
            object.__setattr__(self, "drop_oversized", packing_enabled and not wrapping)
        if self.shuffle_after_pack is None:
            object.__setattr__(self, "shuffle_after_pack", packing_enabled and not wrapping)
        if wrapping and self.pack_num_bins is not None:
            raise ValueError("pack_num_bins is only used by first_fit and best_fit")
        if wrapping and self.drop_oversized is True:
            raise ValueError("Wrapping splits oversized text; drop_oversized=true is incompatible")
        if self.pack_num_bins is not None and self.pack_num_bins <= 0:
            raise ValueError("pack_num_bins must be positive")
        if self.max_num_documents is not None and self.max_num_documents <= 0:
            raise ValueError("max_num_documents must be positive")
        if self.max_num_documents_scope not in ("batch", "bin"):
            raise ValueError("max_num_documents_scope must be 'batch' or 'bin'")


@dataclass(frozen=True, kw_only=True)
class ZephonTrainingOptions:
    """Consumer attention policy and Zephon boundary-target supervision."""

    allow_cross_document_attention: bool = False
    mask_segment_starts: bool = False


@dataclass(frozen=True, kw_only=True)
class ZephonWorkerOptions:
    """Per-stage operator parallelism; None uses the Zephon operator default."""

    tokenize_parallelism: int | None = 3
    map_parallelism: int | None = None
    pack_parallelism: int | None = 1
    fetch_parallelism: int | None = 2
    prefetch_parallelism: int | None = None
    shuffle_parallelism: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{item.name} must be a positive integer or None")


@dataclass(frozen=True, kw_only=True)
class ZephonRuntimeOptions:
    """Pipeline runner, shared-memory transport and prefetch buffering."""

    runner: Literal["process", "threads", "inline"] = "process"
    mtp_mode: bool = True
    prefetch_buffer_size: int = 0


@dataclass(frozen=True, kw_only=True)
class ZephonCacheOptions:
    """Downloaded source cache settings."""

    cache_dir: str | None = None
    cache_limit_bytes: int | str | None = None

    def __post_init__(self) -> None:
        if self.cache_limit_bytes is not None and self.cache_dir is None:
            raise ValueError("cache_limit_bytes requires cache_dir")


@dataclass(frozen=True, kw_only=True)
class ZephonCheckpointOptions:
    """Canonical lane layout and all-rank checkpoint aggregation."""

    canonical_replicas: int | None = None
    aggregate_dir: str | None = None
    aggregate_store: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.aggregate_dir and self.aggregate_store:
            raise ValueError("Set only one of aggregate_dir and aggregate_store")
        if self.canonical_replicas is not None and (
            isinstance(self.canonical_replicas, bool)
            or not isinstance(self.canonical_replicas, int)
            or self.canonical_replicas < 1
        ):
            raise ValueError("canonical_replicas must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class ZephonOptions:
    """Grouped settings nested under ``dataloader.zephon``.

    Dataset adapters select encoding. These groups configure the shared source,
    packing and delivery backbone for both text and image pipelines.
    """

    sources: tuple[ZephonSource, ...] = ()
    seed: int = 42
    source: ZephonSourceOptions = field(default_factory=ZephonSourceOptions)
    packing: ZephonPackingOptions = field(default_factory=ZephonPackingOptions)
    training: ZephonTrainingOptions = field(default_factory=ZephonTrainingOptions)
    workers: ZephonWorkerOptions = field(default_factory=ZephonWorkerOptions)
    runtime: ZephonRuntimeOptions = field(default_factory=ZephonRuntimeOptions)
    cache: ZephonCacheOptions = field(default_factory=ZephonCacheOptions)
    checkpoint: ZephonCheckpointOptions = field(default_factory=ZephonCheckpointOptions)

    def __post_init__(self) -> None:
        if not self.sources or len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("Zephon needs at least one source with unique names")


@dataclass(frozen=True, kw_only=True)
class ZephonLoaderConfig:
    """Shared construction capabilities and validation for Zephon training loaders."""

    zephon: ZephonOptions = field(default_factory=ZephonOptions)
    output_format: BatchOutputFormat = "batched"

    def __post_init__(self) -> None:
        if self.output_format not in ("batched", "thd"):
            raise ValueError("output_format must be batched or thd")

    @property
    def is_packed(self) -> bool:
        """Delivery carries document metadata, including for offline-packed input."""
        return True

    def resolve_sequence_length(self, *, packing_contract: PackedSequenceContract) -> int:
        """Resolve the shifted input limit before starting source workers.

        The effective model config supplies the default, not a tokenizer's
        possibly sentinel model_max_length. Explicit shorter training lengths
        remain valid. Position resets do not change the total input-token limit.
        """
        model_limit = packing_contract.max_sequence_length
        sequence_length = self.zephon.packing.seq_len
        if sequence_length is None:
            if model_limit is None:
                raise ValueError("Set dataloader.zephon.packing.seq_len: the model has no known input-token limit")
            sequence_length = model_limit
        if model_limit is not None and sequence_length > model_limit:
            raise ValueError(f"Zephon seq_len={sequence_length} exceeds the model input-token limit {model_limit}")
        return sequence_length

    def validate_parallelism(self, *, cp_size: int) -> None:
        """DP shards input; TP/PP peers share input, and CP remains unsupported."""
        if cp_size != 1:
            raise ValueError("Zephon does not support context parallelism; set cp_size=1")

    def validate_training(
        self, *, max_steps: int | None, global_batch_size: int, local_batch_size: int, domain_mixture: object = None
    ) -> None:
        """Validate run limits and optimizer-step alignment before model construction."""
        if domain_mixture is not None:
            raise ValueError("domain_mixture loss reweighting is not Zephon source mixing")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError(
                "An unsized Zephon loader needs a positive step_scheduler.max_steps cap; it may end earlier"
            )
        canonical = self.zephon.checkpoint.canonical_replicas
        if canonical is not None:
            if (
                isinstance(local_batch_size, bool)
                or not isinstance(local_batch_size, int)
                or local_batch_size <= 0
                or isinstance(global_batch_size, bool)
                or not isinstance(global_batch_size, int)
                or global_batch_size <= 0
                or global_batch_size % local_batch_size
            ):
                raise ValueError("step_scheduler.global_batch_size must be divisible by positive local_batch_size")
            if (global_batch_size // local_batch_size) % canonical:
                raise ValueError(
                    "step_scheduler.global_batch_size / local_batch_size must be a multiple of "
                    "dataloader.zephon.checkpoint.canonical_replicas so optimizer steps and checkpoints align with lane windows"
                )

    def validate_build(self, *, batch_size: int, cp_size: int, packing_contract: PackedSequenceContract) -> None:
        """Check cheap runtime contracts before processors, workers or sources are built."""
        from nemo_automodel.components.datasets.zephon.attention import require_supported_packing_contract

        self.validate_parallelism(cp_size=cp_size)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Zephon batch_size must be positive")
        # THD consumes lengths and does not interpret the batched mask contract.
        if self.output_format == "batched":
            require_supported_packing_contract(packing_contract)


def validate_zephon_dataset_fields(values: dict[str, Any], *, config_type: type[Any]) -> None:
    """Reject source-specific recipe typos before invoking its dataclass constructor."""
    unknown = set(values) - {item.name for item in fields(config_type)}
    if unknown:
        raise ValueError(f"Unexpected {config_type.__name__} dataset fields: {sorted(unknown)}")


# Group types are also the recipe boundary's schema; no duplicate field lists.
_OPTION_GROUPS = {
    "source": ZephonSourceOptions,
    "packing": ZephonPackingOptions,
    "training": ZephonTrainingOptions,
    "workers": ZephonWorkerOptions,
    "runtime": ZephonRuntimeOptions,
    "cache": ZephonCacheOptions,
    "checkpoint": ZephonCheckpointOptions,
}


def _group_recipe_options(values: dict[str, Any]) -> dict[str, Any]:
    """Accept grouped settings or flat setting names, rejecting duplicates.

    A setting may occur only once per input. The caller merges TOML and YAML
    after grouping so inline overrides replace individual settings, not groups.
    """
    values = dict(values)
    for name, config_type in _OPTION_GROUPS.items():
        group = values.pop(name, {})
        if not isinstance(group, dict):
            raise ValueError(f"Zephon {name} must be a mapping")
        group = dict(group)
        for item in fields(config_type):
            if item.name in values:
                if item.name in group:
                    raise ValueError(f"Specify Zephon {name}.{item.name} only once")
                group[item.name] = values.pop(item.name)
        unknown = group.keys() - {item.name for item in fields(config_type)}
        if unknown:
            raise ValueError(f"Unexpected Zephon {name} option(s): {sorted(unknown)}")
        if group:
            values[name] = group
    return values


def resolve_zephon_recipe_settings(loader_values: dict[str, Any]) -> dict[str, Any]:
    """Read integration TOML, with per-setting inline override precedence.

    Relative paths belong to the TOML that declares them; explicit inline sources
    replace its complete list. This parser remains local to AutoModel.
    """
    unknown = set(loader_values) - {"_target_", "zephon", "output_format"}
    if unknown:
        raise ValueError(f"Unexpected Zephon loader fields: {sorted(unknown)}; batch size comes from step_scheduler")
    inline = dict(loader_values.get("zephon", {}))
    data_config = inline.pop("data_config", None)
    values: dict[str, Any] = {}
    if data_config is not None:
        recipe_path = Path(data_config).expanduser().resolve()
        if not recipe_path.is_file():
            raise ValueError(f"Zephon data_config file does not exist: {recipe_path}")
        with recipe_path.open("rb") as recipe_file:
            values = _group_recipe_options(tomllib.load(recipe_file))
        for index, source in enumerate(values.get("sources", [])):
            path = source.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"Zephon source {source.get('name', index)!r} needs a nonempty path")
            if "://" not in path and not Path(path).is_absolute():
                source["path"] = str(recipe_path.parent / path)
        for group, key in (("cache", "cache_dir"), ("checkpoint", "aggregate_dir")):
            value = values.get(group, {}).get(key)
            if value is not None and not Path(value).is_absolute():
                values[group][key] = str(recipe_path.parent / value)
    for key, value in _group_recipe_options(inline).items():
        values[key] = {**values.get(key, {}), **value} if key in _OPTION_GROUPS else value
    if "repeat" in values:
        source = values.setdefault("source", {})
        if any(key in source for key in ("exhausted_policy", "max_repeats", "stop_after_passes")):
            raise ValueError("repeat cannot be combined with explicit Zephon exhaustion settings")
        repeat = values.pop("repeat")
        if not isinstance(repeat, bool):
            raise ValueError("repeat must be a boolean")
        if repeat:
            source["exhausted_policy"] = "repeat"
    return values


def zephon_options_from_recipe(values: dict[str, Any], *, seed: int) -> ZephonOptions:
    """Construct typed groups at the integration's recipe boundary."""
    values = _group_recipe_options(values)
    unknown = set(values) - {item.name for item in fields(ZephonOptions)}
    if unknown:
        raise ValueError(f"Unexpected Zephon option(s): {sorted(unknown)}")
    values.setdefault("seed", seed)
    sources = []
    for index, source in enumerate(values.get("sources", [])):
        unknown_source = set(source) - {item.name for item in fields(ZephonSource)}
        if unknown_source:
            raise ValueError(f"Unexpected Zephon source {source.get('name', index)!r} fields: {sorted(unknown_source)}")
        if not source.get("name") or not source.get("path"):
            raise ValueError(f"Zephon source {source.get('name', index)!r} needs a nonempty name and path")
        sources.append(ZephonSource(**source))
    values["sources"] = tuple(sources)
    for name, config_type in _OPTION_GROUPS.items():
        if name in values:
            values[name] = config_type(**values[name])
    return ZephonOptions(**values)


@dataclass(frozen=True, kw_only=True)
class ZephonDataloaderConfig(ZephonLoaderConfig):
    """Own construction of the Zephon training stream."""

    dataset_config: ZephonTextSourceConfig | ZephonPretokenizedTextSourceConfig = field(
        default_factory=ZephonTextSourceConfig
    )
    batch_size: int = 1

    @classmethod
    def from_recipe(
        cls,
        dataset_target: object,
        dataset_values: dict[str, Any],
        loader_values: dict[str, Any],
        *,
        seed: int,
        batch_size: int,
        packing: dict[str, Any],
    ) -> ZephonDataloaderConfig:
        """Resolve Zephon-specific recipe settings without generic recipe coupling."""
        if (
            packing.get("packed_sequence_size", 0)
            or packing.get("pack_size", 0)
            or packing.get("packing_strategy", "thd") != "thd"
        ):
            raise ValueError(
                "Native packing is incompatible with Zephon; use dataloader.zephon.packing.seq_len and pack_algorithm"
            )
        dataset_values = dict(dataset_values)
        dataset_values.pop("tokenizer", None)
        values = resolve_zephon_recipe_settings(loader_values)
        source_path = dataset_values.pop("path_or_dataset_id", None)
        if source_path is not None:
            if values.get("sources"):
                raise ValueError("Choose dataset.path_or_dataset_id or dataloader.zephon.sources, not both")
            paths = [source_path] if isinstance(source_path, str) else source_path
            values["sources"] = [{"name": f"source_{index}", "path": path} for index, path in enumerate(paths)]
        if dataset_target in (
            ZephonTextSourceConfig,
            "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig",
        ):
            config_type = ZephonTextSourceConfig
            for key in ("text_field", "tokenize_special_tokens"):
                if key in values:
                    dataset_values.setdefault(key, values.pop(key))
        elif dataset_target in (
            ZephonPretokenizedTextSourceConfig,
            "nemo_automodel.components.datasets.zephon.ZephonPretokenizedTextSourceConfig",
        ):
            config_type = ZephonPretokenizedTextSourceConfig
            if "tokens_field" in values:
                dataset_values.setdefault("tokens_field", values.pop("tokens_field"))
        else:
            raise ValueError("Unsupported Zephon dataset target; select text or pretokenized tokens")
        validate_zephon_dataset_fields(dataset_values, config_type=config_type)
        return cls(
            dataset_config=config_type(**dataset_values),
            zephon=zephon_options_from_recipe(values, seed=seed),
            batch_size=batch_size,
            output_format=loader_values.get("output_format", "batched"),
        )

    @property
    def dataset_builds_on_all_ranks(self) -> bool:
        """Keep Zephon initialization and calibration outside rank-first ordering."""
        return True

    @property
    def packed_sequence_length(self) -> int:
        """Explicit shifted length, or zero until the model-derived limit is known."""
        return self.zephon.packing.seq_len or 0

    def build(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        dp_rank: int,
        dp_world_size: int,
        dataset_build_context: AbstractContextManager[object] | None = None,
        pp_enabled: bool = False,
        supports_seq_lens: bool = True,
        cp_size: int = 1,
        packing_contract: PackedSequenceContract = DEFAULT_PACKED_SEQUENCE_CONTRACT,
        collate_wrapper: Callable[[CollateFn], CollateFn] | None = None,
        ignore_index: int = -100,
    ) -> ZephonDataLoader:
        """Construct the checkpointable stream without mutating this config.

        Runtime arguments match the native text builder. Zephon optionally packs documents and
        ``to_training`` owns collation and shifting in the loader consumer; cheap
        AutoModel adaptations then select the output representation. Native PP collator wrappers do not apply. ``dataset_build_context``
        is deliberately not entered: source initialization may involve all ranks.
        ``cp_size`` is validated; ``ignore_index`` controls Zephon's shifted labels.
        """
        from nemo_automodel.components.datasets.zephon.llm import resolve_token_ids
        from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader, convert_text_batch
        from nemo_automodel.components.datasets.zephon.pipeline import create_text_pipeline

        del dataset_build_context, pp_enabled, collate_wrapper
        if self.output_format == "thd" and not supports_seq_lens:
            raise ValueError("THD output requires a model/backend that accepts seq_lens")
        if tokenizer is None:
            raise ValueError("Zephon requires the resolved model tokenizer")
        self.validate_build(batch_size=self.batch_size, cp_size=cp_size, packing_contract=packing_contract)
        sequence_length = self.resolve_sequence_length(packing_contract=packing_contract)
        resolved = replace(
            self, zephon=replace(self.zephon, packing=replace(self.zephon.packing, seq_len=sequence_length))
        )
        ids = resolve_token_ids(tokenizer, pad_token_id=self.zephon.packing.pad_token_id)
        no_packing = self.zephon.packing.pack_algorithm is None
        return ZephonDataLoader(
            pipeline=create_text_pipeline(resolved, tokenizer=tokenizer, dp_rank=dp_rank, dp_world_size=dp_world_size),
            packing_contract=packing_contract,
            convert_batch=partial(
                convert_text_batch,
                packing_contract=packing_contract,
                output_format=self.output_format,
                ignore_index=ignore_index,
                mask_segment_starts=self.zephon.training.mask_segment_starts,
                allow_cross_document_attention=self.zephon.training.allow_cross_document_attention,
                pad_to_length=sequence_length + 1 if no_packing else None,
                pad_token_id=ids.pad if no_packing else None,
            ),
            output_format=self.output_format,
        )
