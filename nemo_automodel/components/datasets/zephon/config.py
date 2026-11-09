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
import re
import tomllib
from abc import abstractmethod
from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypedDict, TypeVar

from nemo_automodel.components.datasets.loader import (
    DataloaderConfig,
    StreamBuildContext,
    StreamDatasetConfig,
    TextStreamBuildContext,
)
from nemo_automodel.components.datasets.packing import (
    BatchOutputFormat,
    PackedSequenceContract,
)
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from zephon import Pipeline
    from zephon.io import ParquetRGCacheOptions, StoreOptions

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


class TokenEstimationSettings(TypedDict, total=False):
    """Existing Zephon TokenEstimation fields accepted at the recipe boundary."""

    primer: Literal["measure"] | dict[str, float] | float
    calibration_samples: int
    calibration_shards_min: int
    calibration_shards_max: int
    fallback_tokens_per_byte: float


@dataclass(frozen=True, kw_only=True)
class ZephonPrefetchOptions:
    """Cache-warmup operator settings; None fields keep Zephon's defaults."""

    buffer_size: int | None = None
    """Number of samples to look ahead; None uses Zephon's operator default."""
    parallelism: int | None = None
    """Concurrent cache-warmup workers; None uses Zephon's operator default."""

    def __post_init__(self) -> None:
        for name, value in (("buffer_size", self.buffer_size), ("parallelism", self.parallelism)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"source.prefetch.{name} must be a positive integer or None")


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
    # None omits this argument: Zephon chooses one finite pass unless another
    # exhaustion/repeat policy is set. It is not an explicit unlimited-pass value.
    stop_after_passes: int | None = None
    # AutoModel policy: estimate tokens for multi-source text (including SFT and
    # pretokenized input). None delegates to Zephon's sample-based default.
    token_estimation: bool | TokenEstimationSettings | Literal["auto"] | None = "auto"
    # None omits the cache-warmup operator; an empty config enables it with
    # Zephon's defaults. Runtime batch/stage/IPC buffering is independent.
    prefetch: ZephonPrefetchOptions | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("Zephon chunk_size must be positive")
        if not isinstance(self.token_estimation, (bool, dict, type(None))) and self.token_estimation != "auto":
            raise ValueError("token_estimation must be 'auto', a boolean, None or existing TokenEstimation settings")
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

    seq_len: int | None = None  # None uses the model input-token limit at build time.
    pack_algorithm: Literal["wrap", "first_fit", "best_fit", "best_fit_wrap"] | None = "wrap"
    pack_num_bins: int | None = None  # None selects 64 bins for first_fit/best_fit; unused by wrapping.
    max_num_documents: int | None = None
    max_num_documents_scope: Literal["batch", "bin"] = "batch"
    # AutoModel defaults: whole-document packing drops oversized documents and
    # shuffles its output; wrapping and unpacked input do neither by default.
    drop_oversized: bool | None = None
    pad_token_id: int | None = None  # None derives an existing ID from the runtime tokenizer.
    shuffle_after_pack: bool | None = None
    shuffle_buffer_size: int | None = None  # None uses Zephon's shuffle buffer default.

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
        if self.pack_num_bins is None and packing_enabled and not wrapping:
            object.__setattr__(self, "pack_num_bins", 64)
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
    """Per-stage operator parallelism.

    Concrete values are integration tuning defaults. None uses each Zephon
    operator's default, including the image encoder's advertised traits.
    """

    tokenize_parallelism: int | None = 3
    map_parallelism: int | None = None
    pack_parallelism: int | None = 1
    fetch_parallelism: int | None = 2
    shuffle_parallelism: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{item.name} must be a positive integer or None")


@dataclass(frozen=True, kw_only=True)
class ZephonRuntimeOptions:
    """Pipeline execution and queue capacities; None uses Zephon's automatic sizing."""

    runner: Literal["process", "threads", "inline"] = "process"
    mtp_mode: bool = True
    prefetch_batches: int | None = None
    """Pipeline-output prefetch depth in batches; zero disables this prefetch buffer."""
    default_stage_prefetch: int | None = None
    """Prefetch depth between stages; zero disables the optional stage prefetch buffer."""
    op_queue_capacity: int | None = None
    """Positive queue capacity between operators within a stage."""
    mtp_buffer: int | None = None
    """Positive capacity of the MTP engine-to-consumer IPC queue."""
    mtp_prefetch: int | None = None
    """Consumer-side MTP prefetch depth; zero disables the background receiver thread."""

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("prefetch_batches", self.prefetch_batches, 0),
            ("default_stage_prefetch", self.default_stage_prefetch, 0),
            ("op_queue_capacity", self.op_queue_capacity, 1),
            ("mtp_buffer", self.mtp_buffer, 1),
            ("mtp_prefetch", self.mtp_prefetch, 0),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                requirement = "positive" if minimum else "nonnegative"
                raise ValueError(f"{name} must be a {requirement} integer or None")


@dataclass(frozen=True, kw_only=True)
class ZephonParquetRGCacheOptions:
    """Decoded Parquet cache overrides; omitted fields retain Zephon defaults."""

    enabled: bool | None = None
    root: str | None = None
    limit_bytes: int | str | None = None
    min_free_bytes: int | str | None = None

    def build(self) -> ParquetRGCacheOptions:
        """Construct Zephon's cache settings without replacing automatic defaults."""
        _, io = safe_import("zephon.io")
        values = {
            "enabled": self.enabled,
            "root": self.root,
            "limit_bytes": self.limit_bytes,
            "min_free_bytes": self.min_free_bytes,
        }
        return io.ParquetRGCacheOptions.from_any({name: value for name, value in values.items() if value is not None})


@dataclass(frozen=True, kw_only=True)
class ZephonCacheOptions:
    """Downloaded source cache settings."""

    cache_dir: str | None = None
    cache_limit_bytes: int | str | None = None
    parquet_rg_cache: ZephonParquetRGCacheOptions | None = None
    """Separate decoded-cache budget; source shards use cache_limit_bytes."""

    def __post_init__(self) -> None:
        if self.cache_limit_bytes is not None and self.cache_dir is None:
            raise ValueError("cache_limit_bytes requires cache_dir")

    def build(self) -> StoreOptions:
        """Configure source and decoded caches through Zephon's existing I/O API."""
        _, io = safe_import("zephon.io")
        cache = None
        if self.cache_dir is not None:
            cache = {"enabled": True, "root": self.cache_dir, "limit_bytes": self.cache_limit_bytes}
        parquet = io.ParquetOptions.from_any(None)
        if self.parquet_rg_cache is not None:
            parquet = io.ParquetOptions(rg_cache=self.parquet_rg_cache.build())
        return io.StoreOptions(cache=io.CacheOptions.from_any(cache), parquet=parquet)


@dataclass(frozen=True, kw_only=True)
class ZephonCheckpointOptions:
    """Canonical lane layout and all-rank checkpoint aggregation."""

    canonical_replicas: int | None = None  # None uses the runtime DP world size.
    aggregate_dir: str | None = None
    # None selects torch:// when distributed and no aggregate_dir was supplied;
    # otherwise the pipeline keeps Zephon's local checkpoint storage default.
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
    """Grouped settings nested under ``dataset.zephon``.

    Dataset adapters select encoding. These groups configure the shared source,
    packing and delivery backbone for both text and image pipelines. Concrete
    values are integration defaults. Field comments document automatic choices
    and values delegated to Zephon; these remain declarative until build time.
    """

    source_type: ClassVar[type[ZephonSource]] = ZephonSource
    sources: tuple[ZephonSource, ...] = ()
    source: ZephonSourceOptions = field(default_factory=ZephonSourceOptions)
    packing: ZephonPackingOptions = field(default_factory=ZephonPackingOptions)
    training: ZephonTrainingOptions = field(default_factory=ZephonTrainingOptions)
    workers: ZephonWorkerOptions = field(default_factory=ZephonWorkerOptions)
    runtime: ZephonRuntimeOptions = field(default_factory=ZephonRuntimeOptions)
    cache: ZephonCacheOptions = field(default_factory=ZephonCacheOptions)
    checkpoint: ZephonCheckpointOptions = field(default_factory=ZephonCheckpointOptions)

    def __post_init__(self) -> None:
        # A dataset config can supply its source through a native path field.
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("Zephon source names must be unique")


_Context = TypeVar("_Context", bound=StreamBuildContext)
_Result = TypeVar("_Result")


@dataclass(frozen=True, kw_only=True)
class ZephonStreamDatasetConfig(StreamDatasetConfig[_Context, _Result], Generic[_Context, _Result]):
    """Shared construction capabilities and validation for Zephon training loaders."""

    zephon: ZephonOptions = field(default_factory=ZephonOptions)
    output_format: BatchOutputFormat = "batched"
    path_or_dataset_id: str | list[str] | None = None
    """Single local dataset root or HF dataset ID, as on native instruction configs."""
    paths: str | Path | list[str] | None = None
    """Single dataset root; native blends and Megatron binary inputs require explicit migration."""
    split: str | None = None
    """Named HF split when path_or_dataset_id is a bare HF ID; defaults to train."""
    name: str | None = None
    """HF subset/configuration when path_or_dataset_id is a bare HF ID."""

    options_type: ClassVar[type[ZephonOptions]] = ZephonOptions
    # Keep initialization and calibration outside rank-first ordering.
    builds_on_all_ranks: ClassVar[bool] = True

    def __post_init__(self) -> None:
        # Dataclass dataset targets receive YAML dictionaries at this boundary.
        if isinstance(self.zephon, dict):
            object.__setattr__(self, "zephon", zephon_options_from_values(self.zephon, options_type=self.options_type))
        elif not isinstance(self.zephon, self.options_type):
            raise TypeError(f"zephon must be {self.options_type.__name__} or a settings mapping")
        native_source = self._native_source()
        if native_source is not None:
            object.__setattr__(self, "zephon", replace(self.zephon, sources=(native_source,)))
            # Keep only the canonical declaration, including across dataclasses.replace().
            object.__setattr__(self, "path_or_dataset_id", None)
            object.__setattr__(self, "paths", None)
            object.__setattr__(self, "split", None)
            object.__setattr__(self, "name", None)
        if not self.zephon.sources:
            raise ValueError("Zephon needs sources: set path_or_dataset_id, paths or zephon.sources")
        if self.output_format not in ("batched", "thd"):
            raise ValueError("output_format must be batched or thd")

    def _native_source(self) -> ZephonSource | None:
        """Normalize supported native source fields without changing source selection.

        Native multi-file concatenation and unweighted Megatron blends do not
        mean equal-weight Zephon mixtures. Individual files likewise cannot be
        replaced with their parent directories without selecting extra data.
        """
        if self.path_or_dataset_id is not None and self.paths is not None:
            raise ValueError("Choose only one source declaration: path_or_dataset_id, paths or zephon.sources")
        value = self.path_or_dataset_id if self.path_or_dataset_id is not None else self.paths
        if value is None:
            if self.split is not None or self.name is not None:
                raise ValueError("split and name require path_or_dataset_id with a bare HF dataset ID")
            return None
        if self.zephon.sources:
            raise ValueError(
                "Choose only one source declaration: path_or_dataset_id, paths or zephon.sources (including TOML)"
            )
        # The native instruction loader recognizes bare HF IDs only for scalar strings.
        bare_hf_id = (
            self.path_or_dataset_id is not None
            and isinstance(value, str)
            and re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", value) is not None
            and not Path(value).exists()
        )
        if bare_hf_id:
            split = self.split if self.split is not None else "train"
            for field_name, selection in (("split", split), ("name", self.name)):
                if selection is not None and (
                    not isinstance(selection, str) or re.fullmatch(r"[a-zA-Z0-9_.-]+", selection) is None
                ):
                    raise ValueError(
                        f"Zephon requires a simple named HF {field_name}; split slicing and ratios are unsupported"
                    )
            subset = f"{self.name}/" if self.name is not None else ""
            return ZephonSource(name="source_0", path=f"hf://{value}/{subset}{split}")
        if self.name is not None or self.split not in (None, "train"):
            raise ValueError(
                "split and name can select only a bare HF dataset ID; use an explicit source URI otherwise"
            )
        if isinstance(value, list):
            if not value:
                raise ValueError("Zephon source paths must not be empty")
            if len(value) != 1:
                raise ValueError(
                    "Native path lists and weighted blends are not translated into equal-weight Zephon mixtures; "
                    "configure zephon.sources with explicit weights and source.token_estimation instead"
                )
            value = value[0]
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Zephon source must be one nonempty path; split-specific blend mappings are unsupported")
        path = str(value)
        if any(marker in path for marker in ("*", "?", "[", "]")):
            raise ValueError("Zephon source paths require a dataset root, not a file glob or split expression")
        local = "://" not in path
        local_directory = local and Path(path).is_dir()
        suffix = Path(path).suffix.lower()
        if not local_directory and (
            suffix in (".bin", ".idx") or local and (Path(path + ".bin").is_file() or Path(path + ".idx").is_file())
        ):
            raise ValueError(
                "Zephon cannot read Megatron .bin/.idx datasets; export documents to a Zephon-readable format"
            )
        if not local_directory and (
            suffix in (".json", ".jsonl", ".parquet", ".arrow")
            or local
            and Path(path).is_file()
            and suffix != ".manifest"
        ):
            raise ValueError(
                "Zephon source paths require a dataset root, not an individual file or a blend JSON file; "
                "use a directory containing exactly the intended shards (its parent is not selected automatically)"
            )
        return ZephonSource(name="source_0", path=path)

    # Delivery carries document metadata, including for offline-packed input.
    is_packed: ClassVar[bool] = True

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
                raise ValueError("Set dataset.zephon.packing.seq_len: the model has no known input-token limit")
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
        """Validate run limits and optimizer-step alignment before stream construction."""
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
                    "dataset.zephon.checkpoint.canonical_replicas so optimizer steps and checkpoints align with lane windows"
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

    @property
    def packed_sequence_length(self) -> int:
        """Explicit input length, or zero until the model-derived limit is known."""
        return self.zephon.packing.seq_len or 0

    def validate_context(self, context: StreamBuildContext) -> None:
        """Validate runtime training ownership before processors or source I/O."""
        if not context.is_training:
            raise ValueError("Zephon validation is not supported yet; select a native validation dataset")
        schedule = context.training_schedule
        if schedule is None:
            raise ValueError(
                "This recipe must pass a runtime training_schedule to build Zephon; "
                "use the LLM or image finetuning recipe with a positive max_steps"
            )
        if context.batch_size != schedule.local_batch_size:
            raise ValueError("Zephon batch_size must match step_scheduler.local_batch_size for optimizer accounting")
        self.validate_training(
            max_steps=schedule.max_steps,
            global_batch_size=schedule.global_batch_size,
            local_batch_size=context.batch_size,
            domain_mixture=context.domain_mixture,
        )
        self.validate_build(
            batch_size=context.batch_size, cp_size=context.cp_size, packing_contract=context.packing_contract
        )


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
        if name == "source" and group.get("prefetch") is not None:
            prefetch = group["prefetch"]
            if not isinstance(prefetch, dict):
                raise ValueError("Zephon source.prefetch must be a mapping or None")
            unknown = prefetch.keys() - {item.name for item in fields(ZephonPrefetchOptions)}
            if unknown:
                raise ValueError(f"Unexpected Zephon source.prefetch option(s): {sorted(unknown)}")
        if name == "cache" and group.get("parquet_rg_cache") is not None:
            decoded = group["parquet_rg_cache"]
            if not isinstance(decoded, dict):
                raise ValueError("Zephon cache.parquet_rg_cache must be a mapping or None")
            unknown = decoded.keys() - {item.name for item in fields(ZephonParquetRGCacheOptions)}
            if unknown:
                raise ValueError(f"Unexpected Zephon cache.parquet_rg_cache option(s): {sorted(unknown)}")
        if group:
            values[name] = group
    return values


def resolve_zephon_settings(values: dict[str, Any]) -> dict[str, Any]:
    """Read integration TOML, with per-setting inline override precedence.

    Relative paths belong to the TOML that declares them; explicit inline sources
    replace its complete list. This parser remains local to AutoModel.
    """
    inline = dict(values)
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
        decoded = values.get("cache", {}).get("parquet_rg_cache")
        if decoded is not None and decoded.get("root") is not None:
            if not Path(decoded["root"]).is_absolute():
                decoded["root"] = str(recipe_path.parent / decoded["root"])
    for key, value in _group_recipe_options(inline).items():
        if key in _OPTION_GROUPS:
            inherited = values.get(key, {})
            if key == "source" and isinstance(value.get("prefetch"), dict):
                # Override individual nested fields, while explicit null disables
                # the operator even when the TOML file enabled it.
                value = {**value, "prefetch": {**(inherited.get("prefetch") or {}), **value["prefetch"]}}
            if key == "cache" and isinstance(value.get("parquet_rg_cache"), dict):
                value = {
                    **value,
                    "parquet_rg_cache": {**(inherited.get("parquet_rg_cache") or {}), **value["parquet_rg_cache"]},
                }
            values[key] = {**inherited, **value}
        else:
            values[key] = value
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


def zephon_options_from_values(
    values: dict[str, Any], *, options_type: type[ZephonOptions] = ZephonOptions
) -> ZephonOptions:
    """Construct typed groups at the integration's recipe boundary."""
    values = resolve_zephon_settings(values)
    unknown = set(values) - {item.name for item in fields(options_type)}
    if unknown:
        raise ValueError(f"Unexpected Zephon option(s): {sorted(unknown)}")
    sources = []
    source_type = options_type.source_type
    for index, source in enumerate(values.get("sources", [])):
        unknown_source = set(source) - {item.name for item in fields(source_type)}
        if unknown_source:
            raise ValueError(f"Unexpected Zephon source {source.get('name', index)!r} fields: {sorted(unknown_source)}")
        if not source.get("name") or not source.get("path"):
            raise ValueError(f"Zephon source {source.get('name', index)!r} needs a nonempty name and path")
        sources.append(source_type(**source))
    values["sources"] = tuple(sources)
    prefetch = values.get("source", {}).get("prefetch")
    if prefetch is not None:
        values["source"]["prefetch"] = ZephonPrefetchOptions(**prefetch)
    decoded = values.get("cache", {}).get("parquet_rg_cache")
    if decoded is not None:
        values["cache"]["parquet_rg_cache"] = ZephonParquetRGCacheOptions(**decoded)
    # A partial group must inherit its declared factory's defaults, including
    # image-specific source/packing defaults on an options subtype.
    group_fields = {item.name: item for item in fields(options_type) if item.name in _OPTION_GROUPS}
    for name, item in group_fields.items():
        if name in values:
            values[name] = item.default_factory(**values[name])
    return options_type(**values)


@dataclass(frozen=True, kw_only=True)
class ZephonLanguageDatasetConfig(ZephonStreamDatasetConfig[TextStreamBuildContext, "ZephonDataLoader"]):
    """Complete text stream construction shared by explicit encoding configs."""

    @property
    def dataloader_config_type(self) -> type[ZephonDataloaderConfig]:
        """Select the text metadata adapter when the loader target is omitted."""
        return ZephonDataloaderConfig

    @abstractmethod
    def tokenize(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        options: ZephonOptions,
        ids: TokenIds,
        seed: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> Pipeline:
        """Add the selected objective's encoding to the shared source backbone."""

    def build(self, *, context: TextStreamBuildContext) -> ZephonDataLoader:
        """Build the checkpointable training stream from resolved runtime inputs."""
        from nemo_automodel.components.datasets.zephon.llm import resolve_token_ids
        from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader, convert_text_batch
        from nemo_automodel.components.datasets.zephon.pipeline import create_text_pipeline

        if not isinstance(context, TextStreamBuildContext):
            raise TypeError("A Zephon text dataset requires the LLM dataloader, not the VLM recipe")
        self.validate_context(context)
        tokenizer = context.tokenizer
        packing_contract = context.packing_contract
        if self.output_format == "thd" and not context.supports_seq_lens:
            raise ValueError("THD output requires a model/backend that accepts seq_lens")
        if tokenizer is None:
            raise ValueError("Zephon requires the resolved model tokenizer")
        sequence_length = self.resolve_sequence_length(packing_contract=packing_contract)
        resolved = replace(
            self, zephon=replace(self.zephon, packing=replace(self.zephon.packing, seq_len=sequence_length))
        )
        ids = resolve_token_ids(tokenizer, pad_token_id=self.zephon.packing.pad_token_id)
        no_packing = self.zephon.packing.pack_algorithm is None
        return ZephonDataLoader(
            pipeline=create_text_pipeline(
                resolved,
                tokenizer=tokenizer,
                seed=context.seed,
                batch_size=context.batch_size,
                dp_rank=context.dp_rank,
                dp_world_size=context.dp_world_size,
            ),
            packing_contract=packing_contract,
            convert_batch=partial(
                convert_text_batch,
                packing_contract=packing_contract,
                output_format=self.output_format,
                ignore_index=context.ignore_index,
                mask_segment_starts=self.zephon.training.mask_segment_starts,
                allow_cross_document_attention=self.zephon.training.allow_cross_document_attention,
                pad_to_length=sequence_length + 1 if no_packing else None,
                pad_token_id=ids.pad if no_packing else None,
            ),
            output_format=self.output_format,
        )


@dataclass
class ZephonDataloaderConfig(DataloaderConfig):
    """Adapt Zephon text metadata while reusing common stream construction.

    Pipeline settings remain on ``dataset_config.zephon``. The inherited build
    method supplies runtime inputs and returns the actual Zephon loader.
    """

    dataset_config: ZephonLanguageDatasetConfig

    def __post_init__(self) -> None:
        """Reject a native or image dataset before model preparation or source I/O."""
        if not isinstance(self.dataset_config, ZephonLanguageDatasetConfig):
            raise TypeError("ZephonDataloaderConfig requires a Zephon text dataset config")

    @property
    def is_packed(self) -> bool:
        """Whether the Zephon text stream reports packed inputs."""
        return self.dataset_config.is_packed

    @property
    def packed_sequence_length(self) -> int:
        """Input length configured on the Zephon dataset, or zero until resolved."""
        return self.dataset_config.packed_sequence_length

    @property
    def output_format(self) -> BatchOutputFormat:
        """Batched or THD output selected on the Zephon dataset."""
        return self.dataset_config.output_format


@dataclass(frozen=True, kw_only=True)
class ZephonTextDatasetConfig(ZephonLanguageDatasetConfig):
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
        super().__post_init__()
        if not self.text_field:
            raise ValueError("text_field must be nonempty")
        if self.tokenize_special_tokens not in ("bos_eos", "bos", "eos", "none", "tokenizer_default"):
            raise ValueError("Unknown raw-text tokenize_special_tokens policy")
        _validate_insertion_overrides(
            self.tokenize_special_tokens, bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id
        )

    def tokenize(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        options: ZephonOptions,
        ids: TokenIds,
        seed: int,
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
        pipeline = create_source_pipeline(options, seed=seed, dp_rank=dp_rank, dp_world_size=dp_world_size)
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
class ZephonPretokenizedTextDatasetConfig(ZephonLanguageDatasetConfig):
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
        super().__post_init__()
        if not self.tokens_field or self.tokens_field == self.loss_mask_field:
            raise ValueError("Pretokenized text needs distinct nonempty token and loss-mask fields")

    def tokenize(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        options: ZephonOptions,
        ids: TokenIds,
        seed: int,
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
            seed=seed,
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
