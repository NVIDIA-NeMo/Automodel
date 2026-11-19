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


"""Typed options for shared Zephon source construction and runtime delivery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, TypedDict

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from zephon.io import ParquetRGCacheOptions, StoreOptions


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
