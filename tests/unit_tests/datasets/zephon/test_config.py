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


"""Local validation of shared Zephon source, packing and runtime options."""

import pytest

from nemo_automodel.components.datasets.zephon.config import (
    ZephonCacheOptions,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonParquetRGCacheOptions,
    ZephonPrefetchOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonWorkerOptions,
)


@pytest.mark.parametrize(
    "name",
    [
        "fetch_parallelism",
        "map_parallelism",
        "tokenize_parallelism",
        "pack_parallelism",
        "shuffle_parallelism",
    ],
)
def test_operator_parallelism_accepts_default_or_positive_workers(name):
    sources = (ZephonSource(name="text", path="unused"),)
    for value in (None, 1, 3):
        options = ZephonOptions(sources=sources, workers=ZephonWorkerOptions(**{name: value}))
        assert getattr(options.workers, name) == value
    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer or None"):
            ZephonOptions(sources=sources, workers=ZephonWorkerOptions(**{name: invalid}))


@pytest.mark.parametrize(
    "algorithm,expected", [("wrap", False), ("best_fit_wrap", False), ("first_fit", True), ("best_fit", True)]
)
def test_packing_defaults_are_resolved_once(algorithm, expected):
    options = ZephonOptions(
        sources=(ZephonSource(name="text", path="unused"),), packing=ZephonPackingOptions(pack_algorithm=algorithm)
    )
    assert options.packing.drop_oversized is expected
    assert options.packing.shuffle_after_pack is expected


def test_disabled_packing_defaults_and_incompatible_options():
    options = ZephonPackingOptions(pack_algorithm=None)
    assert options.drop_oversized is False and options.shuffle_after_pack is False
    for invalid in ({"pack_num_bins": 2}, {"max_num_documents": 2}, {"drop_oversized": True}):
        with pytest.raises(ValueError, match="require online packing"):
            ZephonPackingOptions(pack_algorithm=None, **invalid)
    for length in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="seq_len must be positive"):
            ZephonPackingOptions(seq_len=length)


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_whole_document_packing_owns_its_default_bin_count(algorithm):
    options = ZephonPackingOptions(pack_algorithm=algorithm)
    assert options.pack_num_bins == 64
    assert options.drop_oversized is True and options.shuffle_after_pack is True
    explicit = ZephonPackingOptions(
        pack_algorithm=algorithm, pack_num_bins=8, drop_oversized=False, shuffle_after_pack=False
    )
    assert explicit.pack_num_bins == 8
    assert explicit.drop_oversized is False and explicit.shuffle_after_pack is False


@pytest.mark.parametrize("algorithm", [None, "wrap", "best_fit_wrap"])
def test_non_bin_packing_has_no_bin_count(algorithm):
    assert ZephonPackingOptions(pack_algorithm=algorithm).pack_num_bins is None


@pytest.mark.parametrize(
    "name,allows_zero",
    [
        ("prefetch_batches", True),
        ("default_stage_prefetch", True),
        ("op_queue_capacity", False),
        ("mtp_buffer", False),
        ("mtp_prefetch", True),
    ],
)
def test_runtime_buffer_controls_keep_zephon_defaults_and_validate_capacities(name: str, allows_zero: bool) -> None:
    assert getattr(ZephonRuntimeOptions(), name) is None
    for value in (None, 1, 8):
        assert getattr(ZephonRuntimeOptions(**{name: value}), name) == value
    if allows_zero:
        assert getattr(ZephonRuntimeOptions(**{name: 0}), name) == 0
    invalid = [-1, True, 1.5, "8"] + ([] if allows_zero else [0])
    for value in invalid:
        with pytest.raises(ValueError, match=f"{name} must be a .* integer or None"):
            ZephonRuntimeOptions(**{name: value})


@pytest.mark.parametrize("name", ["buffer_size", "parallelism"])
def test_source_prefetch_accepts_operator_defaults_or_positive_values(name: str) -> None:
    for value in (None, 1, 8):
        assert getattr(ZephonPrefetchOptions(**{name: value}), name) == value
    for invalid in (0, -1, True, 1.5, "8"):
        with pytest.raises(ValueError, match=f"source.prefetch.{name} must be a positive integer or None"):
            ZephonPrefetchOptions(**{name: invalid})


def test_cache_overrides_preserve_automatic_defaults_and_explicit_disable() -> None:
    from zephon.io import StoreOptions

    assert ZephonCacheOptions().build() == StoreOptions.from_any(None)
    options = ZephonCacheOptions(
        cache_dir="/cache",
        cache_limit_bytes="80gb",
        parquet_rg_cache=ZephonParquetRGCacheOptions(enabled=False, limit_bytes="8gb", min_free_bytes=0),
    ).build()
    assert options.cache.enabled and options.cache.limit_bytes == 80 * 1024**3
    assert options.parquet.rg_cache.enabled is False
    assert options.parquet.rg_cache.limit_bytes == 8 * 1024**3
    assert options.parquet.rg_cache.min_free_bytes == 0
    decoded_only = ZephonCacheOptions(parquet_rg_cache=ZephonParquetRGCacheOptions(root="/decoded")).build()
    assert not decoded_only.cache.enabled
    assert str(decoded_only.parquet.rg_cache.root) == "/decoded"


def test_decoded_cache_rejects_invalid_budgets_through_zephon() -> None:
    with pytest.raises(ValueError, match="limit_bytes must be positive"):
        ZephonParquetRGCacheOptions(limit_bytes=0).build()
    with pytest.raises(ValueError, match="min_free_bytes cannot be negative"):
        ZephonParquetRGCacheOptions(min_free_bytes=-1).build()
