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


"""Shared source construction and runtime options through the public Zephon API."""

from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from zephon import Pipeline
from zephon.work import StaticMixtureWorkSource, TokenEstimation

from nemo_automodel.components.datasets.zephon.config import (
    TokenEstimationSettings,
    ZephonCacheOptions,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonParquetRGCacheOptions,
    ZephonPrefetchOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
)
from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline, finalize_pipeline


@pytest.fixture
def options(tmp_path: Path) -> ZephonOptions:
    path = tmp_path / "documents.jsonl"
    path.write_text('{"input_ids": [2, 3, 4, 5]}\n' * 8)
    return ZephonOptions(
        sources=(ZephonSource(name="text", path=str(tmp_path), fmt="jsonl"),),
        source=ZephonSourceOptions(chunk_size=2, token_estimation=None),
        packing=ZephonPackingOptions(seq_len=3, shuffle_after_pack=False),
        runtime=ZephonRuntimeOptions(runner="threads", mtp_mode=False),
    )


@pytest.mark.parametrize(
    "setting,source_count,enabled",
    [("auto", 1, False), ("auto", 2, True), (False, 2, False), (None, 2, False), (True, 1, True), ({}, 1, True)],
)
def test_token_estimation_policy(
    options: ZephonOptions,
    setting: bool | TokenEstimationSettings | Literal["auto"] | None,
    source_count: int,
    enabled: bool,
) -> None:
    options = replace(
        options,
        sources=tuple(replace(options.sources[0], name=f"source_{i}") for i in range(source_count)),
        source=replace(options.source, token_estimation=setting),
    )
    with (
        patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source,
        patch("zephon.work.TokenEstimation", wraps=TokenEstimation) as estimation,
    ):
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    assert estimation.called is enabled
    assert (source.call_args.kwargs["token_estimation"] is not None) is enabled
    # Resolving an automatic choice must not mutate the reusable config.
    assert options.source.token_estimation == setting


@pytest.mark.parametrize("policy,passes", [(None, 1), ("repeat", None)])
def test_unset_pass_limit_uses_zephon_exhaustion_policy(
    options: ZephonOptions, policy: str | None, passes: int | None
) -> None:
    options = replace(options, source=replace(options.source, exhausted_policy=policy))
    with patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source:
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    kwargs = source.call_args.kwargs
    assert "stop_after_passes" not in kwargs
    actual = StaticMixtureWorkSource(**kwargs)
    assert actual.state_dict()["stop_after_passes"] == passes


def test_explicit_pass_limit_reaches_zephon(options: ZephonOptions) -> None:
    options = replace(options, source=replace(options.source, stop_after_passes=2))
    with patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source:
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    assert source.call_args.kwargs["stop_after_passes"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "prefetch_batches": 24,
            "default_stage_prefetch": 8,
            "op_queue_capacity": 12,
            "mtp_buffer": 6,
            "mtp_prefetch": 2,
        },
        {"prefetch_batches": 0, "default_stage_prefetch": 0, "mtp_prefetch": 0},
    ],
)
def test_runtime_buffers_reach_zephon_without_replacing_none_or_zero(
    options: ZephonOptions, overrides: dict[str, int]
) -> None:
    options = replace(options, runtime=replace(options.runtime, **overrides))
    with patch.object(Pipeline, "options", autospec=True, side_effect=Pipeline.options) as option_calls:
        pipeline = create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
        finalize_pipeline(pipeline, options, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
    forwarded = next(call.kwargs for call in option_calls.call_args_list if "mtp_mode" in call.kwargs)
    for name in ("prefetch_batches", "default_stage_prefetch", "op_queue_capacity", "mtp_buffer", "mtp_prefetch"):
        assert forwarded[name] == overrides.get(name)


@pytest.mark.parametrize(
    "prefetch,expected",
    [
        (None, None),
        (ZephonPrefetchOptions(), {}),
        (ZephonPrefetchOptions(buffer_size=8), {"buffer_size": 8}),
        (ZephonPrefetchOptions(parallelism=3), {"parallelism": 3}),
        (ZephonPrefetchOptions(buffer_size=8, parallelism=3), {"buffer_size": 8, "parallelism": 3}),
    ],
)
def test_source_prefetch_inserts_only_when_enabled_and_delegates_unset_options(
    options: ZephonOptions, prefetch: ZephonPrefetchOptions | None, expected: dict[str, int] | None
) -> None:
    options = replace(options, source=replace(options.source, prefetch=prefetch))
    with patch.object(Pipeline, "prefetch", autospec=True, side_effect=Pipeline.prefetch) as warm_cache:
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    if expected is None:
        warm_cache.assert_not_called()
    else:
        assert warm_cache.call_count == 1
        assert warm_cache.call_args.kwargs == expected


def test_decoded_cache_reaches_shared_runtime_without_enabling_source_cache(options: ZephonOptions) -> None:
    options = replace(
        options,
        cache=ZephonCacheOptions(
            parquet_rg_cache=ZephonParquetRGCacheOptions(root="/decoded", limit_bytes="2gb", min_free_bytes=0)
        ),
    )
    with patch.object(Pipeline, "options", autospec=True, side_effect=Pipeline.options) as calls:
        pipeline = create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
        finalize_pipeline(pipeline, options, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
    io = next(call.kwargs["io_options"] for call in calls.call_args_list if "io_options" in call.kwargs)
    assert not io.cache.enabled
    assert io.parquet.rg_cache.limit_bytes == 2 * 1024**3
    assert io.parquet.rg_cache.min_free_bytes == 0
