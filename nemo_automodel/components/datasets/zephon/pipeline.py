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

"""Construct pipelines exclusively through Zephon's public API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import torch.distributed as dist

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from zephon import Pipeline

    from nemo_automodel.components.datasets.zephon.config import ZephonOptions

logger = logging.getLogger(__name__)


class _SourceOverrides(TypedDict, total=False):
    stop_after_passes: int


class _PrefetchOverrides(TypedDict, total=False):
    buffer_size: int
    parallelism: int


def create_source_pipeline(
    options: ZephonOptions,
    *,
    seed: int,
    dp_rank: int,
    dp_world_size: int,
    measure: Callable[[Any], int] | None = None,
) -> Pipeline:
    """Create the shared source/runtime backbone before objective-specific stages.

    ``measure`` is an adapter-owned token counter for pretokenized or multimodal
    documents. Raw text uses Zephon's built-in tokenizer counting.
    """
    available, zephon = safe_import("zephon")
    if not available:
        raise ImportError("Zephon is a core dependency; install AutoModel with uv sync")
    _, io = safe_import("zephon.io")
    _, work = safe_import("zephon.work")
    if not 0 <= dp_rank < dp_world_size:
        raise ValueError("Invalid Zephon data-parallel rank/world size")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    canonical = options.checkpoint.canonical_replicas
    if canonical is not None and canonical % dp_world_size:
        raise ValueError("canonical_replicas must be a positive multiple of the DP world size")
    if world_size > 1 and not options.checkpoint.run_id:
        raise ValueError("Distributed Zephon checkpointing requires an explicit run_id")
    datasets = [io.Dataset.from_path(name=source.name, path=source.path, fmt=source.fmt) for source in options.sources]
    estimation = options.source.token_estimation
    if estimation == "auto":
        # This is the integration's text mixing policy, not Zephon's default.
        estimation = len(options.sources) > 1
    if isinstance(estimation, dict):
        token_estimation = work.TokenEstimation(measure=measure, **estimation)
    elif estimation:
        token_estimation = work.TokenEstimation(measure=measure)
    else:
        token_estimation = None
    overrides: _SourceOverrides = {}
    if options.source.stop_after_passes is not None:
        overrides["stop_after_passes"] = options.source.stop_after_passes
    logger.info(
        "Zephon token estimation: configured=%s, enabled=%s",
        options.source.token_estimation,
        token_estimation is not None,
    )
    # Omit an unset pass limit so Zephon owns its default and resume semantics.
    source = work.StaticMixtureWorkSource(
        datasets=datasets,
        mixture=work.MixtureSpec({source.name: source.weight for source in options.sources}),
        chunk_size=options.source.chunk_size,
        token_estimation=token_estimation,
        seed=seed,
        shuffle_shards=options.source.shuffle_shards,
        shuffle_within_shard=options.source.shuffle_within_shard,
        shuffle_block_size=options.source.shuffle_block_size,
        exhausted_policy=options.source.exhausted_policy,
        reshuffle_on_repeat=options.source.reshuffle_on_repeat,
        max_repeats=options.source.max_repeats,
        **overrides,
    )
    pipeline = zephon.Pipeline(source)
    prefetch = options.source.prefetch
    if prefetch is not None:
        # Presence inserts cache warmup before Fetch; unset fields leave the
        # operator's defaults to Zephon, independently of runtime queue sizing.
        prefetch_overrides: _PrefetchOverrides = {}
        if prefetch.buffer_size is not None:
            prefetch_overrides["buffer_size"] = prefetch.buffer_size
        if prefetch.parallelism is not None:
            prefetch_overrides["parallelism"] = prefetch.parallelism
        pipeline.prefetch(**prefetch_overrides)
    if options.workers.fetch_parallelism is not None:
        pipeline.fetch_parallelism(options.workers.fetch_parallelism)
    return pipeline


def finalize_pipeline(
    pipeline: Pipeline,
    options: ZephonOptions,
    *,
    batch_size: int,
    seed: int,
    dp_rank: int,
    dp_world_size: int,
) -> None:
    """Batch complete documents and configure/validate the shared runtime.

    Call after objective-specific encoding and packing.
    The pipeline is mutated in place; no iteration or checkpoint is performed.
    The pipeline ends at batch grouping. The loader calls to_training and applies
    model-input adaptations after receiving each original SampleBatch.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    canonical = (
        options.checkpoint.canonical_replicas if options.checkpoint.canonical_replicas is not None else dp_world_size
    )
    if options.packing.shuffle_after_pack:
        pipeline.shuffle(
            seed=seed,
            buffer_size=options.packing.shuffle_buffer_size,
            parallelism=options.workers.shuffle_parallelism,
        )
    pipeline.batch(batch_size, drop_last=True).options(
        runner=options.runtime.runner,
        deterministic=True,
        dp_degree=dp_world_size,
        dp_group_id=dp_rank,
        canonical_replicas=canonical,
        mtp_mode=options.runtime.mtp_mode,
        # Forward None unchanged: Zephon owns automatic sizing. Explicit zero
        # disables optional prefetch buffers and must not be replaced by a default.
        prefetch_batches=options.runtime.prefetch_batches,
        default_stage_prefetch=options.runtime.default_stage_prefetch,
        op_queue_capacity=options.runtime.op_queue_capacity,
        mtp_buffer=options.runtime.mtp_buffer,
        mtp_prefetch=options.runtime.mtp_prefetch,
    )
    if options.cache.cache_dir is not None or options.cache.parquet_rg_cache is not None:
        pipeline.options(io_options=options.cache.build())
    if dist.is_initialized():
        pipeline.options(world_size=world_size, global_rank=dist.get_rank())
    if options.checkpoint.aggregate_dir is not None:
        pipeline.options(aggregate_dir=options.checkpoint.aggregate_dir)
    aggregate_store = options.checkpoint.aggregate_store
    if aggregate_store is None and options.checkpoint.aggregate_dir is None and dist.is_initialized():
        aggregate_store = "torch://"
    if aggregate_store is not None:
        pipeline.options(aggregate_store=aggregate_store)
    if options.checkpoint.run_id is not None:
        pipeline.options(run_id=options.checkpoint.run_id)
    pipeline.preflight_tokenizers(strict=True)
    report = pipeline.validate(strict=True)
    for warning in report.warnings:
        logger.warning("Zephon pipeline validation: %s", warning)
