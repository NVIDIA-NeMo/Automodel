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
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from nemo_automodel.components.datasets.zephon.llm import resolve_token_ids
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from zephon import Pipeline

    from nemo_automodel.components.datasets.zephon.config import ZephonDataloaderConfig, ZephonOptions

logger = logging.getLogger(__name__)


def create_source_pipeline(
    options: ZephonOptions,
    *,
    dp_rank: int,
    dp_world_size: int,
    measure: Callable[[Any], int] | None = None,
) -> Pipeline:
    """Create the shared source/runtime backbone before objective-specific stages.

    ``measure`` is an adapter-owned token counter for encoded or multimodal
    documents. Raw text and SFT use Zephon's built-in tokenizer counting.
    """
    available, zephon = safe_import("zephon")
    if not available:
        raise ImportError("Zephon is a core dependency; install AutoModel with uv sync")
    _, io = safe_import("zephon.io")
    _, work = safe_import("zephon.work")
    if not 0 <= dp_rank < dp_world_size:
        raise ValueError("Invalid Zephon data-parallel rank/world size")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    canonical = (
        options.checkpoint.canonical_replicas if options.checkpoint.canonical_replicas is not None else dp_world_size
    )
    if canonical <= 0 or canonical % dp_world_size:
        raise ValueError("canonical_replicas must be a positive multiple of the DP world size")
    if world_size > 1 and not options.checkpoint.run_id:
        raise ValueError("Distributed Zephon checkpointing requires an explicit run_id")
    datasets = [io.Dataset.from_path(name=source.name, path=source.path, fmt=source.fmt) for source in options.sources]
    estimation = options.source.token_estimation
    if estimation is None:
        estimation = len(options.sources) > 1
    token_estimation = (
        work.TokenEstimation(measure=measure, **estimation)
        if isinstance(estimation, dict)
        else (work.TokenEstimation(measure=measure) if estimation else None)
    )
    source = work.StaticMixtureWorkSource(
        datasets=datasets,
        mixture=work.MixtureSpec({source.name: source.weight for source in options.sources}),
        chunk_size=options.source.chunk_size,
        token_estimation=token_estimation,
        seed=options.seed,
        shuffle_shards=options.source.shuffle_shards,
        shuffle_within_shard=options.source.shuffle_within_shard,
        shuffle_block_size=options.source.shuffle_block_size,
        exhausted_policy=options.source.exhausted_policy,
        reshuffle_on_repeat=options.source.reshuffle_on_repeat,
        max_repeats=options.source.max_repeats,
        stop_after_passes=(
            options.source.stop_after_passes
            if options.source.stop_after_passes is not None
            else (1 if options.source.exhausted_policy is None and options.source.max_repeats is None else None)
        ),
    )
    pipeline = zephon.Pipeline(source)
    if options.runtime.prefetch_buffer_size:
        pipeline.prefetch(
            buffer_size=options.runtime.prefetch_buffer_size, parallelism=options.workers.prefetch_parallelism
        )
    if options.workers.fetch_parallelism is not None:
        pipeline.fetch_parallelism(options.workers.fetch_parallelism)
    return pipeline


def finalize_pipeline(
    pipeline: Pipeline,
    options: ZephonOptions,
    *,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
) -> None:
    """Batch complete documents and configure/validate the shared runtime.

    Call after objective-specific encoding and packing.
    The pipeline is mutated in place; no iteration or checkpoint is performed.
    The pipeline ends at batch grouping. The loader calls to_training and applies
    model-input adaptations after receiving each original SampleBatch.
    """
    _, io = safe_import("zephon.io")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    canonical = (
        options.checkpoint.canonical_replicas if options.checkpoint.canonical_replicas is not None else dp_world_size
    )
    if options.packing.shuffle_after_pack:
        pipeline.shuffle(
            seed=options.seed,
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
    )
    if options.cache.cache_dir is not None:
        pipeline.options(
            io_options=io.StoreOptions(
                cache=io.CacheOptions.from_any(
                    {"enabled": True, "root": options.cache.cache_dir, "limit_bytes": options.cache.cache_limit_bytes}
                )
            )
        )
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


def create_text_pipeline(
    config: ZephonDataloaderConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    dp_rank: int,
    dp_world_size: int,
) -> Pipeline:
    """Materialize text mixtures with public token estimation, packing and batching."""
    options = config.zephon
    if tokenizer is None:
        raise ValueError("Zephon requires the resolved model tokenizer")
    ids = resolve_token_ids(tokenizer, pad_token_id=options.packing.pad_token_id)
    max_documents = options.packing.max_num_documents
    if max_documents is not None and options.packing.max_num_documents_scope == "batch":
        max_documents //= config.batch_size
        if not max_documents:
            raise ValueError("max_num_documents must be at least local_batch_size when scoped to the batch")
    wrapping = options.packing.pack_algorithm in ("wrap", "best_fit_wrap")
    num_bins = (options.packing.pack_num_bins or 64) if not wrapping else None
    pipeline = config.dataset_config.build(
        tokenizer=tokenizer, options=options, ids=ids, dp_rank=dp_rank, dp_world_size=dp_world_size
    )
    pipeline.pack_flat(
        max_length=options.packing.seq_len + 1,
        algorithm=options.packing.pack_algorithm,
        num_bins=num_bins,
        max_sequences_per_bin=max_documents,
        pad_token_id=ids.pad,
        drop_oversized=options.packing.drop_oversized,
        emit_positions=True,
        parallelism=options.workers.pack_parallelism,
    )
    finalize_pipeline(pipeline, options, batch_size=config.batch_size, dp_rank=dp_rank, dp_world_size=dp_world_size)
    logger.info(
        "Zephon stages: %s -> pack_flat(%s, %s) -> batch(%s); token_estimation=%s; sources=%s",
        type(config.dataset_config).__name__,
        options.packing.pack_algorithm,
        options.packing.seq_len + 1,
        config.batch_size,
        options.source.token_estimation,
        [(source.name, source.weight) for source in options.sources],
    )
    return pipeline
