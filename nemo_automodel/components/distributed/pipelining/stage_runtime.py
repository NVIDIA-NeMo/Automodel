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

import logging
import time
import types
from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import _PipelineSchedule

from nemo_automodel.components.distributed.pipelining import schedules
from nemo_automodel.components.distributed.pipelining.model_parts import PipelineModelPart
from nemo_automodel.shared.pipeline import PipelineModelMixin, causal_lm_stage_metas

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRuntime:
    """PyTorch runtime objects built around materialized model parts."""

    schedule: _PipelineSchedule
    stages: list[PipelineStage]


@torch.no_grad()
def scale_grads_by_divisor(stages: list[PipelineStage], divisor: int) -> None:
    """Scale pipeline-stage gradients by a common divisor when supported."""
    for stage in stages:
        if hasattr(stage, "scale_grads"):
            stage.scale_grads(divisor)


def _get_hidden_and_vocab_size(model_config) -> tuple[int, int]:
    """Extract hidden and vocabulary sizes from flat or nested model configs."""
    hidden_size = getattr(model_config, "hidden_size", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    text_config = getattr(model_config, "text_config", None)

    if hidden_size is None and text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
    if vocab_size is None and text_config is not None:
        vocab_size = getattr(text_config, "vocab_size", None)

    if hidden_size is None:
        raise ValueError(
            f"Cannot determine hidden_size from {type(model_config).__name__}. "
            "Expected either model_config.hidden_size or model_config.text_config.hidden_size."
        )
    if vocab_size is None:
        raise ValueError(
            f"Cannot determine vocab_size from {type(model_config).__name__}. "
            "Expected either model_config.vocab_size or model_config.text_config.vocab_size."
        )
    return hidden_size, vocab_size


def _get_stage_metas(
    part: PipelineModelPart,
    model_config,
    microbatch_size: int,
    seq_len: int,
    tensor_dtype: torch.dtype | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Return analytical input and output metadata for one pipeline part."""
    if tensor_dtype is None:
        try:
            model_dtype = next(part.module.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16
    else:
        model_dtype = tensor_dtype

    if isinstance(part.module, PipelineModelMixin):
        metadata = part.module.pipeline_stage_metas(
            is_first=part.is_first,
            microbatch_size=microbatch_size,
            seq_len=seq_len,
            dtype=model_dtype,
        )
        if metadata is not None:
            return metadata

    hidden_size, vocab_size = _get_hidden_and_vocab_size(model_config)
    return causal_lm_stage_metas(
        is_first=part.is_first,
        has_lm_head=getattr(part.module, "lm_head", None) is not None,
        emits_hidden_states=getattr(part.module, "_pp_return_hidden_states", False) is True,
        microbatch_size=microbatch_size,
        input_seq_len=seq_len,
        output_seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dtype=model_dtype,
    )


def create_pipeline_stages(
    parts: list[PipelineModelPart],
    pp_mesh: DeviceMesh,
    pp_axis_name: str,
    device: torch.device,
    *,
    model_config,
    microbatch_size: int,
    seq_len: int | None,
    tensor_dtype: torch.dtype | None = None,
    first_stage_input_meta: torch.Tensor | None = None,
) -> list[PipelineStage]:
    """Create PyTorch stages from already-parallelized model parts."""
    stages = []
    group = pp_mesh.get_group(pp_axis_name)
    for part in parts:
        metadata = {}
        if seq_len is not None:
            inputs_meta, outputs_meta = _get_stage_metas(
                part,
                model_config,
                microbatch_size,
                seq_len,
                tensor_dtype=tensor_dtype,
            )
            if part.is_first and first_stage_input_meta is not None:
                inputs_meta = (first_stage_input_meta,)
            metadata = {"input_args": inputs_meta, "output_args": outputs_meta}
        try:
            stage = PipelineStage(
                part.module,
                part.stage_index,
                part.num_stages,
                device,
                group=group,
                **metadata,
            )
        except TypeError as exc:
            if metadata:
                raise RuntimeError(
                    "Pipeline parallelism with static shapes requires a PyTorch PipelineStage "
                    "constructor that accepts input_args and output_args"
                ) from exc
            raise
        stages.append(stage)

    if seq_len is not None:
        logger.info(
            "Created pipeline stages with static shapes (seq_len=%d, microbatch_size=%d)",
            seq_len,
            microbatch_size,
        )
    return stages


def warmup_pipeline_stage_neighbors(stage: PipelineStage) -> None:
    """Initialize the pairwise NCCL communicators used by a pipeline stage."""
    warmup_tensor = torch.zeros(1, device=stage.device)
    group_rank = stage.group.rank()
    group_ranks = torch.distributed.get_process_group_ranks(stage.group)
    local_world_size = max(torch.cuda.device_count(), 1)
    time.sleep(2 * (min(group_ranks) // local_world_size))

    edges = {(rank, rank + 1) for rank in range(stage.group_size - 1)}
    if stage.group_size > 2:
        edges.add((0, stage.group_size - 1))

    edge_phases: list[list[tuple[int, int]]] = []
    phase_ranks: list[set[int]] = []
    for edge in sorted(edges):
        for phase, ranks in zip(edge_phases, phase_ranks):
            if ranks.isdisjoint(edge):
                phase.append(edge)
                ranks.update(edge)
                break
        else:
            edge_phases.append([edge])
            phase_ranks.append(set(edge))

    for phase in edge_phases:
        for reverse in (False, True):
            for lower_rank, higher_rank in phase:
                src_rank, dst_rank = (higher_rank, lower_rank) if reverse else (lower_rank, higher_rank)
                if group_rank == src_rank:
                    torch.distributed.isend(warmup_tensor, group=stage.group, group_dst=dst_rank).wait()
                elif group_rank == dst_rank:
                    torch.distributed.irecv(warmup_tensor, group=stage.group, group_src=src_rank).wait()

    torch.cuda.synchronize(stage.device)


def configure_pipeline_stage_backward(
    stages: list[PipelineStage],
    *,
    patch_stage_backward_maybe_with_nosync: bool,
    reduce_grad_per_microbatch: bool,
) -> None:
    """Configure the pipeline-stage backward policy."""
    if not patch_stage_backward_maybe_with_nosync and not reduce_grad_per_microbatch:
        return

    from nemo_automodel.components.moe.fsdp_mixin import patched_backward_maybe_with_nosync

    for stage in stages:
        stage.backward_maybe_with_nosync = types.MethodType(patched_backward_maybe_with_nosync, stage)
        stage._reduce_grad_per_microbatch = reduce_grad_per_microbatch

    logger.info(
        "Patched pipeline stages with backward_maybe_with_nosync (reduce_grad_per_microbatch=%s)",
        reduce_grad_per_microbatch,
    )


def build_pipeline_runtime(
    parts: list[PipelineModelPart],
    pp_mesh: DeviceMesh,
    pp_axis_name: str,
    device: torch.device,
    *,
    model_config,
    microbatch_size: int,
    local_batch_size: int,
    seq_len: int | None,
    tensor_dtype: torch.dtype | None,
    first_stage_input_meta: torch.Tensor | None,
    schedule_name: str | None,
    schedule_csv: str | None,
    loss_fn: Callable,
    scale_grads: bool,
    warmup_neighbors: bool,
    patch_stage_backward_maybe_with_nosync: bool,
    reduce_grad_per_microbatch: bool,
) -> PipelineRuntime:
    """Build stages and their schedule around existing model parts."""
    stages = create_pipeline_stages(
        parts,
        pp_mesh,
        pp_axis_name,
        device,
        model_config=model_config,
        microbatch_size=microbatch_size,
        seq_len=seq_len,
        tensor_dtype=tensor_dtype,
        first_stage_input_meta=first_stage_input_meta,
    )
    if warmup_neighbors and seq_len is not None:
        warmup_pipeline_stage_neighbors(stages[0])

    schedule = schedules.build_pipeline_schedule(
        schedule_csv,
        schedule_name,
        microbatch_size,
        local_batch_size,
        stages,
        loss_fn,
        scale_grads=scale_grads,
    )
    configure_pipeline_stage_backward(
        stages,
        patch_stage_backward_maybe_with_nosync=patch_stage_backward_maybe_with_nosync,
        reduce_grad_per_microbatch=reduce_grad_per_microbatch,
    )
    return PipelineRuntime(schedule=schedule, stages=stages)
