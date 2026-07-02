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

"""Reusable tensor-parallel plan primitives for model-owned plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard


class SequenceParallelAllGatherActivation(SequenceParallel):
    """Sequence parallelism that replicates activations after the wrapped module."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        if isinstance(outputs, DTensor):
            if any(isinstance(placement, Shard) for placement in outputs.placements):
                outputs = outputs.redistribute(device_mesh=device_mesh, placements=[Replicate()])
        else:
            raise ValueError(f"Expected a DTensor, got {type(outputs)}")
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


class VocabParallelEmbedding(RowwiseParallel):
    """Row-wise embedding parallelism with PyTorch's MaskPartial fixup."""

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        mod._vocab_parallel_saved_ids = (
            input_tensor.to_local().clone() if isinstance(input_tensor, DTensor) else input_tensor.clone()
        )
        return RowwiseParallel._prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        saved_ids = getattr(mod, "_vocab_parallel_saved_ids", None)
        if saved_ids is not None:
            delattr(mod, "_vocab_parallel_saved_ids")
        if isinstance(outputs, DTensor) and saved_ids is not None:
            placement = outputs.placements[0]
            mask_buffer = getattr(placement, "mask_buffer", None)
            if mask_buffer is not None and getattr(mask_buffer, "data", ...) is None:
                vocab_size = getattr(mod, "num_embeddings", None) or mod.weight.shape[0]
                tp_size = device_mesh.size()
                rank = device_mesh.get_local_rank()
                chunk, remainder = divmod(vocab_size, tp_size)
                local_size = chunk + 1 if rank < remainder else chunk
                local_offset = (
                    rank * (chunk + 1) if rank < remainder else remainder * (chunk + 1) + (rank - remainder) * chunk
                )
                mask_buffer.materialize_mask((saved_ids < local_offset) | (saved_ids >= local_offset + local_size))
        return RowwiseParallel._prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh)


class RotaryEmbedParallel(SequenceParallel):
    """Sequence parallelism for rotary embeddings that receive tuple inputs."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        prepared_inputs = list(inputs)
        if not isinstance(inputs[0], DTensor):
            try:
                prepared_inputs[0] = DTensor.from_local(
                    local_tensor=inputs[0],
                    device_mesh=device_mesh,
                    placements=sequence_sharding,
                    run_check=True,
                )
            except ValueError as error:
                raise ValueError(
                    "Failed to shard tensor for sequence parallelism. "
                    f"Local shape is {inputs[0].shape} at rank {torch.distributed.get_rank()}. "
                    "Different TP ranks must have the same shape. "
                    f"Original error: {error}"
                ) from error
        if not isinstance(inputs[1], DTensor):
            prepared_inputs[1] = DTensor.from_local(
                local_tensor=inputs[1], device_mesh=device_mesh, placements=(Replicate(),), run_check=False
            )
        return type(inputs)(prepared_inputs)

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return type(outputs)([output.to_local() if use_local_output else output for output in outputs])


@dataclass(frozen=True)
class DecoderTPPaths:
    """Concrete module paths declared by one model-local TP sidecar."""

    embedding: str
    layer: str
    norm: str
    output_head: str


def decoder_tp_plan(
    *,
    paths: DecoderTPPaths,
    sequence_parallel: bool = False,
    use_vocab_parallel_embedding: bool = True,
    fused_projections: bool = False,
    activation_aware_norms: bool = True,
    output_head_layout=Shard(-1),
) -> dict[str, ParallelStyle]:
    """Build a shared decoder TP plan from model-local module paths."""
    embedding_style = VocabParallelEmbedding if use_vocab_parallel_embedding else RowwiseParallel
    plan: dict[str, ParallelStyle] = {
        paths.embedding: embedding_style(input_layouts=Replicate()),
        f"{paths.layer}.self_attn.q_proj": ColwiseParallel(),
        f"{paths.layer}.self_attn.k_proj": ColwiseParallel(),
        f"{paths.layer}.self_attn.v_proj": ColwiseParallel(),
        f"{paths.layer}.self_attn.o_proj": RowwiseParallel(),
        f"{paths.layer}.mlp.up_proj": ColwiseParallel(),
        f"{paths.layer}.mlp.gate_proj": ColwiseParallel(),
        f"{paths.layer}.mlp.down_proj": RowwiseParallel(),
        paths.output_head: ColwiseParallel(output_layouts=output_head_layout, use_local_output=False),
    }
    if fused_projections:
        plan.update(
            {
                f"{paths.layer}.self_attn.qkv_proj": ColwiseParallel(),
                f"{paths.layer}.mlp.gate_up_proj": ColwiseParallel(),
            }
        )
    if not sequence_parallel:
        return plan

    norm_style = SequenceParallelAllGatherActivation if activation_aware_norms else SequenceParallel
    plan.update(
        {
            paths.embedding: VocabParallelEmbedding(
                input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=False
            ),
            paths.norm: SequenceParallel(),
            f"{paths.layer}.input_layernorm": norm_style(use_local_output=False)
            if activation_aware_norms
            else norm_style(),
            f"{paths.layer}.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            f"{paths.layer}.post_attention_layernorm": norm_style(use_local_output=False)
            if activation_aware_norms
            else norm_style(),
            f"{paths.layer}.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
            paths.output_head: ColwiseParallel(
                input_layouts=Shard(1), output_layouts=output_head_layout, use_local_output=False
            ),
        }
    )
    return plan
