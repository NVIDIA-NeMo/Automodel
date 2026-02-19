# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)


def _distribute_param(_module, name, device_mesh, src_data_rank, placements):
    param = getattr(_module, name)
    dist_param = nn.Parameter(
        distribute_tensor(param, device_mesh, placements, src_data_rank=src_data_rank),
        requires_grad=param.requires_grad,
    )
    assert dist_param.requires_grad == param.requires_grad
    _module.register_parameter(name, dist_param)


class FusedColwiseParallel(ColwiseParallel):
    """Column-wise parallelism for fused Q/K/V (or Q/KV) linear projections.

    A fused QKV linear has weight whose output is the concatenation
    ``[Q | K | V]``.  Standard ``ColwiseParallel`` shards the first
    dimension contiguously, which crosses Q/K/V boundaries and mixes
    heads from different projections on different TP ranks.

    ``FusedColwiseParallel`` instead splits the weight (and bias) into
    sections and shards **each section** independently with ``Shard(0)``,
    then concatenates the local shards.  This ensures every rank receives
    the correct head subset from every section.

    Sections can be either:

    - **Equal-sized** (default): specified via ``num_sections``
      (e.g. ``num_sections=3`` for MHA where Q, K, V have the same size,
      or ``num_sections=2`` for fused gate_up projections).
    - **Variable-sized**: specified via ``section_sizes``
      (e.g. ``section_sizes=(q_size, kv_size, kv_size)`` for GQA where Q
      has more heads than K/V).  When provided, takes precedence over
      ``num_sections``.

    YAML string: ``"fused_colwise"``

    Note on checkpointing
    ---------------------
    Because the local shard concatenates non-contiguous slices of the
    original weight, ``DTensor.full_tensor()`` will reconstruct a
    **permuted** weight (heads interleaved across sections) rather than
    the original ``[Q | K | V]`` layout.  Distributed-checkpoint
    save/load at the **same** TP degree works correctly; cross-TP-degree
    resharding requires a state-dict adapter that re-partitions the fused
    QKV weight.

    Args:
        num_sections: Number of equal-sized fused sections.  Default ``3``
            for Q/K/V.  Ignored when ``section_sizes`` is provided.
        section_sizes: Explicit per-section sizes along dim-0.  Each
            section must be independently divisible by the TP world size.
            When provided, takes precedence over ``num_sections``.
    """

    def __init__(
        self,
        *,
        num_sections: int = 3,
        section_sizes: Optional[tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_sections = num_sections
        self.section_sizes = section_sizes

    @staticmethod
    def _set_param(module, dotted_name, param):
        """Register *param* on the correct (sub)module for *dotted_name*."""
        parts = dotted_name.rsplit(".", 1)
        if len(parts) == 1:
            module.register_parameter(parts[0], param)
        else:
            submod = module.get_submodule(parts[0])
            submod.register_parameter(parts[1], param)

    def _partition_linear_fn(self, name, module, device_mesh):
        for pname, param in list(module.named_parameters()):
            if isinstance(param, DTensor):
                continue  # already distributed

            dim0 = param.shape[0]

            # Check whether this parameter has the fused output dimension.
            # Adapter parameters (e.g. LoRA lora_A) may have a different
            # dim-0; fall back to plain Shard(0) for those.
            if self.section_sizes is not None:
                needs_fused_split = sum(self.section_sizes) == dim0
            else:
                needs_fused_split = dim0 % self.num_sections == 0

            if not needs_fused_split:
                dist_param = nn.Parameter(
                    distribute_tensor(param.data, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                )
                self._set_param(module, pname, dist_param)
                continue

            if self.section_sizes is not None:
                sections = param.data.split(list(self.section_sizes), dim=0)
            else:
                sections = param.data.chunk(self.num_sections, dim=0)

            local_parts = []
            for sec in sections:
                dt = distribute_tensor(sec, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
                local_parts.append(dt.to_local())

            local_data = torch.cat(local_parts, dim=0)

            dist_param = nn.Parameter(
                DTensor.from_local(
                    local_data,
                    device_mesh,
                    [Shard(0)],
                    run_check=False,
                )
            )
            self._set_param(module, pname, dist_param)


class FusedColwiseParallelLora(FusedColwiseParallel):
    """LoRA-aware ``FusedColwiseParallel``.

    The parent class already handles LoRA adapter parameters gracefully
    (falls back to plain ``Shard(0)`` when dim-0 doesn't match the fused
    output dimension).  This subclass adds a forward hook on ``lora_A``
    that all-gathers its sharded output before it is projected by
    ``lora_B``, mirroring the pattern used in ``ColwiseParallelLora``.
    """

    def _partition_linear_fn(self, name, module, device_mesh):
        super()._partition_linear_fn(name, module, device_mesh)

        def lora_a_output_hook(module, input, output):
            if isinstance(output, DTensor):
                if any(isinstance(p, Shard) for p in output.placements):
                    output = output.redistribute(
                        device_mesh=output.device_mesh, placements=[Replicate()]
                    )
            return output

        if hasattr(module, "lora_A"):
            module.lora_A.register_forward_hook(lora_a_output_hook)


class ColwiseParallelLora(ColwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        def _get_module_and_name(module, name):
            if name.endswith("lora_A.weight"):
                assert hasattr(module, "lora_A"), f"lora_A not found in {module}"
                return module.lora_A, "weight"
            elif name.endswith("lora_B.weight"):
                assert hasattr(module, "lora_B"), f"lora_B not found in {module}"
                return module.lora_B, "weight"
            else:
                return module, name

        for name, param in module.named_parameters():
            _module, _name = _get_module_and_name(module, name)
            _distribute_param(_module, _name, device_mesh, self.src_data_rank, [Shard(0)])

        # Register forward hook on lora_A to all-gather its low rank output
        def lora_a_output_hook(module, input, output):
            if isinstance(output, DTensor):
                if any(isinstance(p, Shard) for p in output.placements):
                    output = output.redistribute(device_mesh=output.device_mesh, placements=[Replicate()])
            return output

        if hasattr(module, "lora_A"):
            module.lora_A.register_forward_hook(lora_a_output_hook)

    def _partition_embedding_fn(self, name, module, device_mesh):
        # colwise shard embedding.weight is straight forward as Shard(1)
        for name, param in module.named_parameters():
            _distribute_param(module, name, device_mesh, self.src_data_rank, [Shard(1)])


class RowwiseParallelLora(RowwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        _distribute_param(module, "weight", device_mesh, self.src_data_rank, [Shard(1)])
        if getattr(module, "bias", None) is not None:
            _distribute_param(module, "bias", device_mesh, self.src_data_rank, [Replicate()])
        if hasattr(module, "lora_A"):
            _distribute_param(module.lora_A, "weight", device_mesh, self.src_data_rank, [Shard(1)])
            _distribute_param(module.lora_B, "weight", device_mesh, self.src_data_rank, [Shard(1)])
        if hasattr(module, "lora_magnitude"):
            _distribute_param(module, "lora_magnitude", device_mesh, self.src_data_rank, [Replicate()])

    def _partition_embedding_fn(self, name, module, device_mesh):
        # rowwise shard embedding.weight is Shard(0)
        for name, param in module.named_parameters():
            _distribute_param(module, name, device_mesh, self.src_data_rank, [Shard(0)])


class SequenceParallelLora(SequenceParallel):
    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(p_name, replicated_param)


def translate_to_lora(plan):
    CLS_MAP = {
        ColwiseParallel: ColwiseParallelLora,
        FusedColwiseParallel: FusedColwiseParallelLora,
        RowwiseParallel: RowwiseParallelLora,
        SequenceParallel: SequenceParallelLora,
    }
    plan.__class__ = CLS_MAP.get(type(plan), plan.__class__)
    return plan


class SequenceParallelAllGatherActivation(SequenceParallel):
    """SequenceParallel that all-gathers activations for sequence parallelism."""

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        """Prepare outputs by redistributing sharded DTensors to replicated placement."""
        # If output is a DTensor with Shard placement, redistribute to Replicate
        if isinstance(outputs, DTensor):
            if any(isinstance(p, Shard) for p in outputs.placements):
                # Redistribute to replicated placement (performs all-gather)
                outputs = outputs.redistribute(device_mesh=device_mesh, placements=[Replicate()])
        else:
            raise ValueError(f"Expected output to be a DTensor, but got {type(outputs)}")

        # Call the parent's prepare_output_fn to handle use_local_output
        return SequenceParallel._prepare_output_fn(use_local_output, mod, outputs, device_mesh)


class VocabParallelEmbedding(RowwiseParallel):
    """``RowwiseParallel`` for ``nn.Embedding`` with a ``MaskPartial`` mask-buffer fixup.

    Some PyTorch versions have a DTensor bug where the ``MaskPartial``
    placement's ``mask_buffer`` is not populated during the embedding
    dispatch, leading to::

        AssertionError: assert self.mask_buffer.data is not None

    This subclass works around the issue by:

    1. Saving the *original* (un-adjusted) ``input_ids`` in a pre-hook.
    2. Recomputing and populating the ``mask_buffer`` in the post-hook
       when the DTensor dispatch failed to do so.

    In PyTorch versions where the dispatch works correctly the mask buffer
    is already populated and the fixup is a no-op.
    """

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # Save the original input_ids (before DTensor index-adjustment)
        # so we can recompute the mask in the output hook if needed.
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            mod._vocab_parallel_saved_ids = input_tensor.to_local().clone()
        else:
            mod._vocab_parallel_saved_ids = input_tensor.clone()

        return RowwiseParallel._prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        saved_ids = getattr(mod, "_vocab_parallel_saved_ids", None)
        if saved_ids is not None:
            delattr(mod, "_vocab_parallel_saved_ids")

        # If the output is a DTensor whose MaskPartial placement has an
        # empty mask_buffer, compute and materialise the mask so that the
        # subsequent ``_reduce_value`` / ``_reduce_shard_value`` succeeds.
        if isinstance(outputs, DTensor) and saved_ids is not None:
            placement = outputs.placements[0]
            mb = getattr(placement, "mask_buffer", None)
            if mb is not None and getattr(mb, "data", ...) is None:
                vocab_size = getattr(mod, "num_embeddings", None) or mod.weight.shape[0]
                tp_size = device_mesh.size()
                rank = device_mesh.get_local_rank()

                chunk = vocab_size // tp_size
                rem = vocab_size % tp_size
                if rank < rem:
                    local_size = chunk + 1
                    local_off = rank * (chunk + 1)
                else:
                    local_size = chunk
                    local_off = rem * (chunk + 1) + (rank - rem) * chunk

                mask = (saved_ids < local_off) | (saved_ids >= local_off + local_size)
                mb.materialize_mask(mask)

        return RowwiseParallel._prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh)


class RotaryEmbedParallel(SequenceParallel):
    """Custom SequenceParallel class for Qwen2 / Gemma3 rotary embeddings because the input is a tuple."""

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        new_inputs = list(inputs)

        if not isinstance(inputs[0], DTensor):
            """Guard the metadata for Sequence Parallel here"""
            try:
                new_inputs[0] = DTensor.from_local(
                    local_tensor=inputs[0],
                    device_mesh=device_mesh,
                    placements=sequence_sharding,
                    run_check=True,
                )
            except ValueError as e:
                raise ValueError(
                    f"Failed to shard tensor for sequence parallelism. Local Shape is ({inputs[0].shape}) "
                    f"at rank {torch.distributed.get_rank()}. Different TP ranks must have the same shape. "
                    f"Original error: {str(e)}"
                ) from e

        if not isinstance(inputs[1], DTensor):
            new_inputs[1] = DTensor.from_local(
                local_tensor=inputs[1],
                device_mesh=device_mesh,
                placements=(Replicate(),),
                run_check=False,
            )

        return type(inputs)(new_inputs)

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return type(outputs)([o.to_local() if use_local_output else o for o in outputs])
