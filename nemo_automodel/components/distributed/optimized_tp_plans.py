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

"""Shared tensor-parallel styles used by model-owned TP plans.

The plans themselves live with their models (``components/models/<name>/parallelization.py``)
or, for architectures the repository does not implement, in ``_transformers/hf_parallel_specs.py``.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard


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


# Legacy YAML alias for ``tp_shard_plan``; ``_get_parallel_plan`` now defers to the model's own plan.
LLAMA_NEMOTRON_SUPER_TP_PLAN_NAME = "llama_nemotron_super_tp_plan"
