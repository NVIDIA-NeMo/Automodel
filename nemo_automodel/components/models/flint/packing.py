# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Flint text-token ownership across TE context and pipeline parallelism."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    round_robin_local_indices,
    shard_batch_aux_only,
)
from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from nemo_automodel.components.models.flint.model import FlintForCausalLM


def _prepare_inputs_for_cp(
    model: FlintForCausalLM,
    batch: dict[str, Any],
    *,
    num_chunks: int,
) -> dict[str, Any]:
    """Choose a shared text sharder while preserving PP's global-id boundary.

    Args:
        model: Native model, possibly one pipeline stage, with optional CP mesh.
        batch: Global token tensors [B,S] and optional document lengths [B,N].
        num_chunks: Number of independent pipeline microbatch token spaces.

    Returns:
        A model-owned sharder and padding metadata. Packed local indices have
        shape [Tlocal] or [chunks,Tlocal]; PP input ids stay [chunks,Tglobal],
        while labels, positions and activations are CP-local exactly once.
    """
    if num_chunks < 1:
        raise ValueError("Flint CP num_chunks must be positive")
    if model.backend.attn == "magi":
        from nemo_automodel.components.models.flint.magi import _prepare_magi_inputs

        return _prepare_magi_inputs(model, batch, num_chunks=num_chunks)
    if model.backend.attn != "te":
        if model.cp_mesh is not None and model.cp_mesh.size() > 1:
            raise ValueError("Flint context parallelism requires Transformer Engine learned-sink attention")
    if batch.get("qkv_format") != "thd":
        prepared: dict[str, Any] = {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }
        mask = batch.get("attention_mask")
        if batch.get("padding_mask") is None and isinstance(mask, torch.Tensor) and mask.ndim == 2:
            prepared["padding_mask"] = ~mask.bool()
        if (
            model.cp_mesh is not None
            and model.cp_mesh.size() > 1
            and isinstance(mask, torch.Tensor)
            and not bool(mask.bool().all())
        ):
            raise ValueError("Padded Flint context-parallel batches require document-aware THD metadata")
        return prepared
    if model.backend.attn != "te":
        raise ValueError("Flint packed THD context/pipeline inputs require Transformer Engine")
    pp_stage = model._is_pipeline_parallel_stage()
    if pp_stage and batch["input_ids"].shape[0] != num_chunks:
        raise ValueError("Flint THD pipeline parallelism requires one physical packed row per microbatch")

    def shard_thd(
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        local_batch: dict[str, Any],
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[Any, dict[str, Any], ShardLayout | None]:
        """Use shared TE partitioning and retain each PP chunk's global indices.

        Args:
            cp_mesh: Optional one-dimensional context-parallel mesh.
            tp_mesh: Unused tensor-parallel mesh.
            local_batch: Full token/target tensors [B,S], lengths [B,N].
            loss_mask: Optional [B,S] training mask, owned by the shared sharder.
            padding_token_id: Dataset pad token; real-token validity comes from
                sequence-length metadata, never equality to this token id.

        Returns:
            Null context, local batch, and optional global token-coordinate map.
            Targets retain [1,Tlocal] for the native head/loss contract.
        """
        del tp_mesh
        source_ids = local_batch["input_ids"]
        source_loss_mask = loss_mask if loss_mask is not None else local_batch.get("loss_mask")
        if source_loss_mask is not None and source_loss_mask.shape != source_ids.shape:
            raise ValueError("Flint THD loss_mask must match global input_ids [batch,sequence]")
        row_shape = tuple(source_ids.shape[:2])
        sharded, indices = make_cp_batch_for_te(
            cp_mesh,
            local_batch,
            padding_token_id=padding_token_id,
            qkv_format="thd",
            num_chunks=num_chunks,
            seq_lens_padding_value=-1000,
            return_local_indices=True,
        )
        if num_chunks > 1:
            cp_size = cp_mesh.size() if cp_mesh is not None else 1
            local_tokens = sharded["input_ids"].shape[-1]
            if cp_size > 1:
                available, tex = safe_import("transformer_engine_torch")
                if not available:
                    raise ImportError("Flint packed CP requires transformer_engine_torch")
                indices = torch.stack(
                    [
                        tex.thd_get_partitioned_indices(
                            cu[cu != -1000],
                            local_tokens * cp_size,
                            cp_size,
                            sharded["cp_rank"],
                        )
                        for cu in sharded["cu_seqlens"]
                    ]
                )
            else:
                indices = torch.arange(local_tokens, device=source_ids.device).expand(num_chunks, -1)
        if source_loss_mask is not None:
            if num_chunks > 1:
                sharded["loss_mask"] = source_loss_mask.reshape(num_chunks, -1).gather(1, indices.long())
            else:
                sharded["loss_mask"] = source_loss_mask.flatten().index_select(0, indices.long()).unsqueeze(0)
        if pp_stage:
            # Stage zero receives metadata-sized global IDs, then selects its
            # local IDs before embedding. Later stages receive local activations.
            sharded["input_ids"] = source_ids.reshape(num_chunks, -1).long().contiguous()
        for key in ("labels", "loss_mask"):
            value = sharded.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                sharded[key] = value.unsqueeze(0)
        if pp_stage:
            for key in ("position_ids", "padding_mask"):
                value = sharded.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 1:
                    sharded[key] = value.unsqueeze(0)
        sharded["_thd_local_indices"] = indices
        layout = (
            None
            if indices.ndim == 2
            else ShardLayout(
                local_token_global_indices=indices,
                padded_seq_len=row_shape[0] * row_shape[1],
                input_row_shape=row_shape,
            )
        )
        return contextlib.nullcontext, sharded, layout

    return {"cp_sharder": ContextParallelSharder(shard_batch=shard_thd)}
