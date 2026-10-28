# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-owned packed Magi CP with shared token ownership across Muse masks."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder, ShardLayout
from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


class _MagiAttentionKey(Protocol):
    """Magi key fields consumed by the Muse integration."""

    total_seqlen_q: int


class _MuseMagiModel(Protocol):
    """Muse model state needed to build model-owned Magi CP inputs."""

    config: MuseSpark1xOmniConfig
    cp_mesh: DeviceMesh | None

    def _is_pipeline_parallel_stage(self) -> bool:
        """Return whether this module is a physical pipeline stage."""
        ...


@dataclass(frozen=True)
class MuseMagiPlan:
    """Per-microbatch keys; all masks retain the same dispatched token order."""

    keys: dict[int, _MagiAttentionKey]

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, window: int, scale: float
    ) -> torch.Tensor:
        """Compute local THD output with Magi's own KV communication.

        Args:
            query: CP-local queries [tokens, query_heads, head_dim].
            key: CP-local keys [tokens, kv_heads, head_dim].
            value: CP-local values [tokens, kv_heads, head_dim].
            window: Sliding-window length, or zero for full causal attention.
            scale: Attention score scale.

        Returns:
            Attention output [tokens, query_heads, head_dim] in the local token order.
        """
        available, api = safe_import("magi_attention.api")
        if not available:
            raise ImportError("Muse Magi CP requires magi_attention with Blackwell FA4 support")
        output, _ = api.calc_attn(
            query.contiguous(), key.contiguous(), value.contiguous(), self.keys[window], softmax_scale=scale
        )
        return output


@dataclass(frozen=True)
class _MuseMagiBatchPlans:
    """Step-owned plans replicated by PP; a chunked index selects each plan."""

    plans: tuple[MuseMagiPlan, ...]

    def select(self, microbatch_index: torch.Tensor) -> MuseMagiPlan:
        """Select without a mutable cursor, before entering checkpointed layers.

        Args:
            microbatch_index: Int64 tensor [1], split by PP and possibly moved by FSDP.

        Returns:
            The immutable microbatch plan retained by that forward's AC graph.
        """
        if microbatch_index.numel() != 1:
            raise ValueError("Muse Magi PP requires one microbatch index per forward")
        index = int(microbatch_index.item())
        if not 0 <= index < len(self.plans):
            raise IndexError(f"Muse Magi PP microbatch index {index} is outside [0, {len(self.plans)})")
        return self.plans[index]


MuseMagiBatchValue = torch.Tensor | str | int | float | bool | MuseMagiPlan | _MuseMagiBatchPlans | None
MuseMagiBatch = dict[str, MuseMagiBatchValue]


def _build_magi_plan(
    cu_seqlens: torch.Tensor, *, config: MuseSpark1xOmniConfig, group: torch.distributed.ProcessGroup
) -> MuseMagiPlan:
    """Build masks with one shared token partition for a packed microbatch.

    Args:
        cu_seqlens: CPU int32 padded document boundaries [documents + 1].
        config: Muse attention configuration, shared across pipeline stages.
        group: This pipeline stage's context-parallel process group.

    Returns:
        Full-causal and sliding-window keys with identical local token ownership.
    """
    available, api = safe_import("magi_attention.api")
    configs_available, configs = safe_import("magi_attention.config")
    if not available or not configs_available:
        raise ImportError("Muse Magi CP requires magi_attention")
    full_key = api.magi_attn_varlen_key(
        cu_seqlens,
        cu_seqlens,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        0,
        group,
        causal=True,
        dist_attn_config=configs.DistAttnConfig(dispatch_config=configs.DispatchConfig(chunk_size=512)),
    )
    keys = {0: full_key}
    local_indices = api.get_position_ids(full_key)
    for window in sorted({int(w or 0) for w in config.layer_sliding_windows}):
        if window:
            keys[window] = api.make_varlen_key_for_new_mask_after_dispatch(
                cu_seqlens, cu_seqlens, full_key, causal=False, window_size=(window - 1, 0)
            )
            if not torch.equal(api.get_position_ids(keys[window]).cpu(), local_indices.cpu()):
                raise RuntimeError("Muse Magi SWA/full masks must share token ownership")
    return MuseMagiPlan(keys)


def prepare_magi_inputs(
    model: _MuseMagiModel, batch: MuseMagiBatch, *, num_chunks: int
) -> dict[str, ContextParallelSharder]:
    """Prepare model-owned packed CP, including independent pipeline microbatches.

    Args:
        model: Full Muse model or a local pipeline stage.
        batch: Global tokens/labels/positions [batch, sequence] and packed
            document lengths [batch, documents]. Each PP microbatch holds one pack.
        num_chunks: Number of independent pipeline microbatches.

    Returns:
        A model-owned CP sharder. In PP, token ids stay global [chunks, sequence];
        labels, positions, masks and indices are CP-local [chunks, local_tokens].
    """
    if batch.get("qkv_format") != "thd":
        raise NotImplementedError("Muse Magi integration requires packed THD inputs")
    if any(batch.get(name) is not None for name in ("pixel_values", "pixel_values_videos", "audio_values")):
        raise NotImplementedError("Muse Magi integration currently supports text-only packed inputs")
    if model.config.attention_dropout:
        raise NotImplementedError("Muse Magi integration requires attention_dropout=0")
    pp_stage = model._is_pipeline_parallel_stage()
    if not pp_stage and num_chunks != 1:
        raise ValueError("Muse Magi without PP requires one packed microbatch")

    def shard_magi(
        cp_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None,
        local_batch: MuseMagiBatch,
        *,
        loss_mask: torch.Tensor | None = None,
        padding_token_id: int = 0,
    ) -> tuple[Callable[[], contextlib.AbstractContextManager[None]], MuseMagiBatch, ShardLayout | None]:
        """Dispatch each pack's token-aligned tensors using its own attention plan.

        Args:
            cp_mesh: Context-parallel mesh for this pipeline stage.
            tp_mesh: Unused tensor-parallel mesh.
            local_batch: Global tokens [chunks, sequence] and document lengths
                [chunks, documents], with one packed row per microbatch.
            loss_mask: Unused external mask; batch loss_mask is dispatched with labels.
            padding_token_id: Value for additional token padding.

        Returns:
            Context factory, sharded batch, and a token layout for non-PP only.
            PP tensors have a leading microbatch axis; plans live until backward.
        """
        del tp_mesh, loss_mask
        from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te

        original_ids = local_batch["input_ids"]
        if (
            num_chunks < 1
            or not isinstance(original_ids, torch.Tensor)
            or original_ids.ndim != 2
            or original_ids.shape[0] != num_chunks
        ):
            raise ValueError("Muse Magi packed inputs require one packed row per microbatch")
        available, api = safe_import("magi_attention.api")
        if not available:
            raise ImportError("Muse Magi CP requires magi_attention")
        global_batch: MuseMagiBatch = make_cp_batch_for_te(
            None, dict(local_batch), qkv_format="thd", padding_token_id=padding_token_id, num_chunks=num_chunks
        )
        # The shared chunked THD converter does not carry optional loss masks.
        # Preserve this model-owned token field before dispatching each pack.
        if "loss_mask" in local_batch and "loss_mask" not in global_batch:
            global_batch["loss_mask"] = local_batch["loss_mask"]
        mesh = cp_mesh if cp_mesh is not None else model.cp_mesh
        if mesh is None:
            raise ValueError("Muse Magi context parallelism requires a context-parallel mesh")
        group = mesh.get_group()
        sequence = original_ids.shape[1]
        all_cu_tensor = global_batch.get("cu_seqlens_padded", global_batch["cu_seqlens"])
        if not isinstance(all_cu_tensor, torch.Tensor):
            raise TypeError("Muse Magi packed boundaries must be tensors")
        all_cu = all_cu_tensor.reshape(num_chunks, -1)
        plans: list[MuseMagiPlan] = []
        indices: list[torch.Tensor] = []
        sharded = dict(global_batch)
        token_fields = {
            "input_ids": padding_token_id,
            "position_ids": 0,
            "labels": -100,
            "loss_mask": 0,
            "padding_mask": True,
        }
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in token_fields if name in global_batch}
        for chunk_index, boundaries in enumerate(all_cu):
            cu = boundaries[boundaries >= 0].to(device="cpu", dtype=torch.int32)
            if int(cu[-1]) != sequence:
                raise ValueError("Muse Magi packed boundaries must span each complete padded microbatch")
            plan = _build_magi_plan(cu, config=model.config, group=group)
            full_key = plan.keys[0]
            local_indices = api.get_position_ids(full_key).to(device=original_ids.device, dtype=torch.long)
            if pp_stage:
                expected_tokens = (sequence + (-sequence) % (2 * group.size())) // group.size()
                if local_indices.numel() != expected_tokens:
                    raise ValueError("Muse Magi PP requires a pack length with matching static CP-local stage shapes")
            plans.append(plan)
            indices.append(local_indices)
            for name in chunks:
                tensor = global_batch[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Muse Magi token field {name!r} must be a tensor")
                tensor = tensor.reshape(num_chunks, -1)[chunk_index]
                chunks[name].append(api.dispatch(tensor, full_key, pad_value=token_fields[name]))
        for name, values in chunks.items():
            sharded[name] = torch.stack(values) if pp_stage or num_chunks > 1 else values[0]
            if not pp_stage and num_chunks == 1 and name in ("labels", "loss_mask"):
                sharded[name] = sharded[name].unsqueeze(0)
        sharded.pop("attention_mask", None)
        sharded["qkv_format"] = "thd"
        if pp_stage:
            # PP input shape remains global; Muse's first stage selects local ids.
            sharded["input_ids"] = original_ids.contiguous()
            sharded["_thd_local_indices"] = torch.stack(indices)
            sharded["_muse_spark_magi_plan"] = _MuseMagiBatchPlans(tuple(plans))
            # PP splits this explicit identity alongside token rows. FSDP may move
            # it to CUDA; resolve it once per stage, outside checkpointed layers.
            sharded["_muse_spark_magi_microbatch"] = torch.arange(num_chunks, dtype=torch.long)
            if num_chunks == 1:
                for name in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen"):
                    if name in sharded:
                        sharded[name] = sharded[name].unsqueeze(0)
            return contextlib.nullcontext, sharded, None
        sharded["_thd_local_indices"] = indices[0]
        sharded["_muse_spark_magi_plan"] = plans[0]
        if group.rank() == 0:
            logger.info(
                "Muse Magi CP: cp=%d local_tokens=%d masks=%s", group.size(), len(indices[0]), sorted(plans[0].keys)
            )
        layout = ShardLayout(
            local_token_global_indices=indices[0],
            padded_seq_len=plans[0].keys[0].total_seqlen_q,
            original_seq_len=original_ids.numel(),
            input_row_shape=tuple(original_ids.shape),
        )
        return contextlib.nullcontext, sharded, layout

    return {"cp_sharder": ContextParallelSharder(shard_batch=shard_magi)}
