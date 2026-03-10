# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Context-Parallel-aware wrapper for Qwen3.5 MoE GatedDeltaNet linear attention.

When a CP mesh is attached (via ``apply_cp``), the forward pass:
  1. Communicates conv1d boundary tokens between adjacent CP ranks so the
     causal convolution produces correct results at rank boundaries.
  2. Calls FLA's ``chunk_gated_delta_rule`` with ``cp_context`` to handle
     cross-rank recurrent-state propagation for the delta rule.

When no CP mesh is set, the module delegates to the original HF forward.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
)

logger = logging.getLogger(__name__)


class CPAwareGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """Drop-in replacement for ``Qwen3_5MoeGatedDeltaNet`` with FLA Context Parallelism.

    All ``__init__`` parameters and weights are inherited unchanged from the HF
    class.  The only addition is ``_cp_mesh`` which is set externally by
    ``apply_cp`` in the parallelizer.
    """

    _cp_mesh: DeviceMesh | None

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._cp_mesh = None
        self._cp_debug_enter_logged = False
        self._cp_debug_layout_logged = False
        self._cp_debug_ref_logged = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        qkv_format: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        seq_index: torch.Tensor | None = None,
    ):
        # Fast path: no CP → original HF forward
        if self._cp_mesh is None or self._cp_mesh.size() <= 1:
            return super().forward(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        return self._forward_with_cp(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            seq_index=seq_index,
        )

    # ------------------------------------------------------------------
    # Conv1d boundary communication
    # ------------------------------------------------------------------
    def _conv1d_with_cp(
        self,
        mixed_qkv: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Run causal conv1d with cross-rank boundary token exchange.

        Args:
            mixed_qkv: [B, D, S_local] tensor (channels-first for conv).
            cp_group: CP process group.

        Returns:
            [B, D, S_local] conv output with correct boundary handling.
        """
        W = self.conv_kernel_size  # typically 4
        cp_rank = dist.get_rank(cp_group)
        cp_world = dist.get_world_size(cp_group)
        B, D, S_local = mixed_qkv.shape

        # Map group-local ranks → global ranks for P2POp.
        group_ranks = dist.get_process_group_ranks(cp_group)

        # Exchange last W-1 tokens with next rank so conv has correct context.
        recv_buf = torch.zeros(B, D, W - 1, device=mixed_qkv.device, dtype=mixed_qkv.dtype)

        ops = []
        if cp_rank > 0:
            ops.append(dist.P2POp(dist.irecv, recv_buf, group_ranks[cp_rank - 1], cp_group))
        if cp_rank < cp_world - 1:
            send_buf = mixed_qkv[:, :, -(W - 1):].contiguous()
            ops.append(dist.P2POp(dist.isend, send_buf, group_ranks[cp_rank + 1], cp_group))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Prepend received tokens (zeros for rank 0) to form extended input.
        extended = torch.cat([recv_buf, mixed_qkv], dim=2)  # [B, D, W-1+S_local]

        # Run causal conv on extended sequence, then take last S_local positions.
        if self.causal_conv1d_fn is not None:
            extended = self.causal_conv1d_fn(
                x=extended,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            extended = F.silu(self.conv1d(extended)[:, :, : W - 1 + S_local])

        # The first W-1 positions correspond to the received prefix;
        # positions W-1 onward are the correct outputs for local tokens.
        return extended[:, :, W - 1:]

    def _cp_debug_enabled(self) -> bool:
        value = os.getenv("NEMO_QWEN35_CP_DEBUG", "").strip().lower()
        return value not in ("", "0", "false", "off")

    def _emit_debug(self, message: str, *args) -> None:
        formatted = message % args if args else message
        logger.warning(formatted)
        print(formatted, flush=True)

    def _cp_debug_ref_enabled(self) -> bool:
        value = os.getenv("NEMO_QWEN35_CP_DEBUG_REF_LAYERS", "").strip().lower()
        if not value:
            return False
        if value == "all":
            return True
        try:
            layers = {int(item.strip()) for item in value.split(",") if item.strip()}
        except ValueError:
            logger.warning(
                "Ignoring invalid NEMO_QWEN35_CP_DEBUG_REF_LAYERS=%r; expected 'all' or comma-separated ints.",
                value,
            )
            return False
        return self.layer_idx in layers

    def _extract_debug_positions(
        self,
        position_ids: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if position_ids is None:
            return None

        if position_ids.ndim == 1:
            local_positions = position_ids
        elif position_ids.ndim == 2:
            local_positions = position_ids[0]
        elif position_ids.ndim == 3:
            local_positions = position_ids[0, 0]
        else:
            return None

        if local_positions.shape[-1] != seq_len:
            return None

        return local_positions.to(dtype=torch.long)

    def _extract_sequence_positions(
        self,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if seq_index is not None:
            if seq_index.ndim == 1:
                local_positions = seq_index
            elif seq_index.ndim == 2:
                local_positions = seq_index[0]
            else:
                local_positions = None

            if local_positions is not None and local_positions.shape[-1] == seq_len:
                return local_positions.to(dtype=torch.long)

        return self._extract_debug_positions(position_ids, seq_len)

    def _all_gather_concat(
        self,
        tensor: torch.Tensor,
        cp_group: dist.ProcessGroup,
        *,
        dim: int,
    ) -> torch.Tensor:
        cp_world = dist.get_world_size(cp_group)
        gathered = [torch.empty_like(tensor) for _ in range(cp_world)]
        dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=dim)

    def _undo_attention_load_balancing(
        self,
        hidden_states: torch.Tensor,
        original_positions: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_rank = dist.get_rank(cp_group)
        seq_len = hidden_states.shape[1]

        cp_order_hidden = self._all_gather_concat(hidden_states, cp_group, dim=1)
        cp_order_positions = self._all_gather_concat(original_positions, cp_group, dim=0)

        sort_order = torch.argsort(cp_order_positions)
        sorted_positions = cp_order_positions.index_select(0, sort_order)
        full_hidden = cp_order_hidden.index_select(1, sort_order)

        start = cp_rank * seq_len
        end = start + seq_len
        return full_hidden[:, start:end], sorted_positions

    def _redo_attention_load_balancing(
        self,
        output: torch.Tensor,
        original_positions: torch.Tensor,
        sorted_positions: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        full_output = self._all_gather_concat(output, cp_group, dim=1)
        restore_indices = torch.searchsorted(sorted_positions, original_positions)
        restored_positions = sorted_positions.index_select(0, restore_indices)
        if not torch.equal(restored_positions, original_positions):
            raise RuntimeError(
                f"Failed to restore Qwen3.5 CP linear-attn output on layer {self.layer_idx}: "
                "sorted positions do not cover the local CP layout."
            )
        return full_output.index_select(1, restore_indices)

    def _describe_position_ids(
        self,
        position_ids: torch.Tensor | None,
        seq_len: int,
    ) -> str:
        if position_ids is None:
            return "position_ids=None"

        shape = tuple(position_ids.shape)
        dtype = getattr(position_ids, "dtype", None)
        device = getattr(position_ids, "device", None)
        ndim = getattr(position_ids, "ndim", None)

        if ndim not in (1, 2, 3):
            return (
                f"position_ids type={type(position_ids).__name__} shape={shape} dtype={dtype} "
                f"device={device} unsupported_ndim={ndim} expected_seq_len={seq_len}"
            )

        last_dim = shape[-1] if len(shape) > 0 else None
        if last_dim != seq_len:
            return (
                f"position_ids type={type(position_ids).__name__} shape={shape} dtype={dtype} "
                f"device={device} last_dim={last_dim} expected_seq_len={seq_len}"
            )

        return (
            f"position_ids type={type(position_ids).__name__} shape={shape} dtype={dtype} "
            f"device={device} last_dim_matches_seq_len={seq_len}"
        )

    def _maybe_log_cp_layout(
        self,
        cp_group: dist.ProcessGroup,
        local_positions: torch.Tensor | None,
        *,
        qkv_format: str | None,
        cu_seqlens: torch.Tensor | None,
        seq_index: torch.Tensor | None,
    ) -> None:
        if self._cp_debug_layout_logged or not self._cp_debug_enabled() or local_positions is None:
            return

        cp_rank = dist.get_rank(cp_group)
        cp_world = dist.get_world_size(cp_group)
        pos_cpu = local_positions.detach().to(device="cpu")
        numel = pos_cpu.numel()
        contiguous = bool(numel <= 1 or torch.all(pos_cpu[1:] - pos_cpu[:-1] == 1))
        payload = {
            "rank": cp_rank,
            "numel": int(numel),
            "head": pos_cpu[: min(8, numel)].tolist(),
            "tail": pos_cpu[-min(8, numel) :].tolist(),
            "min": int(pos_cpu.min().item()) if numel > 0 else None,
            "max": int(pos_cpu.max().item()) if numel > 0 else None,
            "contiguous": contiguous,
        }
        gathered = [None] * cp_world
        dist.all_gather_object(gathered, payload, group=cp_group)

        if cp_rank == 0:
            self._emit_debug(
                "Qwen3.5 CP linear-attn layer=%s qkv_format=%s cu_seqlens=%s seq_index_shape=%s local_position_layout=%s",
                self.layer_idx,
                qkv_format or "bshd",
                None if cu_seqlens is None else tuple(cu_seqlens.shape),
                None if seq_index is None else tuple(seq_index.shape),
                gathered,
            )
            if any(not item["contiguous"] for item in gathered):
                self._emit_debug(
                    "Layer %s sees non-contiguous local positions before linear attention; "
                    "CP likely needs undo/redo load-balancing before conv/recurrent state propagation.",
                    self.layer_idx,
                )
            elif any(
                gathered[idx]["max"] is not None
                and gathered[idx + 1]["min"] is not None
                and gathered[idx]["max"] + 1 != gathered[idx + 1]["min"]
                for idx in range(cp_world - 1)
            ):
                self._emit_debug(
                    "Layer %s rank-order local positions are not globally contiguous; "
                    "build_cp_context([0, S]) is likely assuming the wrong CP layout.",
                    self.layer_idx,
                )

        self._cp_debug_layout_logged = True

    def _maybe_debug_full_reference(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        cp_group: dist.ProcessGroup,
    ) -> None:
        if self._cp_debug_ref_logged or not self._cp_debug_ref_enabled():
            return

        local_positions = self._extract_sequence_positions(
            position_ids,
            seq_index,
            hidden_states.shape[1],
        )
        if local_positions is None:
            return

        cp_rank = dist.get_rank(cp_group)
        cp_world = dist.get_world_size(cp_group)

        with torch.no_grad():
            gathered_hidden = [torch.empty_like(hidden_states) for _ in range(cp_world)]
            dist.all_gather(gathered_hidden, hidden_states.contiguous(), group=cp_group)

            gathered_out = [torch.empty_like(output) for _ in range(cp_world)]
            dist.all_gather(gathered_out, output.contiguous(), group=cp_group)

            gathered_pos = [torch.empty_like(local_positions) for _ in range(cp_world)]
            dist.all_gather(gathered_pos, local_positions.contiguous(), group=cp_group)

            cp_order_hidden = torch.cat(gathered_hidden, dim=1)
            cp_order_out = torch.cat(gathered_out, dim=1)
            cp_order_pos = torch.cat(gathered_pos, dim=0)
            order = torch.argsort(cp_order_pos)
            sorted_pos = cp_order_pos.index_select(0, order)
            expected = torch.arange(
                sorted_pos.numel(),
                device=sorted_pos.device,
                dtype=sorted_pos.dtype,
            )

            if not torch.equal(sorted_pos, expected):
                if cp_rank == 0:
                    self._emit_debug(
                        "Skipping gathered linear-attn reference on layer %s because gathered positions are not a simple 0..S-1 range: head=%s tail=%s",
                        self.layer_idx,
                        sorted_pos[: min(8, sorted_pos.numel())].tolist(),
                        sorted_pos[-min(8, sorted_pos.numel()) :].tolist(),
                    )
                self._cp_debug_ref_logged = True
                return

            full_hidden = cp_order_hidden.index_select(1, order)
            full_cp_out = cp_order_out.index_select(1, order)

            full_attention_mask = None
            if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] == hidden_states.shape[1]:
                gathered_mask = [torch.empty_like(attention_mask) for _ in range(cp_world)]
                dist.all_gather(gathered_mask, attention_mask.contiguous(), group=cp_group)
                cp_order_mask = torch.cat(gathered_mask, dim=1)
                full_attention_mask = cp_order_mask.index_select(1, order)

            ref_out = Qwen3_5MoeGatedDeltaNet.forward(
                self,
                full_hidden,
                cache_params=None,
                cache_position=None,
                attention_mask=full_attention_mask,
            )
            diff = (full_cp_out - ref_out).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())

            if cp_rank == 0:
                self._emit_debug(
                    "Qwen3.5 CP linear-attn gathered reference layer=%s max_abs=%.6e mean_abs=%.6e full_seq_len=%s",
                    self.layer_idx,
                    max_abs,
                    mean_abs,
                    full_hidden.shape[1],
                )

        self._cp_debug_ref_logged = True

    # ------------------------------------------------------------------
    # CP-aware forward
    # ------------------------------------------------------------------
    def _forward_with_cp(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        position_ids: torch.Tensor | None,
        qkv_format: str | None,
        cu_seqlens: torch.Tensor | None,
        seq_index: torch.Tensor | None,
    ) -> torch.Tensor:
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

        batch_size, seq_len, _ = hidden_states.shape

        cp_group = self._cp_mesh.get_group()
        cp_size = self._cp_mesh.size()
        original_hidden_states = hidden_states

        # ---- Build FLA CP context (once, reused for every sequence) ----
        global_seq_len = seq_len * cp_size
        cu_seqlens_single = torch.tensor(
            [0, global_seq_len], dtype=torch.long, device=hidden_states.device,
        )
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens_single,
            group=cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )
        if self._cp_debug_enabled() and not self._cp_debug_enter_logged:
            cp_rank = dist.get_rank(cp_group)
            self._emit_debug(
                "Entered Qwen3.5 CP linear-attn debug layer=%s cp_rank=%s batch=%s seq_local=%s qkv_format=%s cu_seqlens_shape=%s seq_index_shape=%s",
                self.layer_idx,
                cp_rank,
                batch_size,
                seq_len,
                qkv_format or "bshd",
                None if cu_seqlens is None else tuple(cu_seqlens.shape),
                None if seq_index is None else tuple(seq_index.shape),
            )
            self._emit_debug(
                "Qwen3.5 CP linear-attn layer=%s cp_rank=%s %s",
                self.layer_idx,
                cp_rank,
                self._describe_position_ids(position_ids, seq_len),
            )
            self._cp_debug_enter_logged = True
        local_positions = self._extract_sequence_positions(position_ids, seq_index, seq_len)
        if self._cp_debug_enabled() and local_positions is None and not self._cp_debug_layout_logged:
            cp_rank = dist.get_rank(cp_group)
            self._emit_debug(
                "Qwen3.5 CP linear-attn layer=%s cp_rank=%s could not derive local positions from position_ids",
                self.layer_idx,
                cp_rank,
            )
        self._maybe_log_cp_layout(
            cp_group,
            local_positions,
            qkv_format=qkv_format,
            cu_seqlens=cu_seqlens,
            seq_index=seq_index,
        )
        sorted_positions = None
        if local_positions is not None:
            # Attention runs on a load-balanced CP layout, but conv + recurrent state
            # propagation require rank-order sequential tokens.
            hidden_states, sorted_positions = self._undo_attention_load_balancing(
                hidden_states,
                local_positions,
                cp_group,
            )

        # ---- Projections (batched, pointwise) ----
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S_local, conv_dim]
        z = self.in_proj_z(hidden_states)             # [B, S_local, value_dim]
        b = self.in_proj_b(hidden_states)             # [B, S_local, num_v_heads]
        a = self.in_proj_a(hidden_states)             # [B, S_local, num_v_heads]

        # ---- Causal Conv1d with cross-rank boundary exchange ----
        mixed_qkv = mixed_qkv.transpose(1, 2)                  # [B, D, S_local]
        mixed_qkv = self._conv1d_with_cp(mixed_qkv, cp_group)  # [B, D, S_local]
        mixed_qkv = mixed_qkv.transpose(1, 2)                  # [B, S_local, D]

        # ---- Split QKV ----
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # ---- Gate & beta ----
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GVA: repeat q/k heads to match v heads
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ---- Chunk GDN with CP (per-sequence) ----
        # cp_context is built for a single sequence; reuse for each batch element.
        attn_outs = []
        for bi in range(batch_size):
            out_bi, _ = fla_chunk_gated_delta_rule(
                query[bi:bi+1],
                key[bi:bi+1],
                value[bi:bi+1],
                g=g[bi:bi+1],
                beta=beta[bi:bi+1],
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cp_context=cp_context,
            )
            attn_outs.append(out_bi)
        core_attn_out = torch.cat(attn_outs, dim=0)  # [B, S_local, H_v, D_v]

        # ---- Gated RMSNorm + output projection ----
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        if local_positions is not None and sorted_positions is not None:
            output = self._redo_attention_load_balancing(
                output,
                local_positions,
                sorted_positions,
                cp_group,
            )
        self._maybe_debug_full_reference(
            original_hidden_states,
            output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_index=seq_index,
            cp_group=cp_group,
        )
        return output
