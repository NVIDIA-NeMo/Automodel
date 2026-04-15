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
  1. Recovers dense sequence order from PyTorch's load-balanced CP layout using
     ``seq_index`` or ``position_ids``.
  2. Runs the causal conv1d and FLA gated delta rule on that dense ordering.
  3. Restores the output back to the original load-balanced CP layout.

When no CP mesh is set, the module delegates to the original HF forward.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet


class _AllGatherConcatFn(Function):
    """All-gather + concat with autograd-safe backward.

    The forward concatenates equal-sized local shards from all ranks along `dim`.
    Backward all-reduces the concatenated gradient across ranks, then slices out
    the local shard for the current rank.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: dist.ProcessGroup, dim: int):
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor.contiguous(), group=group)

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.local_dim_size = local_tensor.size(dim)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_dim_size
        grad_local = grad_full.narrow(ctx.dim, start, ctx.local_dim_size).contiguous()
        return grad_local, None, None


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

    def _compute_gate(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the gating value ``g`` using fp32 params.

        When ``_fp32_params`` exists (FSDP mixed-dtype), delegates to
        the holder's forward so FSDP unshard/reshard lifecycle is natural.
        Otherwise falls back to the inline computation.
        """
        if hasattr(self, "_fp32_params"):
            return self._fp32_params(a, self.dt_bias)
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    def _forward_no_cp(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: torch.Tensor | None = None,
    ):
        """HF GatedDeltaNet forward with FSDP-safe fp32 gate computation.

        Copied from transformers==5.3.0 Qwen3_5GatedDeltaNet.forward
        with gate computation replaced by self._compute_gate(a).
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = self._compute_gate(a)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)

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
        # Fast path: no CP → run HF forward with fp32-safe gate computation
        if self._cp_mesh is None or self._cp_mesh.size() <= 1:
            return self._forward_no_cp(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )

        # HF decoder layers don't pass position_ids to linear_attn.
        # Use cached value from the decoder layer pre-hook, then clear it.
        if position_ids is None:
            position_ids = getattr(self, "_cached_position_ids", None)
            self._cached_position_ids = None
        return self._forward_with_cp(
            hidden_states,
            position_ids=position_ids,
            seq_index=seq_index,
        )

    # ------------------------------------------------------------------
    # Conv1d boundary communication
    # ------------------------------------------------------------------
    def _conv1d_with_cp(
        self,
        mixed_qkv: torch.Tensor,
        cp_context,
    ) -> torch.Tensor:
        """Run causal conv1d via FLA's CP-aware conv implementation.

        Args:
            mixed_qkv: [B, D, S_local] tensor (channels-first for conv).
            cp_context: FLA CP context built by ``build_cp_context``.

        Returns:
            [B, D, S_local] conv output with correct boundary handling.
        """
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

        conv_in = mixed_qkv.transpose(1, 2).contiguous()  # [B, S_local, D]
        conv_outs = []
        for bi in range(conv_in.shape[0]):
            out_bi, _ = fla_causal_conv1d(
                x=conv_in[bi : bi + 1],
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                cp_context=cp_context,
            )
            conv_outs.append(out_bi)

        return torch.cat(conv_outs, dim=0).transpose(1, 2).contiguous()

    def _extract_local_positions(
        self,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        for positions in (seq_index, position_ids):
            if positions is None:
                continue

            if positions.ndim == 1:
                local_positions = positions
            elif positions.ndim == 2:
                local_positions = positions[0]
            elif positions.ndim == 3:
                local_positions = positions[0, 0]
            else:
                continue

            if local_positions.shape[-1] == seq_len:
                return local_positions.to(dtype=torch.long)

        return None

    def _all_gather_concat(
        self,
        tensor: torch.Tensor,
        cp_group: dist.ProcessGroup,
        *,
        dim: int,
        differentiable: bool = False,
    ) -> torch.Tensor:
        if differentiable:
            return _AllGatherConcatFn.apply(tensor, cp_group, dim)

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

        cp_order_hidden = self._all_gather_concat(hidden_states, cp_group, dim=1, differentiable=True)
        cp_order_positions = self._all_gather_concat(original_positions, cp_group, dim=0)

        sort_order = torch.argsort(cp_order_positions)
        sorted_positions = cp_order_positions.index_select(0, sort_order)
        expected_positions = torch.arange(
            sorted_positions.numel(),
            device=sorted_positions.device,
            dtype=sorted_positions.dtype,
        )
        if not torch.equal(sorted_positions, expected_positions):
            raise RuntimeError(
                f"Qwen3.5 CP linear-attn layer {self.layer_idx} requires dense global token positions "
                "covering 0..S-1 after gathering CP shards."
            )
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
        full_output = self._all_gather_concat(output, cp_group, dim=1, differentiable=True)
        restore_indices = torch.searchsorted(sorted_positions, original_positions)
        restored_positions = sorted_positions.index_select(0, restore_indices)
        if not torch.equal(restored_positions, original_positions):
            raise RuntimeError(
                f"Failed to restore Qwen3.5 CP linear-attn output on layer {self.layer_idx}: "
                "sorted positions do not cover the local CP layout."
            )
        return full_output.index_select(1, restore_indices)

    # ------------------------------------------------------------------
    # CP-aware forward
    # ------------------------------------------------------------------
    def _forward_with_cp(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
    ) -> torch.Tensor:
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

        batch_size, seq_len, _ = hidden_states.shape

        cp_group = self._cp_mesh.get_group()
        cp_size = self._cp_mesh.size()

        local_positions = self._extract_local_positions(position_ids, seq_index, seq_len)
        if local_positions is None:
            raise RuntimeError(
                f"Qwen3.5 CP linear-attn layer {self.layer_idx} requires seq_index or position_ids "
                "with local sequence length metadata to undo load-balanced CP sharding."
            )

        # ---- Build FLA CP context (once, reused for every sequence) ----
        # After undoing the load-balanced attention layout, each rank again owns a
        # contiguous chunk of a dense global sequence of length seq_len * cp_size.
        global_seq_len = seq_len * cp_size
        cu_seqlens_single = torch.tensor(
            [0, global_seq_len],
            dtype=torch.long,
            device=hidden_states.device,
        )
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens_single,
            group=cp_group,
            conv1d_kernel_size=self.conv_kernel_size,
        )
        # Attention runs on a load-balanced CP layout, but conv + recurrent state
        # propagation require rank-order sequential tokens.
        hidden_states, sorted_positions = self._undo_attention_load_balancing(
            hidden_states,
            local_positions,
            cp_group,
        )

        # ---- Projections (batched, pointwise) ----
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S_local, conv_dim]
        z = self.in_proj_z(hidden_states)  # [B, S_local, value_dim]
        b = self.in_proj_b(hidden_states)  # [B, S_local, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [B, S_local, num_v_heads]

        # ---- Causal Conv1d with cross-rank boundary exchange ----
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, D, S_local]
        mixed_qkv = self._conv1d_with_cp(mixed_qkv, cp_context)  # [B, D, S_local]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S_local, D]

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
        g = self._compute_gate(a)

        # GVA: repeat q/k heads to match v heads
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ---- Chunk GDN with CP (per-sequence) ----
        # cp_context is built for a single sequence; reuse for each batch element.
        attn_outs = []
        for bi in range(batch_size):
            out_bi, _ = fla_chunk_gated_delta_rule(
                query[bi : bi + 1],
                key[bi : bi + 1],
                value[bi : bi + 1],
                g=g[bi : bi + 1],
                beta=beta[bi : bi + 1],
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
        output = self._redo_attention_load_balancing(
            output,
            local_positions,
            sorted_positions,
            cp_group=cp_group,
        )
        return output


class _Fp32ParamHolder(torch.nn.Module):
    """Holder for float32 params (A_log) that need a separate FSDP group.

    The ``forward`` computes the gating value ``g`` that HF's
    ``Qwen3_5GatedDeltaNet.forward`` would normally compute inline.
    By doing the computation *inside* this module's forward, FSDP's
    unshard/reshard lifecycle works naturally — the params are
    unsharded during the computation and resharded after.
    """

    def forward(self, a: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * F.softplus(a.float() + dt_bias)


def _make_fp32_getattr(orig_getattr):
    """Create a ``__getattr__`` that resolves fp32 params from ``_fp32_params``.

    Allows ``self.A_log`` to resolve from the holder submodule so that
    code outside forward (e.g. state_dict, checkpointing) can still
    access the parameter by name.
    """

    def _getattr_with_fp32(self, name):
        modules = self.__dict__.get("_modules", {})
        fp32_holder = modules.get("_fp32_params")
        if fp32_holder is not None and name in fp32_holder._parameters:
            return fp32_holder._parameters[name]
        return orig_getattr(self, name)

    return _getattr_with_fp32


def patch_hf_model(model, cp_enabled=False):
    """Patch HF Qwen3.5 GatedDeltaNet modules for FSDP and optional CP support.

    For FSDP compatibility, move float32 bare params (A_log) into a
    ``_fp32_params`` submodule so ``fully_shard_by_dtype`` can wrap them
    in a separate FSDP group.

    Every module's ``__class__`` is swapped to ``CPAwareGatedDeltaNet``
    whose ``forward()`` calls ``self._fp32_params()`` to trigger FSDP
    unshard before accessing the fp32 params.  When ``cp_enabled=True``,
    the CP mesh is also configured.
    """
    import logging

    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    except ImportError:
        return

    _logger = logging.getLogger(__name__)
    _PATCHED_ATTR = "_fp32_getattr_patched"
    patched = 0
    patched_classes = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, Qwen3_5GatedDeltaNet):
            continue

        mod.__class__ = CPAwareGatedDeltaNet
        mod._cp_mesh = None

        # Move float32 bare params into a holder submodule for FSDP.
        # The CPAwareGatedDeltaNet forward calls self._fp32_params()
        # to trigger FSDP unshard; __getattr__ redirects self.A_log
        # to the holder so it returns the unsharded plain tensor.
        holder = None
        for pname in list(mod._parameters.keys()):
            param = mod._parameters[pname]
            if param is not None and param.dtype == torch.float32:
                if holder is None:
                    holder = _Fp32ParamHolder()
                setattr(holder, pname, param)
                del mod._parameters[pname]
        if holder is not None:
            mod.add_module("_fp32_params", holder)

            # Guard against re-wrapping __getattr__ on repeated calls.
            cls = type(mod)
            if cls not in patched_classes and not getattr(cls, _PATCHED_ATTR, False):
                cls.__getattr__ = _make_fp32_getattr(cls.__getattr__)
                setattr(cls, _PATCHED_ATTR, True)
                patched_classes.add(cls)
        patched += 1

    if patched > 0:
        _logger.info(
            "Patched %d GatedDeltaNet modules (cp=%s) with FSDP-safe fp32 param wrapping.",
            patched,
            cp_enabled,
        )
