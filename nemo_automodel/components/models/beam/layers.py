# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Beam gated attention, root10 SSMax, rotary precision, and SwiGLU/router overrides."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, Gate, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

if TYPE_CHECKING:
    from transformers.models.beam import BeamConfig

# Scalable-Softmax constants hard-coded by the HF model (``compute_attention_scaling``).
SSMAX_REFERENCE_LENGTH = 2048
SSMAX_EXPONENT = 0.1


def ssmax_query_scale(position_ids: torch.Tensor) -> torch.Tensor:
    """Compute HF Beam's NoPE query scale in FP32.

    Args:
        position_ids: Integer tensor of shape [batch, sequence] or packed [tokens],
            containing absolute per-document positions, local to this rank under CP.

    Returns:
        FP32 tensor with the same shape/device as position_ids:
        (1 + (position_ids + 1) / 2048) ** 0.1.
    """
    return torch.exp(SSMAX_EXPONENT * torch.log1p((position_ids.float() + 1) / SSMAX_REFERENCE_LENGTH))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply HF's half-vector rotation.

    Args:
        x: Tensor of shape [..., head_dim], with arbitrary leading axes and even head_dim.

    Returns:
        New tensor of the same shape/dtype/device, concatenating -x's second half
        and x's first half along the last axis.
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_rope_tables(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand original FP32 rotary tables into HF's duplicated-half layout.

    Args:
        freqs_cis: FP32 tensor of shape [..., head_dim], with arbitrary leading axes
            corresponding to [batch, sequence] or packed [tokens]. The last axis
            concatenates half-width cosine and sine values, in that order.

    Returns:
        Pair (cos, sin), each a new FP32 tensor of shape [..., 1, head_dim] on the
        input device. Each duplicates its half-width values; the singleton head
        axis broadcasts over query/key heads. No arithmetic rounds the values.
    """
    half = freqs_cis.shape[-1] // 2
    cos, sin = freqs_cis[..., :half], freqs_cis[..., half:]
    return torch.cat((cos, cos), dim=-1).unsqueeze(-2), torch.cat((sin, sin), dim=-1).unsqueeze(-2)


def apply_rope_hf(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply HF Beam's rotary expression and preserve its backward rounding.

    Do not eagerly cast x to FP32: HF rounds each product's gradient to x.dtype.

    Args:
        x: Query/key tensor of shape [batch, sequence, heads, head_dim] or packed
            [tokens, heads, head_dim], in the activation dtype.
        cos: Original FP32 cosine tensor of shape [batch, sequence, 1, head_dim]
            or [tokens, 1, head_dim], broadcastable to x and on the same device.
        sin: Original FP32 sine tensor with the same shape/device as cos.

    Returns:
        New tensor with x's shape/dtype/device, computed as
        (x * cos + rotate_half(x) * sin).to(x.dtype). Inputs are not mutated.
    """
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


def layer_rope_parameters(config: BeamConfig, layer_type: str) -> dict[str, Any] | None:
    """Return the layer type's RoPE parameters, or None for NoPE+SSMax.

    Reject flat pre-Beam configs and non-default rope_type rather than silently changing positional encoding."""
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict) or layer_type not in params:
        keys = sorted(params) if isinstance(params, dict) else type(params).__name__
        raise ValueError(
            f"Beam config needs rope_parameters keyed by layer type; got {keys} without an entry for {layer_type!r}. "
            "Pre-Beam (Reflection) checkpoints must be converted first (convert_reflection_to_beam.py)."
        )
    entry = params[layer_type]
    if entry is not None and entry.get("rope_type", "default") != "default":
        raise NotImplementedError(f"Beam layer type {layer_type!r}: rope_type {entry['rope_type']!r} is not supported")
    return entry


def depth_scaled_residual_coefs(layer_idx: int, is_attention: bool) -> tuple[float, float]:
    """(sublayer_scale, residual_scale) for the depth-scaled residual add.

    1-based residual-add index: ``2L`` for attention, ``2L + 1`` for the MLP; the sublayer output is
    scaled by ``1/sqrt(idx)`` and the residual stream by ``sqrt((idx - 1)/idx)``.
    """
    residual_add_idx = 2 * (layer_idx + 1) + (0 if is_attention else 1)
    return 1.0 / math.sqrt(residual_add_idx), math.sqrt((residual_add_idx - 1) / residual_add_idx)


def window_size_for_backend(sliding_window: int | None, attn_impl: str) -> tuple[int, int] | None:
    """Translate the visible-token window W: SDPA uses W, TE/FA use inclusive offsets (W-1, 0).

    Flex takes sliding_window=W directly."""
    if sliding_window is None:
        return None
    if attn_impl == "sdpa":
        return (int(sliding_window), 0)
    return (int(sliding_window) - 1, 0)


def reject_flex_attention_mask(attention_mask: torch.Tensor | None) -> None:
    """Reject padding on Beam's unmasked-only flex backend without a device synchronization.

    Args:
        attention_mask: Optional tensor, normally [batch, sequence]; any non-None value
            is rejected without inspecting its contents. Use TE or SDPA for padded input."""
    if attention_mask is not None:
        raise NotImplementedError(
            "Beam on the flex attention backend does not accept an attention_mask (padded batches): the shared "
            "FlexAttention ignores padding masks, so valid tokens would attend to padding tokens. Use "
            "backend.attn='te' or 'sdpa' for padded batches (both honour the mask), or pack the sequences."
        )


def packed_sdpa_mask(cu_seqlens: torch.Tensor, total_tokens: int, sliding_window: int | None) -> torch.Tensor:
    """Construct a block-diagonal causal mask for packed SDPA.

    Args:
        cu_seqlens: Integer tensor of shape [documents + 1], beginning at zero and
            ending at total_tokens, with strictly increasing document boundaries.
        total_tokens: Number of tokens in the concatenated packed buffer.
        sliding_window: Number of visible tokens, including the query, or None.

    Returns:
        Bool tensor of shape [1, 1, tokens, tokens] on cu_seqlens.device, where
        tokens is total_tokens and True permits a query/key pair.
    """
    cu = cu_seqlens.to(torch.long)
    doc = torch.zeros(total_tokens, dtype=torch.long, device=cu.device)
    doc[cu[1:-1]] = 1  # sequence starts (skip the first, which starts at 0)
    doc = doc.cumsum(0)
    pos = torch.arange(total_tokens, device=cu.device)
    keep = (doc[None, :] == doc[:, None]) & (pos[None, :] <= pos[:, None])
    if sliding_window is not None:
        keep &= pos[None, :] > pos[:, None] - int(sliding_window)
    return keep[None, None]


def _cp_local_positions(position_ids: torch.Tensor, local_len: int, cp_size: int, cp_rank: int) -> torch.Tensor:
    """Select the two load-balanced sequence chunks owned by this CP rank.

    Args:
        position_ids: Integer tensor of shape [batch, global_sequence] or
            [global_sequence], or the corresponding already-local shape.
        local_len: Tokens per local sequence, equal to global_sequence / cp_size;
            must be even when selecting the two head/tail chunks.
        cp_size: Number of context-parallel ranks.
        cp_rank: Zero-based context-parallel rank.

    Returns:
        Position tensor of shape [batch, local_len] or [local_len]. Already-local
        input (or cp_size=1) is returned by alias; otherwise the selected chunks
        are concatenated into new storage.
    """
    if position_ids.shape[-1] == local_len or cp_size == 1:
        return position_ids
    chunk = local_len // 2
    first = position_ids[..., cp_rank * chunk : (cp_rank + 1) * chunk]
    second_start = (2 * cp_size - 1 - cp_rank) * chunk
    second = position_ids[..., second_start : second_start + chunk]
    return torch.cat([first, second], dim=-1)


class BeamAttention(nn.Module):
    """Gated attention with per-head q/k RMSNorm and full-width output gating.

    Layer type selects full/sliding attention and RoPE or NoPE+SSMax."""

    def __init__(self, config: BeamConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.backend = backend
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        attention_bias = bool(getattr(config, "attention_bias", False))
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, attention_bias, dtype=dtype
        )
        # full-width output gate + per-head bias (always bias-free projection, as in HF)
        self.gate_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, False, dtype=dtype
        )
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads, dtype=dtype))
        # per-head RMSNorm over head_dim
        self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        self.window_size = window_size_for_backend(self.sliding_window, backend.attn)
        self.use_rope = layer_rope_parameters(config, self.layer_type) is not None
        self.use_ssmax = not self.use_rope

        # HF applies attention dropout in training only (``dropout=0.0 if not self.training else attention_dropout``).
        # TE takes it at construction and gates it on ``self.training``; SDPA takes ``dropout_p`` per call, set in
        # ``_attend`` after the mask preprocessing (which drops unknown kwargs); FlexAttention has no dropout.
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0) or 0.0)
        if self.attention_dropout > 0.0 and backend.attn not in ("te", "sdpa"):
            raise NotImplementedError(
                f"Beam: attention_dropout={self.attention_dropout} needs the 'te' or 'sdpa' attention backend; "
                f"the '{backend.attn}' backend has no dropout support"
            )
        backend_kwargs = {"attention_dropout": self.attention_dropout} if backend.attn == "te" else {}
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            num_gqa_groups=self.num_kv_heads,
            **backend_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None,
        rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Apply gated attention with HF-compatible RoPE or NoPE query scaling.

        Args:
            x: Tensor of shape [batch, sequence, hidden] or packed [tokens, hidden],
                in the compute dtype; sequence/tokens are local under CP.
            freqs_cis: Original FP32 concatenated half-width cosine/sine tensor of shape
                [batch, sequence, head_dim] or [tokens, head_dim] for unfused RoPE.
                Fused layout follows position_ids_to_freqs_cis; None for NoPE.
            rope_tables: Optional original FP32 (cos, sin) pair, each of shape
                [batch, sequence, 1, head_dim] or [tokens, 1, head_dim]. Shared with
                other layers without mutation. When absent, derived from freqs_cis.
            position_ids: Integer tensor of shape [batch, sequence] or [tokens].
                Normally local under CP; full-length positions can be selected using cp_size/cp_rank.
            attention_mask: Optional [batch, sequence] valid-token mask (1/True = attend),
                or backend-supported [batch, 1, sequence, sequence] bool/additive mask.
            **attn_kwargs: Backend metadata: integer cu_seqlens/cu_seqlens_padded tensors
                of shape [documents + 1], scalar max_seqlen, qkv_format, cp_size/cp_rank.

        Returns:
            Tensor with the same shape as x, in the output projection's compute dtype.
        """
        if x.dim() == 2:
            qkv_format, lead = "thd", (x.shape[0],)
        else:
            qkv_format, lead = "bshd", (x.shape[0], x.shape[1])

        q = self.q_proj(x).view(*lead, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(*lead, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(*lead, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            if freqs_cis is None:
                raise ValueError(f"RoPE layer type {self.layer_type!r} received no rotary frequencies")
            if self.backend.rope_fusion:
                q, k = apply_rotary_emb_qk(
                    q,
                    k,
                    freqs_cis,
                    format=qkv_format,
                    rope_fusion=True,
                    cu_seqlens=attn_kwargs.get("cu_seqlens", None),
                    cp_size=attn_kwargs.get("cp_size", 1),
                    cp_rank=attn_kwargs.get("cp_rank", 0),
                )
            else:
                # Unfused path (the framework currently forces it, BackendConfig #3027): HF's own expression, bf16 q/k
                # against fp32 tables. ``freqs_cis`` already holds this rank's positions (CP shards position_ids).
                cos, sin = rope_tables if rope_tables is not None else hf_rope_tables(freqs_cis)
                q, k = apply_rope_hf(q, cos, sin), apply_rope_hf(k, cos, sin)

        if self.use_ssmax:
            if position_ids is None:
                raise ValueError("SSMax needs position_ids (absolute positions within each sequence)")
            positions = _cp_local_positions(
                position_ids, lead[-1], attn_kwargs.get("cp_size", 1), attn_kwargs.get("cp_rank", 0)
            )
            scale = ssmax_query_scale(positions).unsqueeze(-1).unsqueeze(-1)  # fp32 [B, S, 1, 1] or [T, 1, 1]
            q = (q.float() * scale).to(q.dtype)  # HF multiplies in fp32 and casts back

        out = self._attend(q, k, v, attention_mask, qkv_format, attn_kwargs)
        # Backends differ in output layout (TE returns [..., heads*head_dim], SDPA/flex [..., heads, head_dim]);
        # the gate and its per-head bias act on the [..., heads, head_dim] view.
        out = out.reshape(*lead, self.num_heads, self.head_dim)

        gate = self.gate_proj(x).view_as(out)
        bias_shape = (1,) * (out.dim() - 2) + (self.num_heads, 1)
        out = out * torch.sigmoid(gate + self.gate_bias.view(bias_shape).to(gate.dtype))
        return self.o_proj(out.flatten(out.dim() - 2))

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        qkv_format: str,
        attn_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run backend attention with the correct packed and sliding-window conventions.

        Args:
            q: Tensor of shape [batch, sequence, heads, head_dim] or packed
                [tokens, heads, head_dim], local under CP and in the compute dtype.
            k: Tensor with q's leading axes and shape [..., kv_heads, head_dim].
            v: Tensor with the same shape/dtype/device as k.
            attention_mask: Optional [batch, sequence] valid-token mask or backend-supported
                [batch, 1, sequence, sequence] bool/additive mask.
            qkv_format: bshd for batched input or thd for packed input.
            attn_kwargs: Backend metadata. Integer cu_seqlens/cu_seqlens_padded tensors
                have shape [documents + 1]; max_seqlen and CP size/rank are scalars.

        Returns:
            Attention output with q's leading axes and either [heads, head_dim]
            or flattened [heads * head_dim] trailing axes, depending on the backend.
        """
        attn_impl = self.backend.attn
        if attn_impl == "flex":
            # FlexAttention takes an explicit argument list and HF-style window semantics. It has no padding
            # mask: BeamModel.forward rejects masked (padded) batches on flex (reject_flex_attention_mask).
            if qkv_format == "thd":
                raise NotImplementedError(
                    "Beam: packed (thd) inputs need the 'te' or 'sdpa' attention backend, not flex."
                )
            q, k, v, _ = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask, attn_impl)
            flex_kwargs = {"enable_gqa": self.num_kv_heads != self.num_heads}
            if self.sliding_window is not None:
                flex_kwargs["sliding_window"] = int(self.sliding_window)
            out = self.attn_func(q, k, v, **flex_kwargs)
            return postprocess_output_for_attn(out, attn_impl)

        if attn_impl == "sdpa" and qkv_format == "thd":
            # SDPA has no varlen support: run the packed buffer as one [1, T] batch with an explicit
            # block-diagonal mask (exactly what transformers does for packed inputs).
            cu_seqlens = attn_kwargs.get("cu_seqlens")
            if cu_seqlens is None:
                raise ValueError("packed (thd) input needs cu_seqlens")
            mask = packed_sdpa_mask(cu_seqlens, q.shape[0], self.sliding_window)
            q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask, attn_impl
            )
            self._sdpa_dropout(_attn_kwargs)
            out = self.attn_func(q, k, v, **_attn_kwargs)
            return postprocess_output_for_attn(out, attn_impl).squeeze(0)

        if attn_impl == "te":
            self._set_sliding_cp_transport(qkv_format)
        call_kwargs = {key: val for key, val in attn_kwargs.items() if key != "window_size"}
        if self.window_size is not None:
            call_kwargs["window_size"] = self.window_size
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(q, k, v, attention_mask, attn_impl, **call_kwargs)
        if attn_impl == "sdpa":
            self._sdpa_dropout(_attn_kwargs)
        out = self.attn_func(q, k, v, **_attn_kwargs)
        return postprocess_output_for_attn(out, attn_impl)

    def _set_sliding_cp_transport(self, qkv_format: str) -> None:
        """Select TE's CP exchange for this sliding-window layer from the input layout.

        Automodel's CP setup gives sliding-window layers TE's ``all_gather`` exchange. With packed (thd) input that
        exchange fails before the first step in TE 2.18 (``cudaErrorIllegalInstruction``), so packed input uses
        ``a2a``, which matches ``all_gather`` numerically; batched (bshd) input keeps ``all_gather``. ``a2a`` splits
        heads across the CP group, so the CP size must divide both head counts. Full-attention layers are not touched.

        Args:
            qkv_format: bshd for batched input or thd for packed input.
        """
        cp_group = getattr(self.attn_module, "cp_group", None)
        if self.sliding_window is None or cp_group is None:
            return
        comm_type = "a2a" if qkv_format == "thd" else "all_gather"
        if getattr(self.attn_module, "cp_comm_type", None) == comm_type:
            return
        if comm_type == "a2a":
            cp_size = torch.distributed.get_world_size(cp_group)
            if self.num_heads % cp_size or self.num_kv_heads % cp_size:
                raise ValueError(
                    f"Beam: context parallelism with packed (thd) input uses TE's a2a exchange for sliding-window "
                    f"layers, which needs the CP size ({cp_size}) to divide the query heads ({self.num_heads}) and "
                    f"the KV heads ({self.num_kv_heads})"
                )
        self.attn_module.set_context_parallel_group(
            cp_group,
            getattr(self.attn_module, "cp_global_ranks", None),
            getattr(self.attn_module, "cp_stream", None),
            cp_comm_type=comm_type,
        )

    def _sdpa_dropout(self, attn_kwargs: dict[str, Any]) -> None:
        """Set training-only dropout on the preprocessed SDPA arguments.

        Args:
            attn_kwargs: Mutable SDPA call mapping. Any attn_mask tensor has broadcast
                layout [batch, heads, query_sequence, key_sequence]; tensor values and
                storage are unchanged. Only the scalar dropout_p entry is updated.
        """
        if self.training and self.attention_dropout > 0.0:
            attn_kwargs["dropout_p"] = self.attention_dropout

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize projection and normalization parameters; buffer_device follows the model initializer API."""
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.gate_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if getattr(linear, "bias", None) is not None:
                nn.init.zeros_(linear.bias)
        nn.init.zeros_(self.gate_bias)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()


class BeamMLP(MLP):
    """Shared dense MLP with HF's fp32 SwiGLU boundary, preserving parameter names and MLP dispatch."""

    def __init__(self, dim: int, inter_dim: int, backend: str, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(dim, inter_dim, backend, dtype=dtype, activation="swiglu", bias=False)
        assert self.is_gated and self.gate_proj is not None and self.swiglu_limit == 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU in FP32 before the down projection.

        Args:
            x: Tensor of shape [..., hidden], with arbitrary leading dimensions,
                in the projection compute dtype.

        Returns:
            Tensor of shape [..., hidden] with the same leading dimensions. The FP32
            activation is cast to the gate projection dtype before the down projection.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj((F.silu(gate.float()) * up.float()).to(gate.dtype))


class BeamGate(Gate):
    """Normalize top-1 routing like HF, including its epsilon and gradients.

    The override also serves the CUDA-graph routing core via Gate._route_scores."""

    def _route_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Route scores with HF's normalization for top-1 as well as larger top-k.

        Args:
            scores: Tensor of shape [tokens, experts], using the shared Gate score dtype.

        Returns:
            Routing weights [tokens, topk], integer expert indices [tokens, topk], and
            optional original/normalized scores [tokens, experts], following Gate's
            contract. Weights include route_scale; top-1 normalization retains HF's epsilon.
        """
        weights, indices, original_scores = super()._route_scores(scores)
        if self.norm_topk_prob and self.topk == 1:
            # ``original_scores`` are still the unnormalized (unbiased) scores here: gather the selected ones rather
            # than undoing ``route_scale`` on the returned weights; then HF's normalization and the scale.
            selected = original_scores.gather(1, indices)
            weights = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20) * self.route_scale
            original_scores = original_scores / (original_scores.sum(dim=-1, keepdim=True) + 1e-20)
        return weights, indices, original_scores


class BeamMoE(MoE):
    """Shared MoE with BeamGate installed before loading or parallelization.

    Shared compiled SwiGLU preserves the FP32 activation boundary; Dynamo eager
    fallback can lose it. Unlike HF, default expert routing weights apply before
    the down projection and expert outputs accumulate in FP32."""

    def __init__(self, config: MoEConfig, backend: BackendConfig) -> None:
        super().__init__(config, backend)
        if not backend.fake_balanced_gate:
            gate = BeamGate(config, gate_precision=backend.gate_precision)
            gate.use_routing_core = self.gate.use_routing_core
            self.gate = gate
        if config.expert_activation != "swiglu":
            raise NotImplementedError("Beam experts use SwiGLU")


def is_moe_layer(config: BeamConfig, layer_idx: int) -> bool:
    """Beam marks every layer ``"dense"`` or ``"sparse"`` in ``config.mlp_layer_types``."""
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        raise ValueError(
            "Beam config has no mlp_layer_types; pre-Beam (Reflection) configs must be converted first "
            "(convert_reflection_to_beam.py)."
        )
    return mlp_layer_types[layer_idx] == "sparse"


class Block(nn.Module):
    """One Beam decoder layer: sandwich norms + depth-scaled residual adds."""

    def __init__(self, layer_idx: int, config: BeamConfig, moe_config: MoEConfig, backend: BackendConfig) -> None:
        super().__init__()
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.layer_idx = layer_idx
        # Read by Automodel's CP setup (moe.parallelizer.apply_cp): sliding-window layers get TE's all_gather CP
        # communication (BeamAttention switches packed input to a2a), full-attention layers p2p.
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = BeamAttention(config, layer_idx, backend)
        if getattr(config, "hidden_act", "silu") != "silu":
            raise NotImplementedError(
                f"Beam: hidden_act {config.hidden_act!r} is not supported (SwiGLU with SiLU only)"
            )
        if is_moe_layer(config, layer_idx):
            self.mlp = BeamMoE(moe_config, backend)
        else:
            self.mlp = BeamMLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)

        def norm() -> nn.Module:
            return initialize_rms_norm_module(
                backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
            )

        # HF (Gemma-2 convention): input -> attention -> post_attention -> residual; pre_feedforward -> MLP ->
        # post_feedforward -> residual.
        self.input_layernorm = norm()
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.post_feedforward_layernorm = norm()
        self.attn_scales = depth_scaled_residual_coefs(layer_idx, is_attention=True)
        self.mlp_scales = depth_scaled_residual_coefs(layer_idx, is_attention=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor | None,
        rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        # Under FSDP2 the block's unit does not cast its inputs (the fp32 RoPE tensors must stay fp32), so the
        # hidden state is cast here to the unit's compute dtype, recorded by ``beam_fully_shard_unit``: the resolved
        # mixed-precision ``param_dtype`` (bf16 by default, also with fp32 master weights), never the storage dtype.
        """Apply attention, feedforward computation and depth-scaled residuals.

        Args:
            x: Tensor [batch, sequence, hidden] or packed [tokens, hidden], local under CP;
                only x is cast to the FSDP compute dtype.
            freqs_cis: FP32 rotary tensor [..., head_dim], aligned with x's leading axes;
                concatenated half-width cos/sin when unfused, fused layout otherwise. None for NoPE.
            rope_tables: Optional FP32 (cos, sin), each [..., 1, head_dim], shared across
                layers without mutation or casting; leading axes match x.
            position_ids: Integer tensor [batch, sequence] or [tokens], absolute within each document.
            attention_mask: Optional [batch, sequence] bool/0-1 mask; True/1 means valid.
            padding_mask: Optional bool tensor matching x's leading axes; True skips routing.
            **attn_kwargs: Backend metadata: qkv_format (bshd/thd), integer cu_seqlens and
                cu_seqlens_padded [documents + 1], scalar max_seqlen, cp_size and cp_rank.

        Returns:
            Tensor with x's shape, covering this rank's local positions."""
        compute_dtype = getattr(self, "_fsdp_compute_dtype", None)
        if compute_dtype is not None and x.is_floating_point() and x.dtype != compute_dtype:
            x = x.to(compute_dtype)
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()
        attn_out = self.self_attn(
            self.input_layernorm(x),
            freqs_cis=freqs_cis,
            rope_tables=rope_tables,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        attn_out = self.post_attention_layernorm(attn_out)
        out_scale, res_scale = self.attn_scales
        x = attn_out * out_scale + x * res_scale

        mlp_out = self._mlp(self.pre_feedforward_layernorm(x), padding_mask)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        out_scale, res_scale = self.mlp_scales
        return mlp_out * out_scale + x * res_scale

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Dispatch to the dense MLP or the routed experts.

        Args:
            x: Tensor of shape [batch, sequence, hidden] or packed [tokens, hidden],
                local to this rank under CP, in the layer compute dtype.
            padding_mask: Optional bool tensor matching x's leading axes; True skips routing.

        Returns:
            Tensor with the same shape as x.
        """
        mlp = unwrap_checkpoint_wrapper(self.mlp)
        if isinstance(mlp, MLP):
            return self.mlp(x)
        assert isinstance(mlp, MoE)
        return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize the layer's norms, attention, MLP and router buffers on buffer_device."""
        for norm in (
            self.input_layernorm,
            self.post_attention_layernorm,
            self.pre_feedforward_layernorm,
            self.post_feedforward_layernorm,
        ):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)
