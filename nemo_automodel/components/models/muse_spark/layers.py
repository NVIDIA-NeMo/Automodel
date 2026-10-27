# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text layers for the native Muse Spark Open implementation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import lru_cache
from typing import Any, ParamSpec, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import FakeBalancedGate, Gate, MoE

_NormArgs = ParamSpec("_NormArgs")
_NormOutput = TypeVar("_NormOutput")


def calculate_gain_center(gain_center_type: str, residual_id: int) -> float:
    """Return the checkpoint-defined zero-centered RMSNorm gain offset."""
    if gain_center_type == "one":
        return 1.0
    if gain_center_type == "zero":
        return 0.0
    if gain_center_type == "inverse_residual_index":
        return 1.0 / (2.0 * math.ceil(residual_id / 2.0))
    if gain_center_type == "inverse_residual_index_v2":
        return 1.0 / (residual_id + 1.0)
    if gain_center_type == "inverse_sqrt_residual_index":
        return 1.0 / math.sqrt(2.0 * math.ceil(residual_id / 2.0))
    raise ValueError(f"Unsupported Muse Spark Open gain center {gain_center_type!r}")


def _rms_norm_core(hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Normalize in FP32, then apply the effective gain before the output cast.

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Optional effective FP32 gain of shape [hidden], on the input device.
        eps: Stabilizer added to the mean squared activation.

    Returns:
        Independent tensor of shape [..., hidden], in the input dtype and device.
    """
    output = hidden_states.float()
    output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        output = output * weight
    return output.to(hidden_states.dtype)


def _input_scale_norm_core(hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Apply the effective gain before FP32 RMS normalization.

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Optional effective FP32 gain of shape [hidden], on the input device.
        eps: Stabilizer added to the mean squared scaled activation.

    Returns:
        Independent tensor of shape [..., hidden], in the input dtype and device.
    """
    output = hidden_states.float()
    if weight is not None:
        output = output * weight
    output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + eps)
    return output.to(hidden_states.dtype)


def _residual_core(
    stream: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    """Blend the residual and branch using the checkpoint's learned gate.

    Args:
        stream: Residual tensor of shape [..., hidden], ordinarily FP32.
        branch: Tensor of shape [..., hidden], matching the stream shape and device.
        gate: Trainable gate of shape [hidden], on the stream device.
        temperature: Gate temperature.
        eps: Lower bound on the squared residual coefficient.

    Returns:
        Independent tensor of shape [..., hidden], in the stream dtype and device.
    """
    gate = gate.float() / temperature
    beta = torch.sigmoid(gate)
    alpha_squared = torch.sigmoid(-gate) * (1.0 + beta)
    alpha = torch.sqrt(torch.clamp(alpha_squared, min=eps))
    return (alpha * stream + beta * branch.to(stream.dtype)).to(stream.dtype)


def _norm_residual_core(
    stream: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor | None,
    gate: torch.Tensor,
    *,
    input_scale: bool,
    norm_eps: float,
    temperature: float,
    residual_eps: float,
) -> torch.Tensor:
    """Fuse a branch norm and residual blend without changing the branch cast.

    Args:
        stream: Residual tensor of shape [..., hidden], ordinarily FP32.
        branch: Tensor of shape [..., hidden], matching the stream shape and device.
            Normalization casts back to this dtype before residual blending.
        weight: Optional effective FP32 norm gain of shape [hidden].
        gate: Trainable residual gate of shape [hidden], on the stream device.
        input_scale: Whether the gain precedes rather than follows normalization.
        norm_eps: RMS normalization stabilizer.
        temperature: Residual gate temperature.
        residual_eps: Lower bound on the squared residual coefficient.

    Returns:
        Independent tensor of shape [..., hidden], in the stream dtype and device.
    """
    norm = _input_scale_norm_core if input_scale else _rms_norm_core
    branch = norm(branch, weight, norm_eps)
    return _residual_core(stream, branch, gate, temperature, residual_eps)


@lru_cache(maxsize=5)
def _compiled_norm_core(core: Callable[_NormArgs, _NormOutput]) -> Callable[_NormArgs, _NormOutput]:
    """Lazily share each of the five pure compiled cores within this process.

    The cache retains callables, never model parameters or activations. Eager
    instances keep their original callables; enabling one model does not change
    another model's dispatch. Compilation starts on the first real tensor call.
    """
    # Preserve explicit BF16 round trips at norm/residual boundaries, including
    # casts in the generated backward. This option is local to these callables.
    compiled = torch.compile(core, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})

    return _checkpoint_recomputed_math(compiled)


def _checkpoint_recomputed_math(fn: Callable[_NormArgs, _NormOutput]) -> Callable[_NormArgs, _NormOutput]:
    """Keep pure compiled math and autotuner scratch tensors outside SAC caching.

    Muse's selective checkpoint policy saves routing results, not norm or
    activation math. Autotuning a new shape during recomputation can execute
    scratch operations absent from the original forward. Exclude these pure
    islands from dispatch-mode accounting in both passes; autograd and the
    enclosing activation checkpoint still run.
    """

    def recomputed_math(*args: _NormArgs.args, **kwargs: _NormArgs.kwargs) -> _NormOutput:
        from torch.utils._python_dispatch import _disable_current_modes

        with _disable_current_modes():
            return fn(*args, **kwargs)

    return recomputed_math


class MuseSparkRMSNorm(nn.Module):
    """FP32-accumulating RMSNorm with optional zero-centered checkpoint gain."""

    _norm_core = staticmethod(_rms_norm_core)

    def __init__(
        self,
        dim: int,
        eps: float,
        *,
        dtype: torch.dtype,
        has_weight: bool = True,
        zero_centered_gamma: bool = False,
        gain_center: float = 1.0,
        compile_norm: bool = False,
    ) -> None:
        """Create a checkpoint-compatible RMS normalization layer.

        Args:
            dim: Size of the last activation dimension.
            eps: Stabilizer added to the mean squared activation.
            dtype: Storage dtype for the optional trainable gain.
            has_weight: Whether to allocate a trainable gain.
            zero_centered_gamma: Whether to add gain_center to the stored gain.
            gain_center: Offset used for zero-centered checkpoint gains.
            compile_norm: Compile the tensor core with preserved precision casts.
        """
        super().__init__()
        self._forward_core = _compiled_norm_core(self._norm_core) if compile_norm else self._norm_core
        self.dim = dim
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.gain_center = gain_center
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype)) if has_weight else None

    def effective_weight(self) -> torch.Tensor | None:
        """Return the actual multiplicative gain with shape ``[hidden]``."""
        if self.weight is None:
            return None
        return self.weight.float() + self.gain_center if self.zero_centered_gamma else self.weight.float()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize activations with the selected eager or compiled core.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Independent normalized tensor with the input shape, dtype and device.
        """
        # Empty expert ranks still need a connected zero gain gradient. Eager
        # empty math provides it without compiling/loading a zero-token kernel
        # while other pipeline ranks may be waiting in communication.
        core = self._norm_core if hidden_states.numel() == 0 else self._forward_core
        return core(hidden_states, self.effective_weight(), self.eps)

    def reset_parameters(self) -> None:
        """Initialize the serialized gain without changing its effective value."""
        if self.weight is not None:
            if self.zero_centered_gamma:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)


class MuseSparkInputScaleRMSNorm(MuseSparkRMSNorm):
    """Apply the learned gain before parameter-free RMS normalization."""

    _norm_core = staticmethod(_input_scale_norm_core)


class MuseSparkResidual(nn.Module):
    """Learned norm-preserving residual connection from the Muse checkpoint."""

    def __init__(self, dim: int, *, dtype: torch.dtype, temperature: float = 0.3, eps: float = 1e-3) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim, dtype=dtype))

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        """Blend activations while preserving the residual stream dtype.

        Args:
            stream: Residual tensor of shape [..., hidden], ordinarily FP32.
            branch: Tensor with the same shape and device as stream.

        Returns:
            Independent tensor with the stream shape, dtype and device.
        """
        return _residual_core(stream, branch, self.gate, self.temperature, self.eps)


class MuseSparkRotaryEmbedding(nn.Module):
    """FP32 adjacent-pair rotary frequencies shared by all text layers."""

    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", torch.empty(dim // 2, dtype=torch.float32), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Recreate FP32 frequencies after meta-device materialization."""
        exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device) / self.dim
        self.inv_freq = 1.0 / (self.theta**exponent)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine/sine tensors shaped ``[batch, sequence, head_dim / 2]``."""
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        inv_freq = inv_freq.to(hidden_states.device)
        positions = position_ids[:, None, :].float()
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq @ positions).transpose(1, 2)
            cosine, sine = frequencies.cos(), frequencies.sin()
        return cosine.to(hidden_states.dtype), sine.to(hidden_states.dtype)


def apply_rotary_embedding(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    upcast: bool = True,
) -> torch.Tensor:
    """Apply adjacent-pair RoPE with the checkpoint's arithmetic precision.

    The original checkpoint computes the rotation in FP32. The HF release
    rounds both products in the Q/K dtype before adding them.
    """
    cosine, sine = position_embeddings
    compute_dtype = torch.float32 if upcast else hidden_states.dtype
    if hidden_states.ndim == 4:
        cosine, sine = cosine.unsqueeze(2).to(compute_dtype), sine.unsqueeze(2).to(compute_dtype)
    elif hidden_states.ndim == 3:
        if cosine.shape[0] != 1:
            raise ValueError("Muse Spark Open THD RoPE requires one flattened position stream")
        cosine = cosine.squeeze(0).unsqueeze(1).to(compute_dtype)
        sine = sine.squeeze(0).unsqueeze(1).to(compute_dtype)
    else:
        raise ValueError(f"Muse Spark Open RoPE expected BSHD or THD, got {tuple(hidden_states.shape)}")
    pairs = hidden_states.to(compute_dtype).unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine),
        dim=-1,
    ).flatten(-2)
    return rotated.to(hidden_states.dtype)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat BSHD key/value heads from ``[B,Nkv,S,D]`` to query-head count."""
    if repeats == 1:
        return hidden_states
    batch, heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(batch, heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, heads * repeats, sequence, head_dim)


def _qk_norm_rope_core(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    eps: float,
    query_scale: float,
    use_rope: bool,
    upcast_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate Q/K while preserving the checkpoint's dtype boundaries.

    Args:
        query: Query tensor [B,S,Hq,D] or packed [T,Hq,D].
        key: Key tensor with matching leading dimensions and Hkv heads.
        cosine: Rotary cosines [B,S,D/2], or [1,T,D/2] for packed inputs.
        sine: Rotary sines with the same shape and dtype as cosine.
        eps: FP32 RMS normalization stabilizer.
        query_scale: Multiplier applied after the normalized query's dtype cast.
        use_rope: Whether this layer uses adjacent-pair rotary embeddings.
        upcast_rope: Whether rotation uses FP32 (legacy) or the Q/K dtype (v2).

    Returns:
        Independent normalized query and key tensors in their original layouts
        and dtypes. BF16 round trips match the unfused model, including the
        query-scale multiplication and the two v2 rotary products.
    """
    query = _rms_norm_core(query, None, eps) * query_scale
    key = _rms_norm_core(key, None, eps)
    if use_rope:
        query = apply_rotary_embedding(query, (cosine, sine), upcast=upcast_rope)
        key = apply_rotary_embedding(key, (cosine, sine), upcast=upcast_rope)
    return query, key


def _output_norm_gate_core(output: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """Fuse parameter-free output RMSNorm and the checkpoint's sigmoid gate.

    Args:
        output: Attention output [B,S,Hq,D] or packed [T,Hq,D].
        gate: Gate projection with the same shape, dtype and device as output.
        eps: FP32 RMS normalization stabilizer.

    Returns:
        Gated output in the original dtype. Both normalization and sigmoid
        retain their original FP32 computation and separate BF16 casts.
    """
    return _rms_norm_core(output, None, eps) * torch.sigmoid(gate.float()).to(output.dtype)


class MuseSparkAttention(nn.Module):
    """Muse GQA with Q/K and output RMSNorm, output gate, RoPE, SDPA, and TE."""

    def __init__(self, config: MuseSpark1xOmniConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.layer_sliding_windows[layer_idx] or None
        self.use_rope = config.layer_uses_rope(layer_idx)
        self.rope_in_fp32 = config.model_type != "muse_spark"
        query_width = self.num_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        dtype = next_dtype(config)
        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, query_width, bias=config.attention_bias, dtype=dtype
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, kv_width, bias=config.attention_bias, dtype=dtype
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, kv_width, bias=config.attention_bias, dtype=dtype
        )
        self.o_gate_proj = initialize_linear_module(
            backend.linear, config.hidden_size, query_width, bias=False, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            backend.linear, query_width, config.hidden_size, bias=config.attention_bias, dtype=dtype
        )
        self.qk_norm = MuseSparkRMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=False,
        )
        self.o_norm = MuseSparkRMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=False,
        )
        self._fused_qk_norm_rope = _compiled_norm_core(_qk_norm_rope_core) if backend.compile_norm else None
        self._fused_output_norm_gate = _compiled_norm_core(_output_norm_gate_core) if backend.compile_norm else None
        if backend.attn == "te":
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=self.num_heads,
                num_qk_channels=self.head_dim,
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                num_gqa_groups=self.num_key_value_heads,
                attention_dropout=config.attention_dropout,
            )
        elif backend.attn not in ("sdpa", "eager", "magi"):
            raise ValueError(
                f"Muse Spark Open supports attention backends 'sdpa', 'eager', 'te', and 'magi', got {backend.attn!r}"
            )

    def _shape_qkv(
        self,
        hidden_states: torch.Tensor,
        *,
        is_thd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``[B,S,H]`` or ``[T,H]`` states into matching BSHD/THD QKV."""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if is_thd:
            tokens = hidden_states.shape[0]
            return (
                query.view(tokens, self.num_heads, self.head_dim),
                key.view(tokens, self.num_key_value_heads, self.head_dim),
                value.view(tokens, self.num_key_value_heads, self.head_dim),
            )
        batch, sequence = hidden_states.shape[:2]
        return (
            query.view(batch, sequence, self.num_heads, self.head_dim),
            key.view(batch, sequence, self.num_key_value_heads, self.head_dim),
            value.view(batch, sequence, self.num_key_value_heads, self.head_dim),
        )

    def _te_window_size(self, *, is_thd: bool, max_real_seqlen: int | None) -> tuple[int, int]:
        """Return TE's left/right window, treating short packed documents as full causal."""
        if self.sliding_window is None:
            return (-1, 0)
        if is_thd and max_real_seqlen is not None and max_real_seqlen <= self.sliding_window:
            return (-1, 0)
        return (self.sliding_window - 1, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Attend over ordinary or packed states using this layer's CP transport.

        Args:
            hidden_states: CP-local states [batch, local_sequence, hidden] or
                packed [local_tokens, hidden].
            position_embeddings: Rotary cosine/sine tensors [batch, local_sequence,
                head_dim] or packed [local_tokens, head_dim].
            attention_mask: Optional mask broadcastable to [batch, heads,
                sequence, sequence]; unused for packed THD.
            kwargs: Attention metadata, including qkv_format and global packed
                document boundaries cu_seqlens [documents + 1].

        Returns:
            Attention output with the same local layout as hidden_states.
        """
        is_thd = kwargs.get("qkv_format") == "thd"
        expected_ndim = 2 if is_thd else 3
        if hidden_states.ndim != expected_ndim:
            raise ValueError(
                f"Muse Spark Open attention expected rank {expected_ndim}, got {tuple(hidden_states.shape)}"
            )
        query, key, value = self._shape_qkv(hidden_states, is_thd=is_thd)
        if self._fused_qk_norm_rope is not None:
            query, key = self._fused_qk_norm_rope(
                query,
                key,
                *position_embeddings,
                self.config.rms_norm_eps,
                self.config.scale_query_by,
                self.use_rope,
                self.rope_in_fp32,
            )
        else:
            query = self.qk_norm(query) * self.config.scale_query_by
            key = self.qk_norm(key)
            if self.use_rope:
                query = apply_rotary_embedding(query, position_embeddings, upcast=self.rope_in_fp32)
                key = apply_rotary_embedding(key, position_embeddings, upcast=self.rope_in_fp32)

        use_te = self.backend.attn == "te"
        max_real = kwargs.get("_muse_spark_thd_max_real_seqlen", kwargs.get("max_seqlen")) if is_thd else None
        if isinstance(max_real, torch.Tensor):
            max_real = int(max_real.item())
        window_size = self._te_window_size(is_thd=is_thd, max_real_seqlen=max_real)
        if use_te and self.attn_module.cp_group is not None:
            # Each layer owns its transport, including later virtual PP chunks
            # that never run batch preparation. TE >= 2.18 supports packed THD
            # AllGather with a real sliding window. Short packed documents still
            # use the equivalent full-causal P2P path.
            communication_type = "p2p" if window_size == (-1, 0) else "all_gather"
            if self.attn_module.cp_comm_type != communication_type:
                self.attn_module.set_context_parallel_group(
                    self.attn_module.cp_group,
                    self.attn_module.cp_global_ranks,
                    self.attn_module.cp_stream,
                    cp_comm_type=communication_type,
                )
        if self.backend.attn == "magi":
            if not is_thd:
                raise NotImplementedError("Muse Magi attention requires packed THD inputs")
            plan = kwargs.get("_muse_spark_magi_plan")
            if plan is None:
                raise ValueError("Muse Magi attention requires model-owned CP batch preparation")
            output = plan.attention(query, key, value, window=int(self.sliding_window or 0), scale=self.scaling)
        elif is_thd:
            if not use_te:
                raise ValueError("Packed Muse Spark Open THD attention requires TE or Magi")
            query, key, value, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query,
                key,
                value,
                None,
                "te",
                window_size=window_size,
                **kwargs,
            )
            output = self.attn_module(query, key, value, **te_kwargs)
            output = postprocess_output_for_attn(output, "te")
            output = output.view(hidden_states.shape[0], self.num_heads, self.head_dim)
        elif use_te:
            query, key, value, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query,
                key,
                value,
                attention_mask,
                "te",
                window_size=window_size,
                **kwargs,
            )
            output = self.attn_module(query, key, value, **te_kwargs)
            output = postprocess_output_for_attn(output, "te")
            output = output.view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        else:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            key = _repeat_kv(key, self.num_key_value_groups)
            value = _repeat_kv(value, self.num_key_value_groups)
            dropout = self.config.attention_dropout if self.training else 0.0
            if self.backend.attn == "eager":
                scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
                if attention_mask is None:
                    attention_mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
                if attention_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + attention_mask
                probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                probabilities = F.dropout(probabilities, p=dropout, training=self.training)
                output = torch.matmul(probabilities, value)
            else:
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=dropout,
                    is_causal=attention_mask is None and query.shape[-2] > 1,
                    scale=self.scaling,
                )
            output = output.transpose(1, 2).contiguous()

        if self._fused_output_norm_gate is not None:
            gate = self.o_gate_proj(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
            output = self._fused_output_norm_gate(output, gate, self.config.rms_norm_eps)
        else:
            output = self.o_norm(output)
            gate = self.o_gate_proj(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
            output = output * torch.sigmoid(gate.float()).to(output.dtype)
        return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1))


class MuseSparkGate(Gate):
    """FP32 MetaP-scaled sigmoid router with bias used for selection only."""

    def __init__(self, config: MoEConfig, router_output_multiplier: float, division_eps: float) -> None:
        super().__init__(config, gate_precision=torch.float32)
        self.router_output_multiplier = router_output_multiplier
        self.division_eps = division_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Route local ``[tokens, hidden]`` states and return FP32 top-k weights."""
        del token_mask, cp_mesh
        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = torch.sigmoid(logits * self.router_output_multiplier)
        selection_scores = scores
        correction_bias = self._local_score_correction_bias()
        if correction_bias is not None:
            selection_scores = selection_scores + correction_bias
        indices = selection_scores.topk(self.topk, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.division_eps)
        return weights * self.route_scale, indices, None


def build_muse_spark_moe_config(
    config: MuseSpark1xOmniConfig,
    dtype: torch.dtype,
    overrides: dict[str, Any] | None = None,
) -> MoEConfig:
    """Translate the released Muse router and compressed-expert contract."""
    values: dict[str, Any] = {
        "dim": config.hidden_size,
        "inter_dim": config.intermediate_size,
        "moe_inter_dim": config.moe_intermediate_size,
        "n_routed_experts": config.num_local_experts,
        "n_shared_experts": 0,
        "n_activated_experts": config.num_experts_per_tok,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "train_gate": True,
        "gate_bias_update_factor": 0.0,
        "aux_loss_coeff": 0.0,
        "score_func": "sigmoid_with_bias",
        "route_scale": float(config.moe.get("route_scale", 1.0)),
        "norm_topk_prob": True,
        "router_bias": False,
        "expert_bias": False,
        "expert_activation": "swiglu",
        "apply_router_weight_after_down": True,
        "router_weights_fp32": True,
        "gate_dtype": torch.float32,
        "force_e_score_correction_bias": True,
        "moe_latent_size": config.expert_hidden_size,
        "dtype": dtype,
    }
    if overrides:
        values.update(overrides)
    return MoEConfig(**values)


class MuseSparkMoE(MoE):
    """Compressed Muse experts with per-expert norms and grouped-GEMM dispatch."""

    def __init__(
        self,
        config: MuseSpark1xOmniConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ) -> None:
        super().__init__(moe_config, backend)
        # Expert activation is another pure compiled island. Keep the router,
        # dispatch/combine, and expert GEMMs under their existing AC policy.
        if hasattr(self.experts, "expert_activation"):
            self.experts.expert_activation = _checkpoint_recomputed_math(self.experts.expert_activation)
        if backend.fake_balanced_gate:
            self.gate = FakeBalancedGate(moe_config, noise=backend.fake_gate_noise)
        else:
            self.gate = MuseSparkGate(
                moe_config,
                config.router_output_multiplier,
                float(config.moe.get("router_score_division_eps", 1e-15)),
            )
        dtype = next_dtype(config)
        self.pre_expert_norm = MuseSparkRMSNorm(
            config.expert_hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=bool(config.use_ffn_learnable_scales),
            zero_centered_gamma=config.norm_zero_centered_gamma,
        )
        self.post_expert_norm = MuseSparkInputScaleRMSNorm(
            config.expert_hidden_size,
            config.rms_post_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=bool(config.use_ffn_learnable_scales or config.post_norm_affine),
            zero_centered_gamma=config.norm_zero_centered_gamma,
            gain_center=calculate_gain_center(config.post_norm_gain_center_type, 2 * layer_idx + 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cp_mesh: Any = None,
    ) -> torch.Tensor:
        """Route ``[..., hidden]`` states through compressed, normalized experts."""
        shape = hidden_states.shape
        tokens = hidden_states.reshape(-1, self.dim)
        token_mask = (
            ~padding_mask.reshape(-1).bool()
            if padding_mask is not None
            else torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        )
        weights, indices, _ = self.gate(tokens, token_mask, cp_mesh if cp_mesh is not None else self.cp_mesh)
        if self.fc1_latent_proj is None or self.fc2_latent_proj is None:
            raise RuntimeError("Muse Spark Open requires latent pooler and unpooler projections")
        expert_input = self.pre_expert_norm(self.fc1_latent_proj(tokens))
        output = self.experts(
            expert_input,
            token_mask,
            weights,
            indices,
            post_expert_output=self.post_expert_norm,
        )
        return self.fc2_latent_proj(output).view(shape)


class MuseSparkDecoderLayer(GradientCheckpointingLayer):
    """One FP32 residual-stream Muse Spark Open decoder layer."""

    def __init__(
        self,
        config: MuseSpark1xOmniConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ) -> None:
        super().__init__()
        dtype = next_dtype(config)
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = MuseSparkAttention(config, layer_idx, backend)
        self.self_attn.pre_qkv_proj_norm = MuseSparkRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=config.norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
        )
        self.post_attn_norm = MuseSparkRMSNorm(
            config.hidden_size,
            config.rms_post_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=config.post_norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
            gain_center=calculate_gain_center(config.post_norm_gain_center_type, 2 * layer_idx + 1),
        )
        self.pre_ffn_norm = MuseSparkRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=config.norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
        )
        self.feed_forward = MuseSparkMoE(config, layer_idx, moe_config, backend)
        self.post_ffn_norm = MuseSparkInputScaleRMSNorm(
            config.hidden_size,
            config.rms_post_norm_eps,
            dtype=dtype,
            compile_norm=backend.compile_norm,
            has_weight=bool(config.use_ffn_learnable_scales or config.post_norm_affine),
            zero_centered_gamma=config.norm_zero_centered_gamma,
            gain_center=calculate_gain_center(config.post_norm_gain_center_type, 2 * layer_idx + 2),
        )
        self.add_attn = MuseSparkResidual(config.hidden_size, dtype=dtype, temperature=config.residual_gate_temperature)
        self.add_ffn = MuseSparkResidual(config.hidden_size, dtype=dtype, temperature=config.residual_gate_temperature)
        self._fused_norm_residual = _compiled_norm_core(_norm_residual_core) if backend.compile_norm else None

    def nemo_checkpoint_context_fn(
        self,
        context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None,
    ) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None:
        """Preserve norm compilation limits on the checkpoint recomputation thread.

        Args:
            context_fn: Existing forward/recompute context factory, if present.

        Returns:
            A factory composing the existing contexts with the forward thread's
            Dynamo compilation budget. Eager layers retain the original factory.
        """
        if self._fused_norm_residual is None:
            return context_fn
        cache_limit = torch._dynamo.config.cache_size_limit
        accumulated_limit = torch._dynamo.config.accumulated_cache_size_limit

        @contextmanager
        def with_compile_budget(context: AbstractContextManager) -> Iterator[None]:
            with (
                context,
                torch._dynamo.config.patch(
                    cache_size_limit=cache_limit,
                    accumulated_cache_size_limit=accumulated_limit,
                ),
            ):
                yield

        def checkpoint_contexts() -> tuple[AbstractContextManager, AbstractContextManager]:
            forward, recompute = context_fn() if context_fn is not None else (nullcontext(), nullcontext())
            return with_compile_budget(forward), with_compile_budget(recompute)

        return checkpoint_contexts

    @property
    def mlp(self) -> MuseSparkMoE:
        """Expose the checkpoint-named feed-forward through the generic MoE protocol."""
        return self.feed_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform batched or packed residual states through attention and MoE.

        Args:
            hidden_states: Residual activations shaped [B, S, H] or [T, H].
            position_embeddings: Cosine and sine tensors shaped [B, S, head_dim / 2].
            attention_mask: Optional mask in the selected attention backend's layout.
            padding_mask: Optional token padding mask shaped [B, S] or [T].
            **kwargs: Additional arguments for the selected attention backend.

        Returns:
            Independent residual states with the input shape, dtype and device.
        """
        branch_dtype = self.self_attn.q_proj.weight.dtype
        attention_input = self.self_attn.pre_qkv_proj_norm(hidden_states).to(branch_dtype)
        attention_output = self.self_attn(
            attention_input,
            position_embeddings,
            attention_mask,
            **kwargs,
        )
        if self._fused_norm_residual is None:
            hidden_states = self.add_attn(hidden_states, self.post_attn_norm(attention_output))
        else:
            hidden_states = self._fused_norm_residual(
                hidden_states,
                attention_output,
                self.post_attn_norm.effective_weight(),
                self.add_attn.gate,
                input_scale=False,
                norm_eps=self.post_attn_norm.eps,
                temperature=self.add_attn.temperature,
                residual_eps=self.add_attn.eps,
            )
        ffn_input = self.pre_ffn_norm(hidden_states).to(branch_dtype)
        ffn_output = self.feed_forward(ffn_input, padding_mask=padding_mask)
        if self._fused_norm_residual is None:
            return self.add_ffn(hidden_states, self.post_ffn_norm(ffn_output))
        return self._fused_norm_residual(
            hidden_states,
            ffn_output,
            self.post_ffn_norm.effective_weight(),
            self.add_ffn.gate,
            input_scale=True,
            norm_eps=self.post_ffn_norm.eps,
            temperature=self.add_ffn.temperature,
            residual_eps=self.add_ffn.eps,
        )


def next_dtype(config: MuseSpark1xOmniConfig) -> torch.dtype:
    """Resolve the config's model dtype without importing recipe utilities."""
    value = getattr(config, "torch_dtype", getattr(config, "dtype", torch.bfloat16))
    if isinstance(value, torch.dtype):
        return value
    mapping = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
    return mapping.get(str(value).replace("torch.", "").lower(), torch.bfloat16)


__all__ = [
    "MuseSparkAttention",
    "MuseSparkDecoderLayer",
    "MuseSparkGate",
    "MuseSparkInputScaleRMSNorm",
    "MuseSparkMoE",
    "MuseSparkRMSNorm",
    "MuseSparkResidual",
    "MuseSparkRotaryEmbedding",
    "build_muse_spark_moe_config",
]
