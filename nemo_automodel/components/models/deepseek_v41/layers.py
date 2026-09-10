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

"""DeepSeek V4.1 layers.

Ported from the released ``inference/model.py`` of
``deepseek-ai/DeepSeek-V4.1-Flash`` for stateless training forwards:

* :class:`DeepseekV41Compressor`: ``r``-to-1 softmax-gated pooling of the KV
  latent (no overlap, no absolute positional embedding, unlike V4's CSA).
  ``r == 1`` is a plain projection.
* :class:`DeepseekV41Indexer`: FP4-QAT lightning indexer whose keys are
  projected from the *shared* compressed latent, with the Hierarchical Sparse
  Indexer candidate pool.
* :class:`DeepseekV41Attention`: sliding-window KV plus ``index_topk``
  compressed positions from a KV cache that is shared across layers (CSA2
  Full / Reindex / Reuse modes), combined in one sparse attention call with a
  per-head sink.
* :class:`DeepseekV41Block`: single-pass mHC block.  The attention input mix
  uses the coefficients produced by the *previous* block's FFN site.
* FP8 (window KV) and FP4 (compressed KV, indexer Q/K) fake quantization with
  a straight-through estimator, matching the released quantization-aware
  training numerics.

Cross-layer state travels in per-layer :class:`DeepseekV41SharedState` snapshots.
The model shares their tensors while isolating assignments during recomputation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4FP32Parameter,
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
)
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4VisionGate
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    dsv4_sinkhorn_normalize,
    dsv4_sparse_attention,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram, EngramLayout
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

__all__ = [
    "DeepseekV41Attention",
    "DeepseekV41AttentionOutput",
    "DeepseekV41Block",
    "DeepseekV41Compressor",
    "DeepseekV41Indexer",
    "DeepseekV41RMSNorm",
    "DeepseekV41RotaryEmbedding",
    "DeepseekV41SharedState",
    "build_compressed_visibility",
    "build_window_topk_indices",
    "fake_quant_fp4",
    "fake_quant_fp8",
    "hc_collapse",
    "hc_expand",
    "make_identity_pre_mix",
    "select_candidate_blocks",
]


class DeepseekV41RMSNorm(nn.Module):
    """Normalize in FP32 and multiply the scale before casting the result."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def reset_parameters(self) -> None:
        """Initialize the scale after construction or meta materialization."""
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the released reference's RMS normalization.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden], with the input dtype.
        """
        value = hidden_states.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * value).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Rotary embedding: the released config spells the YaRN block as ``rope_type``
# (HF convention) whereas the shared V4 module reads ``type``.
# ---------------------------------------------------------------------------


class DeepseekV41RotaryEmbedding(nn.Module):
    """Construct reference FP32 phases from scalar settings after any model cast."""

    def __init__(
        self,
        rope_theta: float,
        head_dim: int,
        partial_rotary_factor: float,
        attention_scaling: float = 1.0,
        device: torch.device | None = None,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        del device  # Frequencies are constructed on the runtime input device.
        self.dim = int(head_dim * partial_rotary_factor)
        self.theta = rope_theta
        self.attention_scaling = attention_scaling
        scaling = rope_scaling or {}
        self.factor = float(scaling.get("factor", 1.0))
        self.original_length = int(scaling.get("original_max_position_embeddings", 0))
        self.beta_fast = float(scaling.get("beta_fast", 32))
        self.beta_slow = float(scaling.get("beta_slow", 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build adjacent-pair rotations while retaining the existing cosine/sine API.

        Args:
            x: Tensor of shape [batch, sequence, hidden], supplying the device.
            position_ids: Integer tensor of shape [batch, sequence], including
                document-relative positions for packed inputs.

        Returns:
            FP32 cosine and sine tensors of shape [batch, sequence, rotary_dim].
            Their second halves repeat the first halves; rotary_dim equals
            head_dim times partial_rotary_factor. No rounded frequency buffer
            survives a model dtype conversion.
        """
        frequencies = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
        )
        if self.original_length > 0:
            low = max(
                math.floor(
                    self.dim
                    * math.log(self.original_length / (self.beta_fast * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                0,
            )
            high = min(
                math.ceil(
                    self.dim
                    * math.log(self.original_length / (self.beta_slow * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                self.dim - 1,
            )
            ramp = (
                (torch.arange(self.dim // 2, device=x.device, dtype=torch.float32) - low) / max(high - low, 1e-3)
            ).clamp(0, 1)
            frequencies = frequencies / self.factor * ramp + frequencies * (1 - ramp)
        angles = position_ids.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * frequencies
        phases = torch.polar(torch.ones_like(angles), angles) * self.attention_scaling
        return torch.cat((phases.real, phases.real), -1), torch.cat((phases.imag, phases.imag), -1)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    """Rotate the final channels using the reference's complex multiplication.

    Args:
        x: Tensor of shape [batch, heads, sequence, channels] or
            [batch, sequence, channels].
        cos: FP32 tensor of shape [batch, sequence, rope_head_dim], with unique
            pair frequencies in its first half.
        sin: FP32 tensor with the same layout as cos; negate for inverse RoPE.
        rope_head_dim: Even number of final channels rotated as adjacent pairs.

    Returns:
        Independent tensor with x's shape and dtype, preserving its other channels.
    """
    pairs = torch.view_as_complex(x[..., -rope_head_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    phases = torch.complex(cos[..., : rope_head_dim // 2], sin[..., : rope_head_dim // 2])
    if x.ndim == 4:
        phases = phases.unsqueeze(1)
    rotated = torch.view_as_real(pairs * phases).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-rope_head_dim], rotated), dim=-1)


# ---------------------------------------------------------------------------
# Quantization-aware fake quantization (straight-through estimator).
# ---------------------------------------------------------------------------

_FP8_E4M3_MAX = 448.0
_FP8_AMAX_FLOOR = 1e-4
_FP4_E2M1_MAX = 6.0
_FP4_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class _StraightThrough(torch.autograd.Function):
    """Return the quantized value in forward and pass the gradient through unchanged."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _pow2_ceil_scale(amax: torch.Tensor, max_value: float) -> torch.Tensor:
    """Compute MX scale exponents without logarithm rounding at bin boundaries.

    Args:
        amax: Positive FP32 maxima of shape [..., groups, 1], with arbitrary
            leading dimensions. Callers clamp to the format's minimum scale.
        max_value: Maximum representable magnitude in the quantized format.

    Returns:
        FP32 power-of-two scales with the same shape as amax, matching the
        reference's reciprocal multiplication and IEEE754 mantissa test.
    """
    bits = (amax * (1 / max_value)).contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    increment = (bits & ((1 << 23) - 1)) != 0
    return ((exponent + increment.to(torch.int32)) << 23).view(torch.float32)


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round ``|x| <= 6`` to the nearest FP4 E2M1 value, ties to even (hardware round-to-nearest-even).

    Even grid indices (``0, 1, 2, 4``) carry an even mantissa, so a value exactly
    halfway between two grid points resolves to the even-indexed neighbour.

    Args:
        x: FP32 tensor of arbitrary shape with magnitudes at most 6.

    Returns:
        Tensor of the same shape and dtype, preserving the sign of zero.
    """
    grid = x.new_tensor(_FP4_E2M1_GRID)
    midpoints = (grid[:-1] + grid[1:]) / 2
    magnitude = x.abs()
    idx = torch.bucketize(magnitude, midpoints, right=False)  # ties -> lower grid point
    tie_to_upper = (
        (idx % 2 == 1) & (idx < midpoints.numel()) & (magnitude == midpoints[idx.clamp(max=midpoints.numel() - 1)])
    )
    idx = idx + tie_to_upper.to(idx.dtype)
    return torch.copysign(grid[idx], x)


def fake_quant_fp8(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block-wise FP8 E4M3 quantize-dequantize with a power-of-two scale per ``block_size`` channels.

    Matches the released ``act_quant(..., scale_fmt="ue8m0", inplace=True)`` used on the
    sliding-window KV cache. A partial final channel group is zero-padded before
    quantization and cropped back to the input width afterwards.
    """
    if block_size <= 0:
        raise ValueError("Cache quantization requires a positive block_size")
    channels = x.shape[-1]
    with torch.no_grad():
        blocks = F.pad(x.detach().float(), (0, -channels % block_size)).unflatten(-1, (-1, block_size))
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(_FP8_AMAX_FLOOR)
        scale = _pow2_ceil_scale(amax, _FP8_E4M3_MAX)
        q = (blocks / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
        q = q.flatten(-2)[..., :channels].to(x.dtype)
    return _StraightThrough.apply(x, q)


def fake_quant_fp4(x: torch.Tensor, block_size: int, scale_format: str) -> torch.Tensor:
    """Block-wise FP4 E2M1 quantize-dequantize.

    ``scale_format="e8m0"`` uses a power-of-two scale (indexer Q/K, 32-channel
    blocks); ``"e4m3"`` uses an E4M3 scale (compressed KV, 16-channel blocks),
    following NVFP4 without the second-level global scale. A partial final
    channel group is zero-padded before quantization and cropped afterwards.
    """
    if scale_format not in ("e8m0", "e4m3"):
        raise ValueError(f"Unknown FP4 scale format: {scale_format}")
    if block_size <= 0:
        raise ValueError("Cache quantization requires a positive block_size")
    channels = x.shape[-1]
    with torch.no_grad():
        blocks = F.pad(x.detach().float(), (0, -channels % block_size)).unflatten(-1, (-1, block_size))
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        if scale_format == "e4m3":
            # Training's compressed KV keeps even an all-zero group's scale nonzero.
            amax = amax.clamp_min(_FP4_E2M1_MAX * 2.0**-9)
            scale = (amax / _FP4_E2M1_MAX).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp_min(_FP4_E2M1_MAX * 2.0**-126)
            scale = _pow2_ceil_scale(amax, _FP4_E2M1_MAX)
        q = _round_to_e2m1((blocks / scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX)) * scale
        q = q.flatten(-2)[..., :channels].to(x.dtype)
    return _StraightThrough.apply(x, q)


# ---------------------------------------------------------------------------
# Hyper-connection stream helpers (reference ``Block.hc_pre`` / ``hc_post``).
# ---------------------------------------------------------------------------


class DeepseekV41HyperConnection(DeepseekV4HyperConnection):
    """Reuse V4 parameter ownership with the V4.1 reference's FP32 operation order."""

    def compute_weights(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project streams before RMS scaling and apply the fused affine map.

        Args:
            hidden_streams: Tensor of shape [batch, sequence, streams, hidden].

        Returns:
            FP32 pre/post tensors of shape [batch, sequence, streams] and
            combination coefficients [batch, sequence, input_streams, output_streams].
        """
        flat = hidden_streams.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        streams = self.hc_mult
        scales = torch.cat(
            (self.scale[0].expand(streams), self.scale[1].expand(streams), self.scale[2].expand(streams * streams))
        ).float()
        logits = torch.addcmul(self.base.float(), mixes, scales)
        pre = torch.sigmoid(logits[..., :streams]) + self.hc_eps
        post = 2 * torch.sigmoid(logits[..., streams : 2 * streams])
        comb = dsv4_sinkhorn_normalize(
            logits[..., 2 * streams :].unflatten(-1, (streams, streams)),
            backend=self.sinkhorn_backend,
            repeat=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        return pre, post, comb


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Initial one-hot input mix that reads stream 0.  ``x``: ``[B, S, ...]``."""
    pre_mix = x.new_zeros(x.shape[0], x.shape[1], hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


def hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the hc copies into one sublayer input. ``[B,S,hc,d] x [B,S,hc] -> [B,S,d]``."""
    return torch.sum(pre_mix.float().unsqueeze(-1) * x.float(), dim=2).to(x.dtype)


def hc_expand(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Expand a sublayer output back to hc copies and mix the residual in through ``comb``.

    Args:
        y: Tensor of shape [batch, sequence, hidden].
        residual: Tensor of shape [batch, sequence, streams, hidden].
        post: FP32 tensor of shape [batch, sequence, streams].
        comb: FP32 tensor of shape [batch, sequence, input_streams, output_streams].

    Returns:
        Tensor of shape [batch, sequence, streams, hidden], in y's dtype.
        Products reduce over input streams in the reference's operation order.
    """
    mixed = post.float().unsqueeze(-1) * y.float().unsqueeze(-2)
    mixed = mixed + (comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(2)
    return mixed.to(y.dtype)


# ---------------------------------------------------------------------------
# Sparse-attention index construction.
# ---------------------------------------------------------------------------


def build_window_topk_indices(seq_ids: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sliding-window key indices per query, ``-1`` where a slot holds nothing.

    Args:
        seq_ids: Binary valid-token mask ``[B, S]``, with zero marking right padding.
        window_size: Number of keys (including the query itself) in the window.

    Returns:
        ``[B, S, min(S, window_size)]`` long tensor of key positions.
    """
    batch, seq_len = seq_ids.shape
    device = seq_ids.device
    width = min(seq_len, window_size)
    q_idx = torch.arange(seq_len, device=device)
    k_idx = (q_idx - window_size + 1).clamp(min=0).unsqueeze(-1) + torch.arange(width, device=device)
    k_idx = torch.where(k_idx > q_idx.unsqueeze(-1), torch.full_like(k_idx, -1), k_idx)  # [S, W]
    k_idx = k_idx.unsqueeze(0).expand(batch, -1, -1)
    safe = k_idx.clamp(min=0)
    key_seq = torch.gather(seq_ids, 1, safe.reshape(batch, -1)).view(batch, seq_len, width)
    valid = (k_idx >= 0) & key_seq.bool()
    return torch.where(valid, k_idx, torch.full_like(k_idx, -1))


def build_compressed_visibility(
    q_positions: torch.Tensor,
    q_seq_ids: torch.Tensor,
    pool_seq_ids: torch.Tensor,
    pool_positions: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Apply the reference's absolute compressed-position visibility.

    Group ``j`` is visible after its last token: ``j < (i + 1) // ratio``.
    ``pool_seq_ids`` stores whether every token of that group is valid. Query
    padding does not change indexer scores; attention masks its final output.
    ``q_seq_ids`` remains accepted for the existing helper interface.

    Returns:
        ``[B, S, P]`` bool mask.
    """
    threshold = ((q_positions + 1) // compress_ratio).unsqueeze(-1)
    allowed = pool_positions.unsqueeze(1) < threshold
    return allowed & pool_seq_ids.bool().unsqueeze(1)


def select_candidate_blocks(
    scores: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Keep high-scoring absolute blocks and the latest causally visible block.

    Args:
        scores: Tensor of shape [batch, sequence, pooled], with unreachable entries at -inf.
        visible_lengths: Integer tensor [batch, sequence, 1], counting causally
            visible compressed positions independently of padding.
        topk_blocks: Maximum number of candidate blocks per query.
        block_size: Number of pooled positions in each block.

    Returns:
        Boolean tensor of shape [batch, sequence, pooled] selecting candidate positions.
    """
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last_block = (visible_lengths - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last_block, torch.inf)
    selected = block_scores.topk(min(topk_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter(-1, selected.indices, selected.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# ---------------------------------------------------------------------------
# Cross-layer shared state (reference ``SharedAttentionRuntime``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeepseekV41SharedState:
    """CSA2 state owned by one full-sequence model forward.

    ``compress_kv`` / ``index_k`` come from KV source layers, ``topk_idxs`` from
    index source layers and ``candidates`` from the candidate source layer.
    Tensor fields retain autograd history and are never modified by consumers.
    A model creates an empty state for every forward, including every microbatch.
    """

    compress_kv: torch.Tensor | None = None  # [B, P, head_dim], post-RoPE (fake-quantized) latent
    index_k: torch.Tensor | None = None  # [B, P, index_head_dim]
    pool_seq_ids: torch.Tensor | None = None  # [B, P] complete valid-group flags (0 = invalid)
    pool_positions: torch.Tensor | None = None  # [B, P] absolute group index, including invalid groups
    allowed: torch.Tensor | None = None  # [B, S, P] bool visibility of pooled positions
    topk_idxs: torch.Tensor | None = None  # [B, S, K] pooled positions, -1 = none
    candidates: torch.Tensor | None = None  # [B, S, P] bool
    compress_ratio: int = 0
    window_topk_idxs: torch.Tensor | None = None  # [B, S, W] sliding-window key positions, shared by all layers


@dataclass(frozen=True)
class DeepseekV41AttentionOutput:
    """Attention result and the shared state for the next layer.

    Attributes:
        hidden_states: Tensor of shape [batch, sequence, hidden].
        state: Shared tensor layouts documented by DeepseekV41SharedState.
    """

    hidden_states: torch.Tensor
    state: DeepseekV41SharedState


# ---------------------------------------------------------------------------
# Compressor and indexer.
# ---------------------------------------------------------------------------


class _InputDtypeLinear(nn.Linear):
    """Keep compressor projection compute in the incoming activation dtype.

    FSDP may preserve FP32 weight storage. Select the compute dtype after this
    module's own pre-forward hook has made that weight available.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project without changing the registered weight's storage dtype.

        Args:
            x: Tensor of shape [..., in_features], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., out_features], with x's dtype.
        """
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class DeepseekV41Compressor(nn.Module):
    """Pool ``compress_ratio`` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent *before* RoPE; the indexer needs the unrotated form and
    the attention rotates afterwards. Projection weights stay in FP32 storage.
    ``compress_ratio == 1`` computes in the activation dtype; ratios above 1
    project and pool in FP32, matching the released checkpoint.
    """

    def __init__(self, config: DeepseekV41Config, compress_ratio: int):
        super().__init__()
        if compress_ratio < 1:
            raise ValueError(f"DeepseekV41Compressor needs compress_ratio >= 1, got {compress_ratio}")
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(config.head_dim)
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.wkv = _InputDtypeLinear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = (
            nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
            if self.compress_ratio > 1
            else None
        )
        self.norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, S, hidden] -> [B, S // ratio, head_dim]`` (a trailing partial group is dropped)."""
        ratio = self.compress_ratio
        if ratio == 1:
            # ``wkv`` may be promoted to fp32 by the model's keep-in-fp32 policy (the
            # ratio-agnostic ``self_attn.compressor.wkv`` entry) and, under FSDP2, the
            # sharded parameter dtype can differ from the unsharded compute dtype.  Read
            # the parameter inside forward (unsharded) and project in the activation
            # dtype, which is what the reference does for ratio-1 compressors.
            latent = self.wkv(hidden_states).to(hidden_states.dtype)
            return self.norm(latent)
        x = hidden_states.float()
        usable = (x.shape[1] // ratio) * ratio
        kv = self.wkv(x[:, :usable]).unflatten(1, (-1, ratio))
        score = self.wgate(x[:, :usable]).unflatten(1, (-1, ratio))
        pooled = (kv * score.softmax(dim=2)).sum(dim=2)
        return self.norm(pooled.to(hidden_states.dtype))

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.wkv.weight, mean=0.0, std=init_std)
        if self.wgate is not None:
            nn.init.normal_(self.wgate.weight, mean=0.0, std=init_std)
        self.norm.reset_parameters()


class DeepseekV41Indexer(nn.Module):
    """Keep the ``index_topk`` best compressed positions per query.

    A small side attention: FP4 query heads against one shared key per compressed
    position, scores rectified then combined by ``weights_proj``.  Only a KV
    source layer owns ``wk`` / ``k_norm`` (the keys are projected from its
    latent); Reindex layers read the published keys.  With a candidate source
    this is the second of two levels (:func:`select_candidate_blocks` is the first).
    """

    def __init__(self, config: DeepseekV41Config, layer_idx: int, backend: BackendConfig | None = None):
        super().__init__()
        self.backend = backend or BackendConfig()
        self.layer_idx = layer_idx
        self.owns_k = config.is_kv_source(layer_idx)
        self.is_candidate_source = layer_idx == int(config.candidate_source_layer_id)
        self.uses_candidates = 0 <= int(config.candidate_source_layer_id) < layer_idx
        self.candidate_topk_blocks = int(config.candidate_topk_blocks)
        self.candidate_block_size = int(config.candidate_block_size)
        self.n_heads = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.index_topk = int(config.index_topk)
        self.softmax_scale = self.head_dim**-0.5
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=model_dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False, dtype=model_dtype)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False, dtype=model_dtype)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype)
        else:
            self.wk = None
            self.k_norm = None
        # The backbone loss consumes only discrete selected positions. Auxiliary
        # indexer training is outside this model's supported loss path.
        self.requires_grad_(False)

    def build_keys(self, latent: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Turn the KV source's pre-RoPE latent ``[B, P, head_dim]`` into index keys ``[B, P, index_head_dim]``."""
        if self.wk is None:
            raise RuntimeError(f"Indexer of layer {self.layer_idx} does not own index keys")
        k = self.k_norm(self.wk(latent))
        k = _apply_partial_rope(k, cos, sin, self.rope_head_dim)
        k = fake_quant_fp4(k, 32, "e8m0")
        return k

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        q_residual: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        state: DeepseekV41SharedState | None = None,
        *,
        latent: torch.Tensor | None = None,
        cos_p: torch.Tensor | None = None,
        sin_p: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Score the shared index keys, or (with ``latent``) publish them.

        Both entry points go through ``__call__`` so FSDP2 unshards this module's
        parameters even when it is wrapped as its own unit; callers must not reach
        into ``wk`` / ``k_norm`` directly.

        Key mode (``latent`` given): ``latent`` is the KV source's pre-RoPE latent
        ``[B, P, head_dim]`` and ``cos_p`` / ``sin_p`` are its RoPE tables
        ``[B, P, qk_rope_head_dim]``; returns index keys ``[B, P, index_head_dim]``.

        Score mode: returns ``[B, S, K]`` pooled positions per query (``-1`` for empty
        slots) together with the ``[B, S, P]`` candidate pool (built here for the
        candidate source layer, otherwise passed through).  Results are returned rather
        than written to ``state`` because FSDP2 may hand this module a copy of it.

        Args:
            hidden_states: ``[B, S, hidden]`` attention input (feeds ``weights_proj``).
            q_residual: ``[B, S, q_lora_rank]`` normalized query latent.
            cos, sin: query RoPE tables ``[B, S, qk_rope_head_dim]``.
            state: Shared state providing ``index_k`` (``[B, P, index_head_dim]``),
                ``allowed`` (``[B, S, P]`` bool visibility) and the candidate pool.
            latent: Optional pre-RoPE tensor of shape [batch, pooled, head_dim]
                selecting key construction instead of scoring.
            cos_p: FP32 pooled cosine table [batch, pooled, qk_rope_head_dim].
            sin_p: FP32 pooled sine table [batch, pooled, qk_rope_head_dim].

        Returns:
            Key mode returns [batch, pooled, index_head_dim]. Score mode returns
            integer positions [batch, sequence, topk], ordered by position with
            invalid slots set to -1, and an optional boolean candidate mask
            [batch, sequence, pooled]. No output carries an autograd history.
        """
        if latent is not None:
            return self.build_keys(latent, cos_p, sin_p)
        if state is None or state.index_k is None or state.allowed is None:
            raise RuntimeError(f"Indexer of layer {self.layer_idx} found no published index keys")
        index_k, allowed = state.index_k, state.allowed
        batch, seq_len, _ = hidden_states.shape
        if index_k.shape[1] == 0:
            candidates = (
                torch.empty(batch, seq_len, 0, dtype=torch.bool, device=hidden_states.device)
                if self.is_candidate_source
                else state.candidates
            )
            return torch.empty(batch, seq_len, 0, dtype=torch.long, device=hidden_states.device), candidates
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)  # [B, S, H, D]
        q = fake_quant_fp4(q, 32, "e8m0")
        weights = self.weights_proj(hidden_states) * (self.softmax_scale * self.n_heads**-0.5)
        # Preserve BF16 projection, score and weighted-reduction boundaries from
        # the reference; promoting these operations can change discrete top-k.
        scores = torch.einsum("bshd,btd->bsht", q, index_k)
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim=2)
        scores = scores.masked_fill(~allowed, float("-inf"))

        candidates = state.candidates
        if self.is_candidate_source:
            lengths = torch.arange(1, seq_len + 1, device=hidden_states.device) // state.compress_ratio
            visible_lengths = lengths.view(1, seq_len, 1).expand(batch, -1, -1)
            candidates = select_candidate_blocks(
                scores,
                visible_lengths,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            if candidates is None or candidates.shape != scores.shape:
                raise RuntimeError(f"Indexer of layer {self.layer_idx} expects a candidate pool from an earlier layer")
            scores = scores.masked_fill(~candidates, float("-inf"))

        topk = min(self.index_topk, scores.shape[-1])
        if topk == 0:
            return scores.new_empty(batch, seq_len, 0, dtype=torch.long), candidates
        indices = scores.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        valid = torch.isfinite(scores.gather(-1, indices))
        return torch.where(valid, indices, torch.full_like(indices, -1)), candidates

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.wq_b.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.weights_proj.weight, mean=0.0, std=init_std)
        if self.wk is not None:
            nn.init.normal_(self.wk.weight, mean=0.0, std=init_std)
            self.k_norm.reset_parameters()


# ---------------------------------------------------------------------------
# Attention.
# ---------------------------------------------------------------------------


class DeepseekV41Attention(nn.Module):
    """Latent attention over two KV sources at once: a sliding window of raw KV plus,
    when ``compress_ratio > 0``, ``index_topk`` compressed positions reaching further back.

    ``compress_ratio > 0`` does not mean the layer compresses its own KV: only
    KV source layers do, the rest read the shared cache (CSA2 Reindex / Reuse).
    Q and the output projection are both low-rank, the latter grouped.
    """

    def __init__(self, config: DeepseekV41Config, layer_idx: int, backend: BackendConfig | None = None):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        if self.backend.attn not in ("eager", "sdpa", "tilelang"):
            raise ValueError("DeepSeek V4.1 attention supports backend.attn='eager', 'sdpa', or 'tilelang'")
        if self.backend.attn == "tilelang" and config.attention_dropout:
            raise ValueError("The TileLang sparse attention backend requires attention_dropout=0")
        if self.backend.linear != "torch" or self.backend.rms_norm != "torch_fp32":
            raise ValueError("DeepSeek V4.1 attention requires torch linear layers and torch_fp32 RMSNorm")
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratio(layer_idx)
        self.is_kv_source = config.is_kv_source(layer_idx)
        self.is_index_source = config.is_index_source(layer_idx)
        if (self.is_kv_source or self.is_index_source) and self.compress_ratio == 0:
            raise ValueError(f"layer {layer_idx} is a CSA2 source but has compress_ratio 0")
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self.sliding_window = int(config.sliding_window)
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False, dtype=model_dtype)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps, dtype=model_dtype)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=model_dtype)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=model_dtype)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=model_dtype)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        ).to(dtype=model_dtype)
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False, dtype=model_dtype)
        self.sinks_param = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.compressor = DeepseekV41Compressor(config, self.compress_ratio) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx, backend=self.backend) if self.is_index_source else None

    @property
    def sinks(self) -> torch.Tensor:
        return self.sinks_param()

    def _publish_compressed_kv(
        self,
        hidden_states: torch.Tensor,
        rotary_compress: nn.Module,
        position_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        state: DeepseekV41SharedState,
    ) -> DeepseekV41SharedState:
        """Publish consecutive absolute compression groups without rewriting padded latents.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            rotary_compress: Rotary module accepting latents and absolute positions.
            position_ids: Contiguous zero-based positions [batch, sequence] or [1, sequence].
            seq_ids: Binary valid-token mask [batch, sequence], with zero for right padding.
            state: Shared attention tensors, with layouts documented by DeepseekV41SharedState.

        Returns:
            New state containing this source's compressed tensors. The input state
            and its tensors are never modified.
        """
        ratio = self.compress_ratio
        batch, seq_len, _ = hidden_states.shape
        n_pooled = seq_len // ratio
        latent = self.compressor(hidden_states)  # [B, P, head_dim], pre-RoPE
        pool_seq_ids = seq_ids[:, : n_pooled * ratio].bool().unflatten(1, (n_pooled, ratio)).all(dim=-1)
        pool_positions = torch.arange(n_pooled, device=hidden_states.device).view(1, -1).expand(batch, -1)
        cos_p, sin_p = rotary_compress(latent, position_ids[:, : n_pooled * ratio : ratio])
        index_k = None
        if self.indexer is not None and self.indexer.owns_k:
            index_k = self.indexer(latent=latent, cos_p=cos_p, sin_p=sin_p)
        latent = _apply_partial_rope(latent, cos_p, sin_p, self.rope_head_dim)
        # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
        latent = fake_quant_fp4(latent, 16, "e4m3")
        return replace(
            state,
            compress_kv=latent,
            index_k=index_k,
            pool_seq_ids=pool_seq_ids,
            pool_positions=pool_positions,
            compress_ratio=ratio,
            allowed=build_compressed_visibility(position_ids, seq_ids, pool_seq_ids, pool_positions, ratio),
            topk_idxs=None,
            candidates=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor],
        rotary_compress: nn.Module,
        position_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        state: DeepseekV41SharedState,
        attention_mask: torch.Tensor | None = None,
    ) -> DeepseekV41AttentionOutput:
        """Run one attention layer.

        Args:
            hidden_states: ``[B, S, hidden]`` (already collapsed and normalized).
            position_embeddings: main RoPE ``(cos, sin)`` for pure sliding-window layers.
            position_embeddings_compress: compress RoPE (YaRN) ``(cos, sin)`` for CSA2 layers.
            rotary_compress: compress rotary module, used to rotate pooled latents.
            position_ids: Contiguous zero-based positions ``[B, S]`` or ``[1, S]``.
            seq_ids: Binary valid-token mask ``[B, S]``, used when attention_mask is omitted.
            state: cross-layer shared state.
            attention_mask: Optional original binary right-padding mask ``[B, S]``.

        Returns:
            Output with hidden_states [batch, sequence, hidden] and the new state,
            whose tensors use the layouts in DeepseekV41SharedState. Input tensors
            and the input state are never mutated.
        """
        batch, seq_len, _ = hidden_states.shape
        if seq_len == 0:
            raise ValueError("DeepSeek V4.1 attention requires a nonempty full sequence")
        positions = torch.arange(seq_len, device=hidden_states.device)
        if position_ids.shape not in ((1, seq_len), (batch, seq_len)) or not torch.equal(
            position_ids, positions.expand_as(position_ids)
        ):
            raise ValueError("DeepSeek V4.1 attention supports only contiguous zero-based full-sequence position_ids")
        if attention_mask is None:
            attention_mask = seq_ids
        if attention_mask.shape != (batch, seq_len):
            raise ValueError("DeepSeek V4.1 attention_mask must have shape [batch, sequence]")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("DeepSeek V4.1 attention_mask must contain only zero and one")
        valid_tokens = attention_mask.bool()
        if torch.any(valid_tokens[:, 1:] & ~valid_tokens[:, :-1]):
            raise ValueError("DeepSeek V4.1 compression supports right padding only")
        # Compress-ratio layers rotate Q/KV with the compress rope (theta=160000 + YaRN),
        # pure sliding-window layers with the base rope (reference ``Attention.__init__``).
        cos, sin = position_embeddings_compress if self.compress_ratio else position_embeddings

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2).contiguous()  # [B, S, H, D]

        kv = self.kv_norm(self.wkv(hidden_states))  # [B, S, D]
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        kv = fake_quant_fp8(kv, 32)

        keys = kv
        next_state = state
        topk_idxs = build_window_topk_indices(valid_tokens, self.sliding_window)
        if self.compress_ratio:
            width = seq_len // self.compress_ratio
            if self.is_kv_source:
                next_state = self._publish_compressed_kv(
                    hidden_states, rotary_compress, position_ids, valid_tokens, next_state
                )
            elif next_state.compress_kv is None or next_state.compress_ratio != self.compress_ratio:
                raise ValueError("A CSA2 consumer requires a preceding Full layer with the same compression ratio")
            if self.is_index_source:
                selected, candidates = self.indexer(hidden_states, q_residual, cos, sin, next_state)
                next_state = replace(next_state, topk_idxs=selected, candidates=candidates)
            if next_state.topk_idxs is None or next_state.compress_kv is None:
                raise ValueError("A Reuse CSA2 layer requires compressed KV and indices from its source")
            if next_state.compress_kv.shape[:2] != (batch, width) or next_state.topk_idxs.shape[:2] != (batch, seq_len):
                raise ValueError("CSA2 state belongs to a different batch or sequence")
            compressed_idxs = torch.where(
                next_state.topk_idxs >= 0, next_state.topk_idxs + seq_len, torch.full_like(next_state.topk_idxs, -1)
            )
            keys = torch.cat([kv, next_state.compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compressed_idxs], dim=-1)

        if self.backend.attn == "tilelang":
            topk_idxs = topk_idxs.masked_fill(~valid_tokens.unsqueeze(-1), -1)
            attn_output = dsv4_sparse_attention(
                q,
                keys.contiguous(),
                self.sinks_param(q),
                topk_idxs,
                self.scaling,
                backend="tilelang",
                reference_rounding=True,
            )  # [B, S, H, D]
        else:
            # Map invalid sparse slots to a separate sentinel column so they
            # cannot overwrite visibility for a real key, including key zero.
            key_len = keys.shape[1]
            allowed = torch.zeros(batch, seq_len, key_len + 1, dtype=torch.bool, device=hidden_states.device)
            valid = (topk_idxs >= 0) & (topk_idxs < key_len)
            indices = torch.where(valid, topk_idxs, key_len).long()
            allowed = allowed.scatter(-1, indices, True)[..., :key_len]

            # The zero-valued extra key contributes exp(sink) only to the
            # denominator, preserving the original dense attention arithmetic.
            keys = torch.cat((keys, keys.new_zeros(batch, 1, self.head_dim)), dim=1)
            bias = torch.zeros(batch, 1, seq_len, key_len, device=hidden_states.device, dtype=torch.float32)
            bias = bias.masked_fill(~allowed.unsqueeze(1), -torch.inf).expand(-1, self.num_heads, -1, -1)
            sink = self.sinks_param(q).view(1, self.num_heads, 1, 1).expand(batch, -1, seq_len, -1)
            bias = torch.cat((bias, sink), dim=-1)
            if self.backend.attn == "sdpa":
                attn_output = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    keys.unsqueeze(1),
                    keys.unsqueeze(1),
                    attn_mask=bias,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=self.scaling,
                ).transpose(1, 2)
            else:
                logits = torch.einsum("bshd,btd->bhst", q.float(), keys.float()) * self.scaling
                probabilities = (logits + bias).softmax(dim=-1)
                probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
                attn_output = torch.einsum("bhst,btd->bshd", probabilities, keys.float()).to(q.dtype)

        # Undo the query rotation on the output so the cache can stay in one shared rotated form.
        attn_output = _apply_partial_rope(attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim).transpose(1, 2)
        grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)
        output = self.wo_b(self.wo_a(grouped).flatten(2))
        return DeepseekV41AttentionOutput(output.masked_fill(~valid_tokens.unsqueeze(-1), 0), next_state)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, DeepseekV41RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.zeros_(self.sinks_param.weight)


# ---------------------------------------------------------------------------
# Transformer block with single-pass mHC.
# ---------------------------------------------------------------------------


class DeepseekV41Block(nn.Module):
    """One transformer block whose residual stream is ``hc_mult`` parallel copies.

    Single-pass mHC (reference ``Block.forward``): the attention site collapses
    the streams with ``pre_mix`` produced by the *previous* block's FFN site, and
    the FFN site uses the mix produced by this block's attention site.  The block
    returns the FFN-site mix for the next block.  An optional Engram module
    writes into the streams before the attention site.
    """

    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV41Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        engram_layout: EngramLayout | None = None,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = int(config.hc_mult)
        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.self_attn = DeepseekV41Attention(config, layer_idx, backend=backend)
        moe_backend = replace(backend, gate_precision=torch.float32)
        self.mlp = MoE(moe_config, moe_backend)
        self.mlp.gate = DeepseekV4VisionGate(
            DeepseekV4Config(vocab_size=config.vocab_size),
            moe_config,
            gate_precision=moe_backend.gate_precision,
            hash_routing=False,
        )
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype)
        self.post_attention_layernorm = DeepseekV41RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )
        hc_kwargs = dict(
            hc_mult=self.hc_mult,
            hidden_size=config.hidden_size,
            hc_sinkhorn_iters=int(config.hc_sinkhorn_iters),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
            sinkhorn_backend="tilelang" if backend.attn == "tilelang" else "torch",
        )
        self.attn_hc = DeepseekV41HyperConnection(**hc_kwargs)
        self.ffn_hc = DeepseekV41HyperConnection(**hc_kwargs)
        self.engram = (
            DeepseekV41Engram(config, layer_idx, engram_layout, backend, engram_process_group=engram_process_group)
            if engram_layout is not None and layer_idx in engram_layout.layer_ids
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        engram_hash_ids: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        state: DeepseekV41SharedState,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepseekV41SharedState]:
        """Transform the HC streams.

        Args:
            x: ``[B, S, hc_mult, hidden]`` residual streams.
            pre_mix: ``[B, S, hc_mult]`` fp32 input mix from the previous block.
            padding_mask: ``[B, S]`` bool, ``True`` at padding (MoE token mask).
            engram_hash_ids: ``[B, S, n_hash_cols]`` hash ids for this layer's Engram.
            engram_mask: ``[B, S]`` bool, ``False`` where Engram must not write.
            vision_token_types: Optional integer tensor [batch, sequence],
                with -1 for text and nonnegative image-token types.
            state: Shared tensors with layouts documented in DeepseekV41SharedState.
                The input state and its tensors retain their history and are never mutated.

        Returns:
            Updated streams of shape [batch, sequence, hc_mult, hidden], FP32 mix
            of shape [batch, sequence, hc_mult], and the updated shared state.
            State tensor layouts are documented in DeepseekV41SharedState.
        """
        if pre_mix.dtype != torch.float32:
            raise TypeError(
                "Single-pass mHC requires FP32 carried coefficients. Configure the FSDP mixed precision policy "
                "with cast_forward_inputs=False and output_dtype=None."
            )
        if self.engram is not None:
            if engram_hash_ids is None:
                raise ValueError(f"layer {self.layer_idx} has an Engram module but received no hash ids")
            x = self.engram(x, engram_hash_ids, engram_mask)

        attn_pre, attn_post, attn_comb = self.attn_hc(x)
        attn_out = self.self_attn(self.input_layernorm(hc_collapse(x, pre_mix)), state=state, **attn_kwargs)
        x = hc_expand(attn_out.hidden_states, x, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(x)
        self.mlp.gate.set_routing_context(None, vision_token_types)
        mlp_out = self.mlp(self.post_attention_layernorm(hc_collapse(x, attn_pre)), padding_mask)
        x = hc_expand(mlp_out, x, ffn_post, ffn_comb)
        return x, ffn_pre, attn_out.state

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=init_std)
        self.mlp.init_weights(buffer_device, init_std=init_std)
        self.mlp.gate.init_dsv4_weights()
        self.attn_hc.init_weights(init_std)
        self.ffn_hc.init_weights(init_std)
        if self.engram is not None:
            self.engram.init_weights()
