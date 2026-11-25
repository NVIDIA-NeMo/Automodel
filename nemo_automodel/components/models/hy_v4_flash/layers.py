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

"""Norms, hyper-connections and interleaved multimodal RoPE of Hy4-flash.

Semantics follow the transformers reference (PR 50903, ``modeling_hy_v4_flash.py``):

* ``HY4FlashRMSNorm`` casts the normalized activations back to the input dtype *before* the weight multiply.
* The hyper-connection (iHC) has no Sinkhorn mixing: a sigmoid ``pre`` weight collapses the ``hc_mult`` streams
  and a sigmoid ``post`` weight scatters the sub-layer output back, the old streams passing through unchanged.
* RoPE rotates interleaved pairs of the trailing ``qk_rope_head_dim`` channels, with the time/height/width
  planes of M-RoPE interleaved in the frequency axis (Qwen3-VL style).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashTextConfig


# ----------------------------------------------------------------------------------------- fp32 cores
# The RMSNorm and the two hyper-connection steps each run several full-size fp32 passes over the activations in
# eager mode (the streams are ``hc_mult`` times the hidden width). The arithmetic lives in the plain functions below
# so ``torch.compile`` can fuse each into one or two kernels: ``BackendConfig.compile_norm`` wraps the RMSNorm core
# and ``BackendConfig.compile_hc`` the hyper-connection cores, with the same lazy once-per-process pattern and static
# shapes as DeepSeek-V4.1's cores. The modules call the module-level dispatchers, so instances built before the hook
# ran still pick up the compiled core. Compiled numerics are allclose to eager, not bitwise-identical.
def _rms_norm_core(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last axis in fp32, cast back, then apply the weight (reference order).

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden].
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of the shape and dtype of ``hidden_states``.
    """
    x = hidden_states.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return weight * x.to(hidden_states.dtype)


def _hc_collapse_core(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    hc_magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the residual streams with sigmoid ``pre`` weights and produce the ``post`` scatter weights.

    Args:
        streams: Tensor of shape [batch, sequence, hc_mult, hidden].
        fn: fp32 tensor of shape [2 * hc_mult, hc_mult * hidden] (``pre`` rows first, then ``post`` rows).
        base: fp32 tensor of shape [2 * hc_mult].
        scale: fp32 tensor of shape [2] (``pre`` scale, ``post`` scale).
        norm_eps: Epsilon of the unweighted RMS of the flattened streams.
        hc_eps: Offset added to both weight groups.
        hc_magnitude: Range of the ``post`` weights.

    Returns:
        ``post``: fp32 tensor of shape [batch, sequence, hc_mult]; ``collapsed``: tensor of shape
        [batch, sequence, hidden] in the dtype of ``streams``.
    """
    hc_mult = streams.shape[2]
    wide = streams.float()
    flat = wide.flatten(2)
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre_logits, post_logits = mixes.split(hc_mult, dim=-1)
    pre = torch.sigmoid(pre_logits * scale[0] + base[:hc_mult]) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_logits * scale[1] + base[hc_mult:]) + hc_eps
    return post, (pre.unsqueeze(-1) * wide).sum(dim=2).to(streams.dtype)


def _hc_expand_core(post: torch.Tensor, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Scatter a sub-layer output onto the streams: ``post * output + residual`` in fp32.

    Args:
        post: fp32 tensor of shape [batch, sequence, hc_mult].
        output: Tensor of shape [batch, sequence, hidden].
        residual: Tensor of shape [batch, sequence, hc_mult, hidden]; its dtype is the result dtype.

    Returns:
        Tensor of shape [batch, sequence, hc_mult, hidden] in the dtype of ``residual``.
    """
    return (post.unsqueeze(-1) * output.float().unsqueeze(-2) + residual.float()).to(residual.dtype)


_rms_norm = _rms_norm_core
_hc_collapse = _hc_collapse_core
_hc_expand = _hc_expand_core
_NORM_COMPILED = False
_HC_COMPILED = False


def compile_norm_core() -> None:
    """Wrap the fp32 RMSNorm core with ``torch.compile`` (``BackendConfig.compile_norm``); runs once per process.

    Static shapes (one specialization per norm site shape and grad mode) and no ``fullgraph``: past dynamo's
    recompile limit a shape runs eager instead of raising.
    """
    global _rms_norm, _NORM_COMPILED
    if _NORM_COMPILED:
        return
    _rms_norm = torch.compile(_rms_norm_core, dynamic=False)
    _NORM_COMPILED = True


def compile_hc_cores() -> None:
    """Wrap the hyper-connection collapse and expand cores with ``torch.compile`` (``BackendConfig.compile_hc``).

    Runs once per process, static shapes: a new sequence length compiles the two cores again, so the knob is meant
    for fixed-length batches (padded or packed). No ``fullgraph``: past the recompile limit a core runs eager.
    """
    global _hc_collapse, _hc_expand, _HC_COMPILED
    if _HC_COMPILED:
        return
    _hc_collapse = torch.compile(_hc_collapse_core, dynamic=False)
    _hc_expand = torch.compile(_hc_expand_core, dynamic=False)
    _HC_COMPILED = True


class HY4FlashRMSNorm(nn.Module):
    """RMSNorm that multiplies the weight after casting back to the activation dtype.

    Args:
        hidden_size: Size of the normalized axis.
        eps: Added to the mean square.
        dtype: Dtype of the weight.
        static_shape: Whether every call sees the same input shape. The compiled core (``BackendConfig.compile_norm``)
            specializes on the shape, so a norm whose input length follows the data (the indexer's pooled group
            keys: one call per document) passes ``False`` and always runs the eager core.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype | None = None, static_shape: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps
        self.static_shape = static_shape

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of the same shape and dtype as ``hidden_states``.
        """
        core = _rms_norm if self.static_shape else _rms_norm_core
        return core(hidden_states, self.weight, self.eps)


class HY4FlashHyperConnection(nn.Module):
    """iHC site: collapse ``hc_mult`` residual streams to one input and produce the scatter weights.

    All parameters are fp32 and the arithmetic runs in fp32 whatever the activation dtype.
    """

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_magnitude = config.hc_magnitude
        self.norm_eps = config.rms_norm_eps
        self.fn = nn.Parameter(torch.empty(2 * self.hc_mult, self.hc_mult * config.hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(2 * self.hc_mult, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
        """Initialize as the reference does: small ``fn``, ``scale`` 0.01 and a ``-log(hc_mult - 1)`` pre bias."""
        nn.init.normal_(self.fn, mean=0.0, std=initializer_range)
        nn.init.constant_(self.scale, 0.01)
        base = torch.zeros(2 * self.hc_mult, dtype=self.base.dtype, device=self.base.device)
        base[: self.hc_mult] = -math.log(max(self.hc_mult - 1, 1))
        if isinstance(self.base, DTensor):  # already sharded by FSDP: shard the full value the same way
            base = distribute_tensor(base, self.base.device_mesh, self.base.placements)
        self.base.copy_(base)

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the collapsed sub-layer input and the post weights.

        Args:
            streams: Tensor of shape [batch, sequence, hc_mult, hidden].

        Returns:
            ``post``: fp32 tensor of shape [batch, sequence, hc_mult]; ``collapsed``: tensor of shape
            [batch, sequence, hidden] in the dtype of ``streams``.
        """
        return _hc_collapse(
            streams,
            self.fn.float(),
            self.base.float(),
            self.scale.float(),
            self.norm_eps,
            self.hc_eps,
            self.hc_magnitude,
        )

    @staticmethod
    def expand(post: torch.Tensor, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Scatter a sub-layer output onto the streams: ``post * output + residual``.

        Args:
            post: fp32 tensor of shape [batch, sequence, hc_mult].
            output: Tensor of shape [batch, sequence, hidden].
            residual: Tensor of shape [batch, sequence, hc_mult, hidden]; its dtype is the result dtype.

        Returns:
            Tensor of shape [batch, sequence, hc_mult, hidden] in the dtype of ``residual``.
        """
        return _hc_expand(post, output, residual)


class HY4FlashHyperHead(nn.Module):
    """Final collapse of the streams before the output norm."""

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self, initializer_range: float) -> None:
        """Initialize as the reference does."""
        nn.init.normal_(self.hc_fn, mean=0.0, std=initializer_range)
        nn.init.constant_(self.hc_scale, 0.01)
        nn.init.constant_(self.hc_base, -math.log(max(self.hc_mult - 1, 1)))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """Collapse the streams.

        Args:
            streams: Tensor of shape [batch, sequence, hc_mult, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden] in the dtype of ``streams``.
        """
        wide = streams.float()
        flat = wide.flatten(2)
        mixes = F.linear(flat, self.hc_fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * wide).sum(dim=2).to(streams.dtype)


def inverse_frequencies(rope_dim: int, theta: float) -> torch.Tensor:
    """fp32 inverse frequencies ``theta ** (-2i / rope_dim)`` of shape [rope_dim // 2], computed from the config."""
    return 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))


def mrope_cos_sin(
    position_ids: torch.Tensor, inv_freq: torch.Tensor, mrope_section: list[int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved M-RoPE tables.

    The frequency axis takes the time plane everywhere except every third slot (starting at 1 for height and
    at 2 for width) within ``3 * section`` entries, which take the height / width planes.

    Args:
        position_ids: Integer tensor of shape [3, batch, sequence] (time, height, width positions).
        inv_freq: fp32 tensor of shape [rope_dim // 2].
        mrope_section: Three entries summing to ``rope_dim // 2``.
        dtype: Output dtype.

    Returns:
        ``(cos, sin)``, each of shape [batch, sequence, rope_dim // 2] (one value per rotated pair).
    """
    freqs = position_ids[..., None].float() * inv_freq.to(position_ids.device)
    merged = freqs[0].clone()
    for plane, offset in ((1, 1), (2, 2)):
        merged[..., slice(offset, mrope_section[plane] * 3, 3)] = freqs[plane][
            ..., slice(offset, mrope_section[plane] * 3, 3)
        ]
    return merged.cos().to(dtype), merged.sin().to(dtype)


def apply_interleaved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved channel pairs ``(x[2i], x[2i + 1])`` of the last axis in fp32.

    Args:
        x: Tensor of shape [..., rope_dim].
        cos: Tensor broadcastable to [..., rope_dim // 2] (e.g. [tokens, 1, rope_dim // 2] for per-head queries).
        sin: Tensor of the shape of ``cos``.

    Returns:
        Tensor of the shape and dtype of ``x``.
    """
    pairs = x.float().unflatten(-1, (-1, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    c, s = cos.float(), sin.float()
    out = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
    return out.flatten(-2).to(x.dtype)
