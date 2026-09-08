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

"""Layers for the Titans model (linear *and* deep neural memory).

The token mixer is :class:`NeuralMemory`. It supports two memory kinds behind one
API, selected by ``mem_depth``:

* ``mem_depth == 1`` -- a linear (matrix-valued) associative memory updated by the
  gated delta rule with a *data-dependent momentum* over the per-token surprise
  (the recurrence documented below).
* ``mem_depth >= 2`` -- a *deep* (MLP) memory whose weights are the inner-loop
  state, updated by **chunkwise test-time gradient descent**: the per-token
  surprise is the gradient of the associative loss ``||M_W(k_t) - v_t||^2`` w.r.t.
  the MLP weights, carried by the same momentum + forget recurrence. The Phase-1
  per-head scalar gates map onto the deep update exactly: learning rate
  ``theta_t <- beta`` (delta step), momentum ``eta_t <- eta``, and forget
  keep-factor ``<- exp(g)`` (the GDN decay). ``chunk_size == 1`` is exact per-token
  GD; ``chunk_size > 1`` re-anchors the gradient at each chunk start (the Titans
  approximation). See :meth:`NeuralMemory._deep_recurrence`.

The linear recurrence (per head, state ``S`` of shape ``[K, V]``):

    alpha_t = exp(g_t)                                   # forget / decay gate
    pred_t  = (alpha_t * S_{t-1})^T k_t                  # memory readout for k_t
    surprise_t = k_t (x) [ beta_t (v_t - pred_t) ]       # delta-rule surprise (outer product)
    M_t = eta_t * M_{t-1} + surprise_t                   # data-dependent momentum
    S_t = alpha_t * S_{t-1} + M_t                        # write to memory
    o_t = S_t^T q_t                                      # retrieve

With ``eta_t == 0`` the momentum carry vanishes (``M_t = surprise_t``) and the
recurrence is *exactly* the Gated DeltaNet update implemented by fla's
``chunk_gated_delta_rule``. This is the reduction that makes Titans a strict
generalization of GDN, and the momentum-disabled path delegates straight to the
fla kernel for speed.

``A_log`` / ``dt_bias`` are intrinsically fp32 (``A_log`` is exponentiated, so
bf16 rounding becomes a proportional error on the decay rate that the recurrence
compounds across the sequence) and are kept fp32 regardless of compute dtype --
see ``_keep_in_fp32_modules`` on the model and ``state_dict_adapter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.shared.import_utils import safe_import

_HAVE_FLA, _fla = safe_import("fla.ops.gated_delta_rule")


@dataclass
class NeuralMemoryState:
    """Recurrent state carried between chunk-aligned deep-memory calls."""

    weights: tuple[torch.Tensor, ...]
    momentum: tuple[torch.Tensor, ...]
    qkv_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class _FP32DecayGateParameters(nn.Module):
    """Isolate decay parameters in a dtype-uniform FSDP2 subtree."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """Evaluate the decay gate while FSDP owns parameter unsharding."""
        return -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)


def titans_delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    eta: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive (sequential) Titans linear-memory recurrence.

    Mirrors fla's ``naive_recurrent_gated_delta_rule`` convention (decay applied to
    the state *before* the delta correction) and adds the data-dependent momentum
    carry ``M``. At ``eta == 0`` it is numerically identical to the gated delta rule.

    Args:
        q, k, v: ``[B, T, H, D]`` query / key / value (q, k expected L2-normalized).
        g: ``[B, T, H]`` log-decay (``g <= 0``); ``alpha_t = exp(g_t)``.
        beta: ``[B, T, H]`` delta-rule surprise step size (learning rate).
        eta: ``[B, T, H]`` data-dependent momentum coefficient in ``[0, 1)``.
        scale: query scale; defaults to ``1/sqrt(D)``.

    Returns:
        ``[B, T, H, D]`` retrieved values.

    Note:
        This is an O(T) python loop kept for correctness/reference and for the
        ``mem_depth=1`` momentum path. The momentum-disabled path uses the fused
        fla chunk kernel; the chunked momentum kernel is future work (Phase 2).
    """
    q, k, v, beta, g, eta = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g, eta))
    B, H, T, K = q.shape
    Vd = v.shape[-1]
    if scale is None:
        scale = 1.0 / (K**0.5)
    q = q * scale

    S = q.new_zeros(B, H, K, Vd)
    M = q.new_zeros(B, H, K, Vd)
    o = q.new_zeros(B, H, T, Vd)
    for t in range(T):
        alpha = g[:, :, t].exp()[..., None, None]  # [B,H,1,1]
        S = S * alpha
        pred = (S * k[:, :, t][..., None]).sum(-2)  # (alpha S)^T k -> [B,H,V]
        dv = (v[:, :, t] - pred) * beta[:, :, t][..., None]  # [B,H,V]
        surprise = k[:, :, t].unsqueeze(-1) * dv.unsqueeze(-2)  # [B,H,K,V]
        M = M * eta[:, :, t][..., None, None] + surprise
        S = S + M
        o[:, :, t] = torch.einsum("bhd,bhdm->bhm", q[:, :, t], S)
    return o.transpose(1, 2).contiguous()


class TitansRMSNorm(nn.Module):
    """RMSNorm with an optional multiplicative gate (SiLU of a gate tensor)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        out = x.float()
        if gate is not None:
            out = out * F.silu(gate.float())
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * self.weight.float()
        return out.type_as(x)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class NeuralMemory(nn.Module):
    """Titans neural-memory token mixer (linear or deep), one shared API.

        ``NeuralMemory(dim, mem_dim=64, num_heads=None, mem_depth=1,
                       chunk_size=16, momentum=True, forget=True)``
        ``forward(x: [B, S, dim]) -> retrieved: [B, S, dim]``

    With ``mem_depth == 1`` the memory is a per-head matrix
    ``S in R^{mem_dim x mem_dim}`` updated by the gated delta rule with
    data-dependent momentum. With ``mem_depth >= 2`` the memory is a per-head MLP
    whose weights are updated by chunkwise test-time gradient descent; the same
    projections (``q/k/v_proj``) and per-head scalar gates drive both paths (see
    :meth:`_deep_recurrence` for the gate mapping).

    Args:
        dim: Residual-stream dimension (input and output width).
        mem_dim: Per-head key/value (memory) dimension.
        num_heads: Number of memory heads. Defaults to ``dim // mem_dim``.
        mem_depth: Memory depth. ``1`` is the linear matrix memory; ``>=2`` is a
            deep MLP memory updated by test-time gradient descent.
        expansion_factor: Hidden-width multiplier for intermediate deep-memory
            layers.
        memory_residual_norm: Add the memory input to an RMS-normalized MLP
            output, with the norm scale included in the fast weights.
        qkv_conv_kernel_size: Width of the causal depthwise convolution after
            Q/K/V projection. ``0`` disables it.
        chunk_size: Chunk size. For ``mem_depth==1`` it is the hint forwarded to the
            fla GDN kernel; for ``mem_depth>=2`` it is the test-time-GD chunk length
            (``1`` is exact per-token GD, larger re-anchors the gradient per chunk).
        momentum: Enable the data-dependent momentum (Titans). If ``False``, the
            recurrence is exactly Gated DeltaNet (delegates to fla).
        forget: Enable the data-dependent decay/forget gate. If ``False``, decay is
            fixed to 1 (pure, non-gated delta rule).
        dtype: Compute dtype for projection weights (``A_log`` / ``dt_bias`` are fp32).
    """

    def __init__(
        self,
        dim: int,
        mem_dim: int = 64,
        num_heads: int | None = None,
        mem_depth: int = 1,
        expansion_factor: float = 4.0,
        memory_residual_norm: bool = True,
        qkv_conv_kernel_size: int = 4,
        chunk_size: int = 16,
        momentum: bool = True,
        forget: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if mem_depth < 1:
            raise ValueError(f"NeuralMemory mem_depth must be >= 1 (1 = linear, >=2 = deep MLP); got {mem_depth}.")
        if num_heads is None:
            if dim % mem_dim != 0:
                raise ValueError(f"dim ({dim}) must be divisible by mem_dim ({mem_dim}) when num_heads is None.")
            num_heads = dim // mem_dim

        self.dim = dim
        self.mem_dim = mem_dim
        self.num_heads = num_heads
        self.mem_depth = mem_depth
        self.expansion_factor = expansion_factor
        self.memory_residual_norm = memory_residual_norm
        self.qkv_conv_kernel_size = qkv_conv_kernel_size
        self.chunk_size = chunk_size
        self.momentum = momentum
        self.forget = forget
        self.inner_dim = num_heads * mem_dim

        # q/k/v projections (q,k in mem_dim per head -> keys/queries; v in mem_dim).
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False, dtype=dtype)
        if qkv_conv_kernel_size > 0:
            conv_kwargs = {
                "in_channels": self.inner_dim,
                "out_channels": self.inner_dim,
                "kernel_size": qkv_conv_kernel_size,
                "groups": self.inner_dim,
                "bias": False,
                "dtype": dtype,
            }
            self.q_conv = nn.Conv1d(**conv_kwargs)
            self.k_conv = nn.Conv1d(**conv_kwargs)
            self.v_conv = nn.Conv1d(**conv_kwargs)
        # Per-head scalar gates: beta (delta step), a (feeds decay gate), eta (momentum).
        self.b_proj = nn.Linear(dim, num_heads, bias=False, dtype=dtype)
        self.a_proj = nn.Linear(dim, num_heads, bias=False, dtype=dtype)
        if momentum:
            self.m_proj = nn.Linear(dim, num_heads, bias=False, dtype=dtype)
        # Output gate (z) and projection (mirrors GatedDeltaNet's gated norm).
        self.g_proj = nn.Linear(dim, self.inner_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=False, dtype=dtype)
        self.norm = TitansRMSNorm(mem_dim, eps=1e-6)

        # Intrinsically-fp32 decay-gate params (per head). g = -exp(A_log)*softplus(a+dt_bias).
        if forget:
            self._fp32_params = _FP32DecayGateParameters(num_heads)

        # Deep (mem_depth>=2) memory: a per-head MLP whose weights ARE the memory,
        # updated online by test-time gradient descent. These nn.Parameters are the
        # learnable *initial* weights (the chunk-0 anchor, re-used each forward); the
        # inner loop updates a per-sequence copy.
        if mem_depth >= 2:
            hidden_dim = max(1, int(mem_dim * expansion_factor))
            layer_dims = (mem_dim, *((hidden_dim,) * (mem_depth - 1)), mem_dim)
            self.mem_weights = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(num_heads, d_in, d_out, dtype=dtype))
                    for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:])
                ]
            )
            if memory_residual_norm:
                self.mem_norm_weight = nn.Parameter(torch.ones(num_heads, mem_dim, dtype=dtype))
            self._init_mem_weights()

    def _init_mem_weights(self) -> None:
        """Xavier-uniform init of each per-head deep-memory weight matrix."""
        for w in self.mem_weights:
            for h in range(self.num_heads):
                nn.init.xavier_uniform_(w[h])
        if self.memory_residual_norm:
            nn.init.ones_(self.mem_norm_weight)

    def _decay_gate(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the log-decay gate ``g`` in fp32. Returns ``[B, S, H]`` (``g <= 0``)."""
        if not self.forget:
            return torch.zeros_like(a, dtype=torch.float32)
        return self._fp32_params(a)

    @property
    def A_log(self) -> torch.Tensor:
        """Per-head log decay rate, stored in the fp32 FSDP subtree."""
        return self._fp32_params.A_log

    @property
    def dt_bias(self) -> torch.Tensor:
        """Per-head decay-controller bias, stored in the fp32 FSDP subtree."""
        return self._fp32_params.dt_bias

    def _apply_causal_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a depthwise convolution and return its streaming history."""
        x = x.transpose(1, 2)
        history_size = self.qkv_conv_kernel_size - 1
        if history is None:
            history = x.new_zeros(x.shape[0], x.shape[1], history_size)
        if history.shape != (x.shape[0], x.shape[1], history_size):
            raise ValueError(
                "Invalid QKV convolution history shape: "
                f"expected {(x.shape[0], x.shape[1], history_size)}, got {tuple(history.shape)}."
            )
        joined = torch.cat((history, x), dim=-1)
        next_history = joined[:, :, -history_size:] if history_size > 0 else joined[:, :, :0]
        return conv(joined).transpose(1, 2), next_history

    # ------------------------------------------------------------------ #
    # Deep (mem_depth>=2) memory: per-head MLP + chunkwise test-time GD.  #
    # All deep tensors fold the head axis into the batch (B*H).          #
    # ------------------------------------------------------------------ #
    def _deep_init_weights(self, batch: int, dtype: torch.dtype) -> list[torch.Tensor]:
        """Per-(batch*head) copies of the learnable initial memory weights.

        Args:
            batch: Outer batch size ``B``.
            dtype: Working dtype for the inner-loop recurrence.

        Returns:
            One tensor per MLP layer, each ``[B*H, d_in, d_out]`` (the chunk-0 anchor).
        """
        out = []
        for w in self.mem_weights:
            wd = w.to(dtype)  # [H, in, out]; autograd flows back to the nn.Parameter
            out.append(wd.unsqueeze(0).expand(batch, *wd.shape).reshape(batch * self.num_heads, *wd.shape[1:]))
        if self.memory_residual_norm:
            norm = self.mem_norm_weight.to(dtype)
            out.append(norm.unsqueeze(0).expand(batch, *norm.shape).reshape(batch * self.num_heads, self.mem_dim))
        return out

    def _mem_forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        """Apply the memory MLP. ``x``: ``[N, c, mem_dim]``; weights: ``[N, in, out]``."""
        residual = x
        mlp_weights = weights
        norm_weight = None
        if self.memory_residual_norm:
            *mlp_weights, norm_weight = weights

        h = x
        for i, w in enumerate(mlp_weights):
            if i > 0:
                h = F.gelu(h)
            h = torch.einsum("ncm,nmo->nco", h, w)
        if norm_weight is not None:
            inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            h = residual + h * inv_rms * norm_weight[:, None, :]
        return h

    def _mem_grads(self, k: torch.Tensor, v: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        """Analytic per-token gradients of ``(1/m)||M_W(k) - v||^2`` w.r.t. each weight.

        Args:
            k: Keys ``[N, c, mem_dim]`` (the MLP inputs).
            v: Targets ``[N, c, mem_dim]``.
            weights: Anchor weights, each ``[N, in, out]``.

        Returns:
            Per-token gradients, one tensor per layer shaped ``[N, c, in, out]``.
        """
        mlp_weights = weights
        norm_weight = None
        if self.memory_residual_norm:
            *mlp_weights, norm_weight = weights

        layer_inputs: list[torch.Tensor] = []  # matmul input of each layer
        pre_acts: list[torch.Tensor] = []  # matmul outputs (pre next-layer GELU)
        h = k
        for i, w in enumerate(mlp_weights):
            if i > 0:
                h = F.gelu(h)
            layer_inputs.append(h)
            h = torch.einsum("ncm,nmo->nco", h, w)
            pre_acts.append(h)
        raw_out = h
        if norm_weight is not None:
            inv_rms = torch.rsqrt(raw_out.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            normalized = raw_out * inv_rms
            out = k + normalized * norm_weight[:, None, :]
        else:
            out = raw_out

        m = out.shape[-1]
        delta = (2.0 / m) * (out - v)  # dL/d(out)  [N, c, mem_dim]
        norm_grad = None
        if norm_weight is not None:
            norm_grad = delta * normalized
            scaled_delta = delta * norm_weight[:, None, :]
            projection = (scaled_delta * raw_out).mean(dim=-1, keepdim=True)
            delta = scaled_delta * inv_rms - raw_out * projection * inv_rms.pow(3)
        # Exact derivative of (erf) GELU, for backprop through hidden activations.
        sqrt2 = 2.0**0.5
        sqrt2pi = (2.0 * torch.pi) ** 0.5

        grads: list[torch.Tensor] = [torch.empty(0)] * len(mlp_weights)
        for i in reversed(range(len(mlp_weights))):
            grads[i] = torch.einsum("ncm,nco->ncmo", layer_inputs[i], delta)
            if i > 0:
                delta = torch.einsum("nco,nmo->ncm", delta, mlp_weights[i])
                pre = pre_acts[i - 1]
                dgelu = 0.5 * (1.0 + torch.erf(pre / sqrt2)) + pre * torch.exp(-0.5 * pre * pre) / sqrt2pi
                delta = delta * dgelu
        if norm_grad is not None:
            grads.append(norm_grad)
        return grads

    @staticmethod
    def _deep_scan_matrix(gate_log_cumsum: torch.Tensor) -> torch.Tensor:
        """Lower-triangular scan matrix ``M[n,t,j] = prod_{l=j+1..t} gate_l``.

        Args:
            gate_log_cumsum: ``[N, c]`` cumulative log-gates ``c_t = sum_{0..t} log g``.

        Returns:
            ``[N, c, c]`` matrix with ``exp(c_t - c_j)`` for ``j <= t`` else ``0``.

        Note:
            The mask is applied *in log space* (fill the strict upper triangle with
            ``-inf`` before ``exp``) rather than multiplying ``exp(diff)`` by a 0/1
            mask. For ``j > t`` the gates are non-increasing so ``diff > 0`` and
            ``exp(diff)`` can overflow to ``inf``; a ``0 * inf`` then yields ``nan``
            in the backward pass. Masking the exponent keeps both passes finite (the
            valid ``j <= t`` part has ``diff <= 0``, so ``exp(diff) <= 1``).
        """
        c = gate_log_cumsum.shape[1]
        diff = gate_log_cumsum.unsqueeze(2) - gate_log_cumsum.unsqueeze(1)  # [N, t, j]
        mask = torch.tril(torch.ones(c, c, dtype=torch.bool, device=gate_log_cumsum.device))
        return torch.exp(diff.masked_fill(~mask, float("-inf")))

    def _deep_chunk_update(
        self,
        surprise: list[torch.Tensor],  # s_t = -theta_t * grad ; each [N, c, in, out]
        eta: torch.Tensor,  # [N, c] momentum gates
        keep: torch.Tensor,  # [N, c] forget keep-factor (= exp(g))
        weight_carry: list[torch.Tensor],  # W_{-1} (chunk anchor) ; each [N, in, out]
        momentum_carry: list[torch.Tensor],  # S_{-1} ; each [N, in, out]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Closed-form within-chunk momentum + forget scan; returns end-of-chunk state.

        Implements, per token ``t`` inside the chunk::

            S_t = eta_t * S_{t-1} + s_t          # momentum on surprise
            W_t = keep_t * W_{t-1} + S_t         # forget (weight decay), keep_t = exp(g_t)

        carried from ``weight_carry`` / ``momentum_carry`` at token ``-1``.
        """
        if self.momentum:
            eta_cum = torch.cumsum(torch.log(eta.clamp_min(1e-20)), dim=1)  # [N, c]
            m_eta = self._deep_scan_matrix(eta_cum)  # [N, c, c]
            eta_prefix = torch.exp(eta_cum)  # prod_{0..t} eta -> carry coeff
        keep_cum = torch.cumsum(torch.log(keep.clamp_min(1e-20)), dim=1)  # [N, c]
        m_keep = self._deep_scan_matrix(keep_cum)  # [N, c, c]
        keep_prefix = torch.exp(keep_cum)  # prod_{0..t} keep -> carry coeff

        new_weights: list[torch.Tensor] = []
        new_momentum: list[torch.Tensor] = []
        for s, w0, s0 in zip(surprise, weight_carry, momentum_carry):
            carry_shape = (s.shape[0], s.shape[1], *((1,) * (s.ndim - 2)))
            if self.momentum:
                S = torch.einsum("ntj,nj...->nt...", m_eta, s)
                S = S + eta_prefix.reshape(carry_shape) * s0[:, None]
            else:
                S = s  # no momentum: surprise passes straight through
            W = torch.einsum("ntj,nj...->nt...", m_keep, S)
            W = W + keep_prefix.reshape(carry_shape) * w0[:, None]
            new_weights.append(W[:, -1])
            new_momentum.append(S[:, -1])
        return new_weights, new_momentum

    def _deep_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        theta: torch.Tensor,
        g: torch.Tensor,
        eta: torch.Tensor | None,
        past_state: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Chunkwise test-time gradient descent on the per-head deep MLP memory.

        Maps the Phase-1 gates onto the deep update: the learning rate ``theta_t``
        is ``beta`` (the delta-rule step), the momentum is ``eta``, and the forget
        keep-factor is ``exp(g)`` (the GDN decay). Each chunk retrieves with its
        anchor weights, then advances the memory by one chunk of test-time GD.

        Keys/queries are L2-normalized (matching the linear path) before driving
        the memory.

        Args:
            q, k, v: ``[B, S, H, mem_dim]`` query / key / value (per head).
            theta: ``[B, S, H]`` per-head learning rate (Phase-1 ``beta``).
            g: ``[B, S, H]`` log-decay (``g <= 0``); keep-factor is ``exp(g)``.
            eta: ``[B, S, H]`` momentum coefficient, or ``None`` when momentum is off.

        Returns:
            ``[B, S, H, mem_dim]`` retrieved values (anchor-causal at chunk granularity).
        """
        B, S, H, D = q.shape
        c = self.chunk_size
        # Work in fp32 for bf16/fp16 inputs (stable cumsum/exp/grads); preserve fp64.
        compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        def fold(t: torch.Tensor) -> torch.Tensor:  # [B,S,H,...] -> [B*H,S,...]
            tt = t.transpose(1, 2)  # [B,H,S,...]
            return tt.reshape(B * H, *tt.shape[2:]).to(compute_dtype)

        # L2-normalize keys/queries so the test-time-GD inner loop sees bounded
        # inputs (mirrors the linear path and stabilizes the deep memory).
        qf = F.normalize(fold(q), dim=-1)
        kf = F.normalize(fold(k), dim=-1)
        vf = fold(v)
        thetaf = fold(theta)  # [B*H, S]
        keepf = fold(g).exp()  # [B*H, S] keep-factor in (0, 1]
        etaf = fold(eta) if eta is not None else torch.zeros_like(thetaf)

        # Pad to a multiple of chunk_size: padded tokens get theta=0 (no surprise)
        # and keep=1 (no forgetting), so they cannot perturb any real token's state.
        pad = (c - S % c) % c
        if pad:
            qf = F.pad(qf, (0, 0, 0, pad))
            kf = F.pad(kf, (0, 0, 0, pad))
            vf = F.pad(vf, (0, 0, 0, pad))
            thetaf = F.pad(thetaf, (0, pad), value=0.0)
            keepf = F.pad(keepf, (0, pad), value=1.0)
            etaf = F.pad(etaf, (0, pad), value=0.0)
        n_chunks = (S + pad) // c

        N = B * H
        qc = qf.reshape(N, n_chunks, c, D)
        kc = kf.reshape(N, n_chunks, c, D)
        vc = vf.reshape(N, n_chunks, c, D)
        thetac = thetaf.reshape(N, n_chunks, c)
        keepc = keepf.reshape(N, n_chunks, c)
        etac = etaf.reshape(N, n_chunks, c)

        if past_state is None:
            weights = self._deep_init_weights(B, compute_dtype)  # chunk anchor, each [N, ...]
            momentum = [torch.zeros_like(w) for w in weights]
        else:
            weights, momentum = past_state
            weights = [w.to(device=q.device, dtype=compute_dtype) for w in weights]
            momentum = [m.to(device=q.device, dtype=compute_dtype) for m in momentum]

        retrieved_chunks = []
        for i in range(n_chunks):
            # retrieve this chunk with the anchor weights (state before this chunk)
            retrieved_chunks.append(self._mem_forward(qc[:, i], weights))
            # surprise = -theta * grad, evaluated at the anchor weights
            grads = self._mem_grads(kc[:, i], vc[:, i], weights)
            surprise = [-thetac[:, i].reshape(N, c, *((1,) * (gr.ndim - 2))) * gr for gr in grads]
            # advance memory: momentum + forget recurrence within the chunk
            weights, momentum = self._deep_chunk_update(surprise, etac[:, i], keepc[:, i], weights, momentum)

        retrieved = torch.cat(retrieved_chunks, dim=1)[:, :S]  # [N, S, D]
        retrieved = retrieved.reshape(B, H, S, D).transpose(1, 2).reshape(B, S, H, D)
        if return_state:
            return retrieved, (weights, momentum)
        return retrieved

    def forward(
        self,
        x: torch.Tensor,
        past_state: NeuralMemoryState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, NeuralMemoryState]:
        """Map ``x: [B, S, dim]`` to retrieved memory output ``[B, S, dim]``."""
        B, S, _ = x.shape
        H, D = self.num_heads, self.mem_dim
        if (past_state is not None or return_state) and self.mem_depth < 2:
            raise NotImplementedError("Streaming state is currently supported only for deep memory (mem_depth >= 2).")
        if (past_state is not None or return_state) and S % self.chunk_size != 0:
            raise ValueError(
                f"Streaming calls must be chunk-aligned: sequence length {S} is not divisible by {self.chunk_size}."
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        next_qkv_history: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if self.qkv_conv_kernel_size > 0:
            histories = past_state.qkv_history if past_state is not None else (None, None, None)
            q, q_history = self._apply_causal_conv(q, self.q_conv, histories[0])
            k, k_history = self._apply_causal_conv(k, self.k_conv, histories[1])
            v, v_history = self._apply_causal_conv(v, self.v_conv, histories[2])
            next_qkv_history = (q_history, k_history, v_history)
        q = q.view(B, S, H, D)
        k = k.view(B, S, H, D)
        v = v.view(B, S, H, D)
        beta = self.b_proj(x).sigmoid()  # [B,S,H]
        g = self._decay_gate(self.a_proj(x))  # [B,S,H] fp32, <= 0

        if self.mem_depth >= 2:
            # Deep MLP memory updated by chunkwise test-time gradient descent.
            eta = self.m_proj(x).sigmoid() if self.momentum else None
            deep_past = None
            if past_state is not None:
                deep_past = (list(past_state.weights), list(past_state.momentum))
            deep_result = self._deep_recurrence(
                q,
                k,
                v,
                beta,
                g,
                eta,
                past_state=deep_past,
                return_state=return_state,
            )
            if return_state:
                core, (next_weights, next_momentum) = deep_result
            else:
                core = deep_result
        else:
            # Linear matrix memory (gated delta rule + momentum).
            # L2-normalize q,k (matches fla use_qk_l2norm_in_kernel=True).
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            v = v.float()

            if self.momentum:
                eta = self.m_proj(x).sigmoid()  # [B,S,H] in (0,1)
                core = titans_delta_rule_recurrence(q, k, v, g, beta, eta)
            elif _HAVE_FLA:
                # Momentum disabled == Gated DeltaNet: use the fused fla chunk kernel.
                core, _ = _fla.chunk_gated_delta_rule(
                    q, k, v, g.float(), beta.float(), use_qk_l2norm_in_kernel=False, output_final_state=False
                )
            else:
                eta = torch.zeros(B, S, H, device=x.device)
                core = titans_delta_rule_recurrence(q, k, v, g, beta, eta)

        core = core.reshape(B, S, H, D)
        gate = self.g_proj(x).view(B, S, H, D)
        core = self.norm(core, gate)  # gated RMSNorm, per head
        core = core.reshape(B, S, self.inner_dim).type_as(self.o_proj.weight)
        output = self.o_proj(core)
        if return_state:
            if next_qkv_history is None:
                empty = x.new_empty(B, self.inner_dim, 0)
                next_qkv_history = (empty, empty, empty)
            state = NeuralMemoryState(
                weights=tuple(next_weights),
                momentum=tuple(next_momentum),
                qkv_history=next_qkv_history,
            )
            return output, state
        return output

    def init_weights(self, init_std: float = 0.02):
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.b_proj, self.a_proj, self.g_proj, self.o_proj):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=init_std)
        if self.qkv_conv_kernel_size > 0:
            for conv in (self.q_conv, self.k_conv, self.v_conv):
                nn.init.trunc_normal_(conv.weight, mean=0.0, std=init_std)
        if self.momentum:
            nn.init.trunc_normal_(self.m_proj.weight, mean=0.0, std=init_std)
        if self.mem_depth >= 2:
            self._init_mem_weights()
        self.norm.reset_parameters()
        if self.forget:
            # Match the GatedDeltaNet init: dt_bias=1, A_log=log(U(0,16)).
            self.dt_bias.data.fill_(1.0)
            self.A_log.data.uniform_(0, 16).log_()


class TitansMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, init_std: float = 0.02):
        for lin in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=init_std)


class TitansBlock(nn.Module):
    """Pre-norm decoder block: NeuralMemory token mixer + SwiGLU MLP."""

    def __init__(self, config, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.input_layernorm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory = NeuralMemory(
            dim=config.hidden_size,
            mem_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            mem_depth=config.mem_depth,
            expansion_factor=config.memory_expansion_factor,
            memory_residual_norm=config.memory_residual_norm,
            qkv_conv_kernel_size=config.qkv_conv_kernel_size,
            chunk_size=config.chunk_size,
            momentum=config.momentum,
            forget=config.forget,
            dtype=dtype,
        )
        self.post_attention_layernorm = TitansRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TitansMLP(config.hidden_size, config.intermediate_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.memory(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def init_weights(self, init_std: float = 0.02):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.memory.init_weights(init_std)
        self.mlp.init_weights(init_std)
