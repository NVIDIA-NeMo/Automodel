# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""fp32 decay gates shared by the linear-attention and Mamba model families.

GatedDeltaNet (Qwen3-Next, Qwen3.5, Qwen3.8-Flash-Next) and Kimi Delta Attention
(Kimi K3, Kimi Linear, GLM-5.3) carry intrinsically-fp32 decay parameters: ``A_log``
is exponentiated, so bf16 rounding becomes a proportional error on the decay rate
that the recurrence compounds across the sequence. Each family stores ``A_log`` and
``dt_bias`` as plain fp32 parameters under their HF names, lists them in
``_keep_in_fp32_modules_strict`` (so ``cast_model_to_dtype`` restores them and FSDP2
keeps them computing in fp32 under a bf16 ``param_dtype``), and computes the gate here.

The gate functions cast their own activation input to fp32: under FSDP mixed precision
the block's inputs arrive in bf16 while the parameters stay fp32.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.shared.import_utils import safe_import_from

# ``_keep_in_fp32_modules_strict`` tokens, matched as substrings of the canonical parameter
# FQN. They carry the owning attribute so they cannot match unrelated parameters.
GDN_FP32_PARAM_TOKENS: tuple[str, ...] = ("linear_attn.A_log", "linear_attn.dt_bias")
MAMBA_FP32_PARAM_TOKENS: tuple[str, ...] = ("e_score_correction_bias", "mixer.A_log", "mixer.dt_bias", "mixer.D")

_FLA_MSG = "The fused KDA gate requires the flash-linear-attention/fla extra. Install with `uv sync --extra fla`."
HAVE_FUSED_KDA_GATE, _fla_fused_kda_gate = safe_import_from("fla.ops.kda.gate", "fused_kda_gate", msg=_FLA_MSG)
try:
    _FUSED_KDA_GATE_PARAMS = inspect.signature(_fla_fused_kda_gate).parameters if HAVE_FUSED_KDA_GATE else {}
except (TypeError, ValueError):
    _FUSED_KDA_GATE_PARAMS = {}
# Older FLA releases take ``(g, A_log, head_k_dim, g_bias=...)`` with a flat gate; newer ones take
# ``(g, A_log, dt_bias=..., lower_bound=...)`` with a per-head gate.
_FUSED_KDA_GATE_HAS_G_BIAS = "g_bias" in _FUSED_KDA_GATE_PARAMS
_FUSED_KDA_GATE_HAS_LOWER_BOUND = "lower_bound" in _FUSED_KDA_GATE_PARAMS


def gdn_decay_gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Compute the GatedDeltaNet decay gate ``g = -exp(A_log) * softplus(a + dt_bias)`` in fp32.

    This is the formula of HF ``Qwen3NextGatedDeltaNet.forward`` with every operand in fp32.

    Args:
        a: Decay-rate pre-activation of shape ``[batch, sequence, num_v_heads]`` in the
            block's compute dtype.
        A_log: Log decay parameter of shape ``[num_v_heads]``.
        dt_bias: Decay bias parameter of shape ``[num_v_heads]``.

    Returns:
        fp32 gate with the shape of ``a``.
    """
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


def pin_gdn_params_fp32(module: nn.Module) -> None:
    """Rebuild ``module.A_log`` / ``module.dt_bias`` as fp32 parameters, preserving their values.

    HF's GatedDeltaNet constructors create both in the default dtype; the fp32 contract
    requires fp32 storage independent of the model dtype.

    Args:
        module: A GatedDeltaNet module that already owns ``A_log`` and ``dt_bias``.
    """
    module.A_log = nn.Parameter(module.A_log.detach().to(torch.float32))
    module.dt_bias = nn.Parameter(module.dt_bias.detach().to(torch.float32))


def _per_head_gate(g: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Return ``g`` viewed as ``[..., heads, head_dim]``."""
    return g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)


def _torch_kda_decay_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    head_dim: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Compute the KDA decay gate with torch fp32 operations (see :func:`kda_decay_gate`)."""
    gate = _per_head_gate(g, head_dim)
    num_heads = gate.shape[-2]
    gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
    decay = A_log.float().view(num_heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(decay * gate)
    return -decay * F.softplus(gate)


def _fused_kda_decay_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    head_dim: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Call FLA's fused KDA gate across its API generations (see :func:`kda_decay_gate`)."""
    if _FUSED_KDA_GATE_HAS_G_BIAS:
        if lower_bound is not None:
            return _torch_kda_decay_gate(g, A_log, dt_bias, head_dim, lower_bound)
        return _fla_fused_kda_gate(g, A_log.view(1, 1, -1, 1), head_dim, g_bias=dt_bias)
    gate = _per_head_gate(g, head_dim)
    if _FUSED_KDA_GATE_HAS_LOWER_BOUND:
        return _fla_fused_kda_gate(gate, A_log.view(-1), dt_bias=dt_bias, lower_bound=lower_bound)
    if lower_bound is not None:
        return _torch_kda_decay_gate(gate, A_log, dt_bias, head_dim, lower_bound)
    return _fla_fused_kda_gate(gate, A_log.view(-1), dt_bias=dt_bias)


def kda_decay_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    head_dim: int,
    lower_bound: float | None,
    use_fused: bool,
) -> torch.Tensor:
    """Compute the Kimi Delta Attention decay gate in fp32.

    Without ``lower_bound`` the gate is ``-exp(A_log) * softplus(g + dt_bias)``; with it, the
    bounded form ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` used by Kimi K3 and
    GLM-5.3. ``A_log`` is read per head, so ``[heads]`` and ``[1, 1, heads, 1]`` layouts are
    both accepted.

    Args:
        g: Raw gate projection of shape ``[..., heads * head_dim]`` or ``[..., heads, head_dim]``.
        A_log: Log decay parameter with ``heads`` elements.
        dt_bias: Decay bias parameter of shape ``[heads * head_dim]``.
        head_dim: Per-head KDA dimension.
        lower_bound: Lower bound of the bounded decay, or ``None`` for the softplus form.
        use_fused: Run FLA's fused Triton gate. Requires the fla extra and CUDA tensors; the
            torch path is used when the installed FLA API cannot express ``lower_bound``.

    Returns:
        fp32 decay of shape ``[..., heads, head_dim]``.
    """
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    if use_fused:
        return _fused_kda_decay_gate(g, A_log, dt_bias, head_dim, lower_bound)
    return _torch_kda_decay_gate(g, A_log, dt_bias, head_dim, lower_bound)
