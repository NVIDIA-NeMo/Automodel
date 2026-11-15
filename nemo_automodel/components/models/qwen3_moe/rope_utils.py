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

"""Qwen rotary arithmetic: FP32 intermediates and TE-compatible contraction."""

import torch

from nemo_automodel.components.models.gpt_oss.rope_utils import (
    _apply_rotary_emb_qk as _rotate_coefficients,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import (
    apply_rotary_emb_qk as _apply_shared_rotary,
)

# The shared helper deliberately emulates staged low-precision rounding. Qwen
# instead rotates FP32 values and allows the same fused multiply-add as TE.
# Disabling emulation on BF16 inputs would not establish this precision contract.
_compiled_fp32_rotary = torch.compile(_rotate_coefficients, dynamic=True, options={"emulate_precision_casts": False})


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    format: str = "bshd",
    rope_fusion: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    concentration: float | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate in FP32, apply concentration, then restore the original Q/K dtype.

    Args:
        q: Query [batch, sequence, heads, channels] or packed [tokens, heads, channels].
        k: Key with q's token axes and its own key-head count.
        freqs_cis: FP32 unscaled coefficients [..., rotary_dim] for eager rotation,
            or duplicated angles [positions, 1, 1, rotary_dim] for TE fusion.
        format: BSHD or packed THD layout.
        rope_fusion: Select the existing TE kernel rather than compiled coefficients.
        cu_seqlens: Physical cumulative token boundaries [documents + 1] for THD.
        concentration: Optional scale applied once after FP32 rotation.
        cp_size: Context-parallel degree.
        cp_rank: Rank in the existing context-parallel group.

    Returns:
        Rotated Q/K with the original shapes and dtypes, in independent storage.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    scale = 1.0 if concentration is None else concentration
    scaled = scale != 1.0
    if rope_fusion:
        # Unit-scale BF16 stays on TE's fast path. Scaling must precede the
        # final BF16 rounding, matching the eager FP32 arithmetic below.
        if scaled:
            q, k = q.float(), k.float()
        q, k = _apply_shared_rotary(
            q,
            k,
            freqs_cis,
            format=format,
            rope_fusion=True,
            cu_seqlens=cu_seqlens,
            concentration=scale if scaled else None,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
    else:
        q, k = q.float(), k.float()
        if q.is_cuda and not torch.compiler.is_compiling():
            q, k = _compiled_fp32_rotary(q, k, freqs_cis)
        else:
            q, k = _rotate_coefficients(q, k, freqs_cis)
        if scaled:
            q, k = q * scale, k * scale
    return q.to(q_dtype), k.to(k_dtype)
