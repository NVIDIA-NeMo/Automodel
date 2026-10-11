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

"""Step-5 RMSNorm with optional fusion and unchanged FP32 scale semantics."""

import torch

from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    """Normalize [..., width] in FP32 before rounding once to the output dtype."""
    raw = hidden.float()
    normalized = raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
    return (normalized * (weight.float() + 1)).to(output_dtype)


_rms_norm_compiled = torch.compile(
    _rms_norm,
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True, "eager_numerics.division_rounding": True},
)


class Step5RMSNorm(Step3p5RMSNorm):
    """Fuse the main-stream norm while retaining FP32 additive scale parameters.

    Indexer normalization deliberately uses its separate implementation: its
    checkpoint requires rounding the scale to the activation dtype first.

    Args:
        hidden_size: Normalized feature width.
        eps: Additive variance epsilon.
        compile_norm: Enable the compiled CUDA core. CPU always uses eager math.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, *, compile_norm: bool = False) -> None:
        super().__init__(hidden_size, eps)
        self.compile_norm = compile_norm

    def forward(self, hidden: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Normalize [..., hidden_size] before the optional projection-input cast.

        Args:
            hidden: Activations with the normalized feature axis last. Decoder
                residual inputs remain FP32 through normalization.
            output_dtype: Final output dtype; defaults to the input dtype.

        Returns:
            Normalized activations with the input shape. The learned FP32 scale
            and every intermediate retain their original precision.
        """
        core = _rms_norm_compiled if self.compile_norm and hidden.is_cuda else _rms_norm
        return core(hidden, self.weight, self.variance_epsilon, hidden.dtype if output_dtype is None else output_dtype)
