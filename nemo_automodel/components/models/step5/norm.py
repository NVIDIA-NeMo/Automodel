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


def _rms_norm_with_rstd(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize in FP32 and retain the row statistic needed by backward."""
    raw = hidden.float()
    rstd = torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
    normalized = raw * rstd
    return (normalized * (weight.float() + 1)).to(output_dtype), rstd


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    """Normalize [..., width] in FP32 before rounding once to the output dtype."""
    return _rms_norm_with_rstd(hidden, weight, eps, output_dtype)[0]


# Residual and activation inputs have different precision and layout families.
# Give them separate Dynamo entries: mixing FP32/BF16, grad/no_grad, output
# casts, and strided QKV in one entry exhausts the default specialization cache.
@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True, "eager_numerics.division_rounding": True},
)
def _residual_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> torch.Tensor:
    return _rms_norm(hidden, weight, eps, output_dtype)


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True, "eager_numerics.division_rounding": True},
)
def _activation_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the same reduction outputs in grad/no_grad. Otherwise Inductor can
    # pick a different summation tree when the backward statistic is dead,
    # changing BF16 rounding at the final norm during checkpoint replay.
    return _rms_norm_with_rstd(hidden, weight, eps, output_dtype)


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
        dtype = hidden.dtype if output_dtype is None else output_dtype
        if self.compile_norm and hidden.is_cuda:
            # Normalize the leading layout so decoder, Q/K, and final norms
            # share a compiled rank. Interleaved key projections stay strided
            # views: only the normalized feature axis must remain last.
            rows = hidden.reshape(-1, hidden.shape[-1])
            if hidden.dtype == torch.float32:
                output = _residual_rms_norm(rows, self.weight, self.variance_epsilon, dtype)
            else:
                output, _ = _activation_rms_norm(rows, self.weight, self.variance_epsilon, dtype)
            return output.reshape(hidden.shape)
        return _rms_norm(hidden, self.weight, self.variance_epsilon, dtype)
