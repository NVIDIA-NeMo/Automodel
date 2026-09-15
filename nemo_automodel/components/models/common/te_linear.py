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

"""Linear type compatibility for projections replaced in HuggingFace models."""

import torch

from nemo_automodel.components.models.common.utils import _patch_te_modules
from nemo_automodel.shared.import_utils import safe_import_from

HAVE_TE, TransformerEngineLinear = safe_import_from("transformer_engine.pytorch", "Linear")
if not HAVE_TE:
    raise ImportError("linear='te' requires Transformer Engine. Install nemo-automodel[cuda].")


class TELinear(torch.nn.Linear, TransformerEngineLinear):
    """Execute TE kernels while retaining native ``nn.Linear`` initialization hooks.

    ``nn.Linear`` must precede TE in the MRO: TE's cooperative base constructor
    must reach ``nn.Module.__init__``, not ``nn.Linear.__init__`` (which requires
    dimensions). Explicitly select TE's constructor, forward, and reset so the
    PyTorch base supplies only Linear type compatibility and metadata formatting.
    No model-owned initializer or owner reference is captured on the instance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device | str,
        params_dtype: torch.dtype,
    ) -> None:
        """Construct a replacement using the existing TE backend patches."""
        _patch_te_modules()
        TransformerEngineLinear.__init__(
            self, in_features, out_features, bias=bias, device=device, params_dtype=params_dtype
        )

    def forward(self, inp: torch.Tensor, is_first_microbatch: bool | None = None) -> torch.Tensor:
        """Apply the patched TE linear kernel.

        Args:
            inp: Tensor of shape [..., in_features], with arbitrary leading dimensions.
            is_first_microbatch: Whether to refresh the cached FP8 weights.

        Returns:
            Tensor of shape [..., out_features], preserving the leading dimensions.
        """
        return TransformerEngineLinear.forward(self, inp, is_first_microbatch=is_first_microbatch)

    reset_parameters = TransformerEngineLinear.reset_parameters
