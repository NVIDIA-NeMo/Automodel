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

import torch

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.shared.import_utils import safe_import_from


class QuackCrossEntropy(MaskedCrossEntropy):
    """Masked cross-entropy backed by QuACK's fused CUDA kernel."""

    def __init__(self, fp32_upcast: bool = False, ignore_index: int = -100, reduction: str = "sum"):
        super().__init__(fp32_upcast=fp32_upcast, ignore_index=ignore_index, reduction=reduction)
        available, cross_entropy = safe_import_from(
            "quack.cross_entropy",
            "cross_entropy",
            msg="QuackCrossEntropy requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
        )
        if not available:
            raise ImportError("QuackCrossEntropy requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
        self._quack_cross_entropy = cross_entropy

    def _cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._quack_cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
