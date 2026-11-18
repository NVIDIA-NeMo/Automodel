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


import torch


def normalize_token_loss(loss: torch.Tensor, count: int | torch.Tensor) -> torch.Tensor:
    """Normalize a sum-reduced loss, retaining the all-masked zero gradient.

    Args:
        loss: Scalar loss tensor.
        count: Supervised-token count as an integer or a scalar on the loss device.

    Returns:
        Scalar loss normalized by the token count.
    """
    if isinstance(count, torch.Tensor):
        # Keep the reciprocal in opmath precision, like division by a Python scalar.
        dtype = torch.float64 if loss.dtype == torch.float64 else torch.float32
        inverse = count.clamp_min(1).to(torch.float64).reciprocal().to(dtype)
        scaled = (loss.to(dtype) * inverse).to(loss.dtype)
        return torch.where(count != 0, scaled, loss * 0.0)
    return loss * 0.0 if count == 0 else loss / count
