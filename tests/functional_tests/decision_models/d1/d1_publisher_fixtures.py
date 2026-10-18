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

"""Nontrivial checkpoint inputs for independent D1 publisher comparisons."""

import torch


def initialize_nonunit_rmsnorm(model: torch.nn.Module) -> int:
    """Exercise affine rounding with deterministic non-unit publisher weights."""
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__.endswith("RMSNorm"):
                weight = module.weight
                weight.copy_(torch.linspace(0.65, 1.35, weight.numel(), device=weight.device, dtype=weight.dtype))
                count += 1
    if count < 7:
        raise ValueError(f"Expected complete tiny D1 RMSNorm coverage, found {count}")
    return count
