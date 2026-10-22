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

"""Module-owned precision boundaries for single-rank mixed-precision execution."""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn


def preserve_fp32_module_compute(model: nn.Module, *, module_type: type[nn.Module], device_type: str) -> nn.Module:
    """Exclude a model-declared module type from unsharded FSDP2 autocast.

    Explicit FP32 operand casts do not prevent autocast from lowering eligible
    operations such as linear projections. The owning model declares the module
    type whose entire computation must retain its upstream precision contract.

    Args:
        model: Model to adapt in place before parallelization.
        module_type: Module type whose forward must run outside autocast.
        device_type: Device on which the selected model executes.

    Returns:
        The original model with unchanged parameter identities and state keys.
    """
    for module in model.modules():
        if isinstance(module, module_type):
            autocast = torch.autocast(device_type, enabled=False)
            forward = module.forward
            if isinstance(forward, MethodType):
                # Use the unbound function so deepcopy binds to the new module.
                module.forward = MethodType(autocast(forward.__func__), module)
            else:
                module.forward = autocast(forward)
    return model
