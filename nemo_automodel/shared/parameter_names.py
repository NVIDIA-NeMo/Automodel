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

"""Utilities for stable logical parameter names."""

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX


def canonical_parameter_fqn(name: str) -> str:
    """Remove activation-checkpoint wrapper components from a parameter FQN.

    PyTorch exposes ``CheckpointWrapper`` as a registered child module through
    ``named_parameters()``, but strips the same component from state-dict keys.
    Canonicalizing name-sensitive consumers keeps their logical names aligned
    with the stable state-dict contract.

    Args:
        name: Module-qualified parameter name.

    Returns:
        The logical parameter name without checkpoint-wrapper components.
    """
    return name.replace(_CHECKPOINT_PREFIX, "")


# Module flags set by the packed-expert storage mixins (mxfp4 / nvfp4): their frozen base
# weights live in integer codes plus scale tensors whose storage dtype must never be cast.
PACKED_EXPERT_RESIDENT_ATTRS = ("_mxfp4_resident", "_nvfp4_resident")


def owns_packed_expert_storage(module: nn.Module) -> bool:
    """Whether ``module`` keeps its frozen expert weights in a packed (quantized) storage format."""
    return any(getattr(module, attr, False) for attr in PACKED_EXPERT_RESIDENT_ATTRS)


def packed_expert_parameter_names(module: nn.Module) -> set[str]:
    """Names of the packed-storage parameters of a packed-expert module (empty for any other module).

    The storage mixins publish them through ``packed_parameter_names()``. The codes and scales they name are the
    weight format itself, so dtype-reducing passes leave exactly these parameters alone and treat every other
    parameter of the module (adapters, biases) like those of any other module.
    """
    if not owns_packed_expert_storage(module):
        return set()
    names = getattr(module, "packed_parameter_names", None)
    return set(names()) if callable(names) else set()
