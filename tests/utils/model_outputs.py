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

"""Checks on what a model forward returns, as FSDP2's root module sees it."""

import torch
from torch.utils._pytree import tree_leaves


def leaf_tensors_requiring_grad(output: object) -> list[torch.Tensor]:
    """Find the leaf tensors that require grad in a forward's output, such as a returned parameter.

    FSDP2's root module flattens its forward's output the same way and registers a backward hook on
    every tensor in it that requires grad. A hook on a leaf tensor is never removed, so each forward
    that returns one leaves another hook behind.

    Args:
        output: The forward's return value: tensors of any shape, alone or nested in a
            ``ModelOutput``, dict, list or tuple.

    Returns:
        Those leaf tensors, each in its own shape; empty when the output holds none.
    """
    return [t for t in tree_leaves(output) if isinstance(t, torch.Tensor) and t.is_leaf and t.requires_grad]
