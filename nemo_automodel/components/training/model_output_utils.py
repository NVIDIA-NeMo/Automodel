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

"""Utilities for working with model output objects.

HuggingFace `ModelOutput` types typically store `hidden_states` as a tuple of tensors
(`tuple[Tensor, ...]`) when `output_hidden_states=True`.

However, some custom models may store the *final* hidden state tensor directly in the
`hidden_states` field (i.e., a single `[B, T, H]` tensor) to reduce memory/overhead.

Downstream training code should be robust to both representations.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    # DTensor is not a torch.Tensor subclass; treat it as tensor-like for extraction.
    from torch.distributed.tensor import DTensor  # type: ignore

    _TENSOR_LIKE = (torch.Tensor, DTensor)
except Exception:  # pragma: no cover
    DTensor = None  # type: ignore[assignment]
    _TENSOR_LIKE = (torch.Tensor,)


def get_final_hidden_states(model_output: Any) -> Optional[Any]:
    """Return the final hidden-states tensor from a HF-like model output.

    Supports both common layouts:
    - `hidden_states` is a tuple/list of tensors (HF default) → return last non-None entry
    - `hidden_states` is a single tensor-like object → return it as-is

    Args:
        model_output: A HF `ModelOutput`-like object, or a plain dict with a `hidden_states` key.

    Returns:
        The final hidden states tensor-like object, or None if not present.
    """
    if model_output is None:
        return None

    if isinstance(model_output, dict):
        hidden_states = model_output.get("hidden_states", None)
    else:
        hidden_states = getattr(model_output, "hidden_states", None)

    if hidden_states is None:
        return None

    if isinstance(hidden_states, _TENSOR_LIKE):
        return hidden_states

    if isinstance(hidden_states, (tuple, list)):
        # Some models may include None entries; select the last non-None entry.
        for hs in reversed(hidden_states):
            if hs is None:
                continue
            if not isinstance(hs, _TENSOR_LIKE):
                raise TypeError(f"Expected hidden_states entries to be tensor-like, got {type(hs)}")
            return hs
        return None

    raise TypeError(
        f"Unexpected hidden_states type {type(hidden_states)}; expected a tensor-like object or a tuple/list of them."
    )
