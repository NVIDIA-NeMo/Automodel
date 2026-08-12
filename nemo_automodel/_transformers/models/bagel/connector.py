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

"""Projection layers that connect BAGEL vision features to text hidden states."""

from __future__ import annotations

from collections import OrderedDict

from torch import nn
from transformers.activations import ACT2FN


class _Activation(nn.Module):
    """Wrap a Transformers activation callable in an ``nn.Module``."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.fn = ACT2FN[name]

    def forward(self, hidden_states):
        return self.fn(hidden_states)


class BagelMultiModalProjector(nn.Sequential):
    """Project vision-tower features into the language-model hidden size."""

    def __init__(self, in_dim: int, out_dim: int, hidden_act: str) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_dim, out_dim)),
                    ("activation", _Activation(hidden_act)),
                    ("fc2", nn.Linear(out_dim, out_dim)),
                ]
            )
        )
