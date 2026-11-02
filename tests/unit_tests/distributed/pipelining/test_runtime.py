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

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from nemo_automodel.components.distributed.pipelining.runtime import collect_pipeline_runtime_initializers


@dataclass
class _Initializer:
    resource_key: object
    signature: object

    def prepare(self, *, num_tokens: int, device: torch.device) -> None:
        pass


class _Provider(nn.Module):
    def __init__(self, *initializers):
        super().__init__()
        self.initializers = initializers

    def get_pipeline_runtime_initializers(self):
        return self.initializers


def test_collector_deduplicates_compatible_shared_resource_and_preserves_order():
    first = _Initializer(("backend", "shared"), ("config", 1))
    duplicate = _Initializer(("backend", "shared"), ("config", 1))
    other = _Initializer(("backend", "other"), ("config", 2))
    part = nn.Module()
    part.first = _Provider(first)
    part.duplicate = _Provider(duplicate)
    part.other = _Provider(other)

    assert collect_pipeline_runtime_initializers([part]) == [first, other]


def test_collector_rejects_conflict_with_provider_locations():
    part = nn.Module()
    part.first = _Provider(_Initializer("shared", "signature-a"))
    part.second = _Provider(_Initializer("shared", "signature-b"))

    with pytest.raises(RuntimeError, match=r"model_parts\[0\]\.first.*model_parts\[0\]\.second"):
        collect_pipeline_runtime_initializers([part])


def test_collector_requires_hashable_resource_contract():
    part = _Provider(_Initializer([], "signature"))

    with pytest.raises(TypeError, match="keys and signatures must be hashable"):
        collect_pipeline_runtime_initializers([part])
