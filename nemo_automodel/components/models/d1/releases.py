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

"""Pinned publisher sources and execution precision for supported native D1 releases."""

from typing import NamedTuple

import torch


class D1Release(NamedTuple):
    """Identify one immutable publisher implementation and checkpoint."""

    repository: str
    revision: str
    reference_class: str


RELEASES: dict[str, D1Release] = {
    "d1-3b": D1Release("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432", "modeling_d1.D1Model"),
    "d1-omni": D1Release(
        "LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674", "modeling_d1.D1OmniModel"
    ),
}
# Omni's publisher recommends float32/float16 and warns against bfloat16.
RELEASE_DTYPES: dict[str, torch.dtype] = {"d1-3b": torch.bfloat16, "d1-omni": torch.float32}
