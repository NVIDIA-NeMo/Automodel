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

"""Independent persistent correction bias for the shared MTP router call."""

from __future__ import annotations

import torch
from torch import nn


class Ernie5_5MTPRouterState(nn.Module):
    """Own the frozen FP32 selection bias of shape [experts].

    Shared router weights belong to the decoder and are never registered here.
    The admitted SFT policy does not update this bias or accumulate MTP loads.
    """

    def __init__(self, experts: int) -> None:
        super().__init__()
        self.register_buffer("e_score_correction_bias", torch.zeros(experts, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize after construction or meta-device materialization."""
        self.e_score_correction_bias.zero_()
