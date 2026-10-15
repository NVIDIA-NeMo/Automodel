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

"""Optional Step-5 kernel choices."""

from dataclasses import dataclass

from nemo_automodel.components.models.common import BackendConfig


@dataclass(kw_only=True)
class Step5BackendConfig(BackendConfig):
    """Model-owned execution options, disabled by default.

    Attributes:
        csa_forward_full_blocks: Skip custom masks on proven fully selected
            forward blocks for single-document, unpadded B1 CP1 inputs on SM90
            with head dimension 192. Backward block lists stay unchanged.
            Requires attn="fa4".
        csa_backward_full_blocks: Skip custom masks on proven complete backward
            blocks under the same single-document guard. May change FP32
            dK/dV accumulation order. Requires attn="fa4".
        swa_varlen: Use native FA4 window attention for valid single-batch
            SM90 D192 query prefixes. Padding keeps its custom-mask path.
            Requires enabled sparse metadata; communication is unchanged.
    """

    csa_forward_full_blocks: bool = False
    csa_backward_full_blocks: bool = False
    swa_varlen: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.csa_forward_full_blocks, bool):
            raise TypeError("csa_forward_full_blocks must be a bool")
        if self.csa_forward_full_blocks and self.attn != "fa4":
            raise ValueError("csa_forward_full_blocks requires attn=fa4")
        if not isinstance(self.csa_backward_full_blocks, bool):
            raise TypeError("csa_backward_full_blocks must be a bool")
        if self.csa_backward_full_blocks and self.attn != "fa4":
            raise ValueError("csa_backward_full_blocks requires attn=fa4")
        if not isinstance(self.swa_varlen, bool):
            raise TypeError("swa_varlen must be a bool")
