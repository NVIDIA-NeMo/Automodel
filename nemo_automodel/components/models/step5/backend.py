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

"""Step-5 backend options with conservative distributed memory defaults."""

from dataclasses import dataclass

from nemo_automodel.components.models.common import BackendConfig


@dataclass(kw_only=True)
class Step5BackendConfig(BackendConfig):
    """Model-owned retention budget for existing BF16 decoder FSDP units.

    Attributes:
        bf16_fsdp_retention_mib_per_part: Maximum padded unsharded parameter
            payload retained until backward in each local pipeline part, in
            MiB. Zero disables retention. With two local parts the combined
            bound is twice this value. Embeddings, heads, routed experts and
            FP32 groups retain their configured policy. This bounds weights,
            not total CUDA memory, activations or collective workspaces.
    """

    bf16_fsdp_retention_mib_per_part: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.bf16_fsdp_retention_mib_per_part, bool) or not isinstance(
            self.bf16_fsdp_retention_mib_per_part, int
        ):
            raise TypeError("bf16_fsdp_retention_mib_per_part must be an int")
        if self.bf16_fsdp_retention_mib_per_part < 0:
            raise ValueError("bf16_fsdp_retention_mib_per_part must be nonnegative")
