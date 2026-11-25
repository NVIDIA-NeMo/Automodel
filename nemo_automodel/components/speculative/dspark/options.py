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

"""Options for constructing a model-owned DSpark draft."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DSparkDraftOptions:
    """Recipe options consumed by model-owned DSpark draft constructors."""

    num_draft_layers: int
    target_layer_ids: list[int]
    block_size: int
    num_anchors: int
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    confidence_head_alpha: float
    confidence_head_with_markov: bool
    confidence_head_stop_gradient: bool = False
