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

"""Combined projection modules for efficient transformer implementations."""

from nemo_automodel.components.models.common.combined_projection.combined_mlp import CombinedGateUpMLP
from nemo_automodel.components.models.common.combined_projection.combined_qkv import CombinedQKVAttentionMixin

__all__ = ["CombinedQKVAttentionMixin", "CombinedGateUpMLP"]
