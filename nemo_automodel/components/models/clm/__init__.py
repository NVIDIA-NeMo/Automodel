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

"""Contrastive Language Model (CLM) decision heads (https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)."""

from nemo_automodel.components.models.clm.heads import (
    CLMHeadConfig,
    CLMHeadPair,
    EmbeddingTable,
    load_clm_heads,
    save_clm_heads,
)

__all__ = ["CLMHeadConfig", "CLMHeadPair", "EmbeddingTable", "load_clm_heads", "save_clm_heads"]
