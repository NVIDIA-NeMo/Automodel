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

"""Small real upstream models; no model downloads."""

import pytest
import torch


@pytest.fixture
def tiny_model():
    # Diffusers is optional in CPU-only installs; model tests need the release
    # that first provides Wan-Animate-2. Pure preprocessing tests remain active.
    diffusers = pytest.importorskip("diffusers", minversion="0.40.0")
    torch.manual_seed(7)
    return diffusers.WanAnimate2Transformer3DModel(
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=16,
        text_len=8,
        num_heads=2,
        num_layers=2,
        use_img_emb=True,
        cross_attn_norm=True,
    )
