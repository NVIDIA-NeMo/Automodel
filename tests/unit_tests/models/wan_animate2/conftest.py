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

"""Small real upstream models and cached triplet inputs; no model downloads."""

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.models.wan_animate2.adapter import WanAnimate2Adapter


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


@pytest.fixture
def training_batch():
    torch.manual_seed(11)
    latents = torch.randn(1, 16, 2, 4, 4)
    return {
        "video_latents": latents,
        "reference_latents": torch.randn(1, 16, 1, 4, 4),
        "driving_latents": torch.randn_like(latents),
        "cond_zero_latents": torch.randn_like(latents),
        "clip_fea": torch.randn(1, 3, 1280),
        "clip_fea_ref": torch.randn(1, 3, 1280),
        "text_embeddings": torch.randn(1, 5, 16),
        "prompt_ref_embeddings": torch.randn(1, 4, 16),
    }


@pytest.fixture
def training_inputs(training_batch):
    """Return two-stream inputs with a supervised reference slot."""
    batch = training_batch
    latents = WanAnimate2Adapter().prepare_latents(batch["video_latents"], batch)
    context = FlowMatchingContext(
        batch=batch,
        latents=latents,
        noisy_latents=torch.randn_like(latents),
        task_type="i2v",
        data_type="video",
        timesteps=torch.tensor([500.0]),
        sigma=torch.tensor([0.5]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return WanAnimate2Adapter().prepare_inputs(context)
