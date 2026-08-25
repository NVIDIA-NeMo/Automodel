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

"""Real-CUDA functional coverage for Qwen image-edit flow matching."""

from __future__ import annotations

import importlib.util

import pytest
import torch

_DIFFUSERS_AVAILABLE = importlib.util.find_spec("diffusers") is not None

pytestmark = [
    pytest.mark.skipif(not _DIFFUSERS_AVAILABLE, reason="diffusers is not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def test_qwen_image_edit_flow_matching_updates_model() -> None:
    """Run production input packing, flow matching, backward, and AdamW on CUDA."""
    from diffusers import QwenImageTransformer2DModel

    from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline
    from nemo_automodel._diffusers.models.qwen_image_edit.adapter import QwenImageEditAdapter

    torch.manual_seed(123)
    device = torch.device("cuda")
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pipeline = FlowMatchingPipeline(
        model_adapter=QwenImageEditAdapter(),
        timestep_sampling="uniform",
        flow_shift=1.0,
        cfg_dropout_prob=0.0,
        mix_uniform_ratio=0.0,
        use_loss_weighting=False,
        device=device,
    )
    batch = {
        "image_latents": torch.randn((1, 4, 4, 4)),
        "context_latents": [torch.randn((1, 4, 4, 4))],
        "text_embeddings": torch.randn((1, 5, 12)),
        "text_attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "data_type": "image",
    }

    parameters = dict(model.named_parameters())
    required_gradients = (
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.add_q_proj.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
    )
    initial_parameter = parameters[required_gradients[0]].detach().clone()

    weighted_loss, loss, loss_mask, metrics = pipeline.step(
        model,
        batch,
        device=device,
        dtype=torch.float32,
        global_step=1,
        collect_metrics=False,
    )

    assert weighted_loss.shape == batch["image_latents"].shape
    assert loss_mask is None
    assert metrics == {}
    assert torch.isfinite(loss) and loss > 0

    loss.backward()
    for name in required_gradients:
        gradient = parameters[name].grad
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all(), f"non-finite gradient for {name}"
    assert torch.linalg.vector_norm(parameters[required_gradients[0]].grad) > 0

    optimizer.step()
    assert not torch.equal(parameters[required_gradients[0]], initial_parameter)
