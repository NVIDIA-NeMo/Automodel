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

"""Qwen3.5 SDPA vision outputs and gradients with host sequence metadata."""

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model as HFQwen3_5Model

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration


def _tiny_model() -> Qwen3_5ForConditionalGeneration:
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        layer_types=["full_attention"],
        attn_implementation="sdpa",
    )
    vision_config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
    )
    config = Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False, dispatcher="torch")
    return Qwen3_5ForConditionalGeneration(config, backend=backend).train()


def test_sdpa_host_seqlens_matches_hf_vision_forward_and_backward() -> None:
    torch.manual_seed(42)
    model = _tiny_model().model
    grid = torch.tensor([[1, 2, 2], [2, 2, 2]], dtype=torch.long)
    pixels = torch.randn(12, 12)

    def run(reference: bool):
        model.zero_grad(set_to_none=True)
        patches = pixels.detach().clone().requires_grad_()
        if reference:
            output = HFQwen3_5Model.get_image_features(model, patches, grid, return_dict=True)
        else:
            output = model.get_image_features(patches, grid, return_dict=True)
        features = tuple(value.detach().clone() for value in output.pooler_output)
        sum(value.square().sum() for value in output.pooler_output).backward()
        return (
            features,
            patches.grad.detach().clone(),
            tuple(block.attn.qkv.weight.grad.detach().clone() for block in model.visual.blocks),
        )

    reference_features, reference_input_grad, reference_weight_grads = run(reference=True)
    actual_features, actual_input_grad, actual_weight_grads = run(reference=False)

    assert [value.shape for value in actual_features] == [torch.Size([4, 16]), torch.Size([8, 16])]
    for reference, actual in zip(reference_features, actual_features):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_input_grad, reference_input_grad, rtol=0, atol=0)
    for reference, actual in zip(reference_weight_grads, actual_weight_grads):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
