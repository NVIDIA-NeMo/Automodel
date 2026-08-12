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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig, KimiK3VisionConfig
from nemo_automodel.components.models.kimi_k3.multimodal import KimiK3ForConditionalGeneration


def _tiny_vlm() -> KimiK3ForConditionalGeneration:
    text_config = KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=2,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=None,
    )
    vision_config = KimiK3VisionConfig(
        patch_size=2,
        init_pos_emb_height=4,
        init_pos_emb_width=2,
        init_pos_emb_time=1,
        vt_num_attention_heads=2,
        vt_num_hidden_layers=1,
        vt_hidden_size=16,
        vt_intermediate_size=32,
        merge_kernel_size=(2, 2),
        attn_implementation="eager",
        mm_projector_type="patchmergerv2",
        mm_hidden_size=16,
        qkv_hidden_size=16,
        norm_type="rmsnorm",
        text_hidden_size=text_config.hidden_size,
    )
    config = KimiK3Config(
        text_config=text_config,
        vision_config=vision_config,
        media_placeholder_token_id=63,
        pad_token_id=0,
    )
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = KimiK3ForConditionalGeneration(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_k3_vlm_forward_expands_image_placeholder():
    torch.manual_seed(23)
    model = _tiny_vlm()
    input_ids = torch.tensor([[1, 63, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[1, -100, 2, -100]])
    pixel_values = torch.randn(8, 3, 2, 2)
    grid_thws = torch.tensor([[1, 4, 2]])

    with torch.no_grad():
        image_features = model._extract_image_features(pixel_values, grid_thws)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        merged_embeddings, merged_attention_mask, merged_labels, position_ids = (
            model._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        )
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels,
        )

    assert [features.shape for features in image_features] == [(2, 32)]
    assert merged_embeddings.shape == (1, 5, 32)
    assert merged_attention_mask.tolist() == [[1, 1, 1, 1, 0]]
    assert merged_labels is not None
    assert merged_labels.tolist() == [[1, -100, -100, 2, -100]]
    assert position_ids.tolist() == [[0, 1, 2, 3, 1]]
    torch.testing.assert_close(merged_embeddings[:, 1:3], image_features[0].unsqueeze(0))
    assert output.logits.shape == (1, 5, 64)
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None and torch.isfinite(output.loss)


def test_k3_vlm_rejects_placeholder_feature_count_mismatch():
    model = _tiny_vlm()
    input_ids = torch.tensor([[1, 63, 2]])
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = [torch.randn(2, 32), torch.randn(1, 32)]

    with pytest.raises(ValueError, match="Received 1 image placeholders for 2 image feature groups"):
        model._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids,
            torch.ones_like(input_ids),
        )
