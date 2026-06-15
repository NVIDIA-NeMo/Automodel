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

"""Regression test: KimiK25VL RoPE ``freqs_cis`` must stay fp32 after bf16 cast.

The DeepseekV3-based language backbone registers ``freqs_cis`` as a real-valued
fp32 buffer (inverse frequencies). ``from_pretrained`` casts the model to bf16 via
``cast_model_to_dtype``; the model lists ``"freqs_cis"`` in ``_keep_in_fp32_modules``
so the buffer is restored to fp32. This builds via ``from_config`` and applies the
same ``cast_model_to_dtype`` the fixed ``from_pretrained`` uses (``from_pretrained``
itself needs a real checkpoint dir for DCP). Runs on CPU.
"""

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.kimi_k25_vl.model import (
    KimiK25VLConfig,
    KimiK25VLForConditionalGeneration,
    MoonViT3dConfig,
)


def _tiny_text_config():
    return DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_rope_head_dim=16,  # freqs_cis length = qk_rope_head_dim // 2 = 8
        qk_nope_head_dim=16,
        v_head_dim=16,
        kv_lora_rank=16,
        q_lora_rank=None,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        max_position_embeddings=64,
        torch_dtype=torch.bfloat16,
    )


def _tiny_vision_config():
    return MoonViT3dConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=14,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        init_pos_emb_time=4,
        merge_kernel_size=[2, 2],
        merge_type="sd2_tpool",
    )


def test_bf16_cast_keeps_rope_freqs_cis_fp32():
    config = KimiK25VLConfig(vision_config=_tiny_vision_config(), text_config=_tiny_text_config())
    config.torch_dtype = torch.bfloat16

    model = KimiK25VLForConditionalGeneration.from_config(config, torch_dtype=torch.bfloat16)
    # Mirror the fixed from_pretrained cast (which routes through cast_model_to_dtype).
    cast_model_to_dtype(model, torch.bfloat16)

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    assert rope_buffers, "no inv_freq/freqs_cis buffers found on the model"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    assert model.model.language_model.model.embed_tokens.weight.dtype == torch.bfloat16
