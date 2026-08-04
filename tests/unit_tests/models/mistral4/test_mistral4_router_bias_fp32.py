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

"""Regression tests for Mistral4 adaptive routing-bias precision."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
from nemo_automodel.components.models.mistral4.model import (
    _HF_MISTRAL3_AVAILABLE,
    Mistral4ForCausalLM,
)


def _tiny_text_config() -> Mistral4Config:
    return Mistral4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        kv_lora_rank=2,
        q_lora_rank=None,
        qk_rope_head_dim=2,
        v_head_dim=2,
        qk_nope_head_dim=2,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        torch_dtype=torch.float32,
    )


def _torch_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _assert_router_biases_are_fp32(model: torch.nn.Module) -> None:
    biases = [(name, buffer) for name, buffer in model.named_buffers() if name.endswith("e_score_correction_bias")]

    assert biases, "Mistral4 should create adaptive routing-bias buffers"
    for name, bias in biases:
        assert bias.dtype == torch.float32, f"routing bias {name} was cast to {bias.dtype}"
        torch.testing.assert_close(bias, torch.zeros_like(bias))


def test_text_initialize_weights_bf16_keeps_router_bias_fp32() -> None:
    model = Mistral4ForCausalLM(_tiny_text_config(), backend=_torch_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    _assert_router_biases_are_fp32(model)
    assert model.lm_head.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not _HF_MISTRAL3_AVAILABLE, reason="transformers Mistral3 model is unavailable")
def test_multimodal_initialize_weights_bf16_keeps_router_bias_fp32() -> None:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    from nemo_automodel.components.models.mistral4.model import Mistral3ForConditionalGeneration

    config = Mistral3Config(
        text_config=_tiny_text_config().to_dict(),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_channels": 3,
            "image_size": 4,
            "patch_size": 2,
        },
        image_token_index=10,
        spatial_merge_size=2,
    )
    model = Mistral3ForConditionalGeneration(config, backend=_torch_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    _assert_router_biases_are_fp32(model)
    assert model.lm_head.weight.dtype == torch.bfloat16
