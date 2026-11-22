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

"""GLM-MoE-DSA initialization must preserve the PEFT storage policy."""

import pytest
import torch

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding


@pytest.mark.parametrize("init_dtype", [torch.float32, torch.bfloat16])
def test_glm_moe_dsa_initialization_preserves_frozen_and_adapter_storage(init_dtype):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

    config = GlmMoeDsaConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        max_position_embeddings=32,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "sparse"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    model = GlmMoeDsaForCausalLM(
        config,
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    ).float()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*"], lora_dtype=torch.float32))
    prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
    adapters = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
    assert any("experts.lora_" in name for name in adapters)
    assert any("lora_A.weight" in name for name in adapters)

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=init_dtype)

    for name, parameter in model.named_parameters():
        assert parameter.dtype == (torch.float32 if parameter.requires_grad else torch.bfloat16), name
        assert torch.isfinite(parameter).all(), name
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, adapters[name], rtol=0, atol=0)
