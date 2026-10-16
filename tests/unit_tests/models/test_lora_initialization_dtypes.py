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

"""CPU regressions for model-owned initialization after LoRA injection."""

import pytest
import torch

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.common import BackendConfig


@pytest.fixture(params=["glm_moe_dsa", "gemma4_dense"])
def model(request):
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    if request.param == "glm_moe_dsa":
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
        result = GlmMoeDsaForCausalLM(config, backend=backend)
    else:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

        from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

        config = Gemma4Config(
            text_config=Gemma4TextConfig(
                vocab_size=64,
                hidden_size=32,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                num_hidden_layers=2,
                intermediate_size=64,
                max_position_embeddings=32,
                enable_moe_block=False,
                layer_types=["full_attention", "sliding_attention"],
                sliding_window=16,
            )
        )
        result = Gemma4ForConditionalGeneration(config, backend=backend)
    return result.to(device="cpu", dtype=torch.bfloat16)


@pytest.mark.parametrize("lora_dtype", [torch.float32, torch.bfloat16, None])
def test_initialize_weights_preserves_adapter_storage(model, lora_dtype):
    apply_lora_to_linear_modules(
        model,
        PeftConfig(target_modules=["*"], lora_dtype=lora_dtype),
        skip_freeze=True,
    )
    adapters = {name: param for name, param in model.named_parameters() if "lora_" in name}
    assert adapters
    expected_dtype = torch.bfloat16 if lora_dtype is None else lora_dtype
    assert all(param.dtype == expected_dtype for param in adapters.values())
    before = {name: param.detach().clone() for name, param in adapters.items()}

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    after = dict(model.named_parameters())
    for name, param in adapters.items():
        assert after[name] is param
        assert param.dtype == expected_dtype
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
