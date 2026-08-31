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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4.config import HyV4Config
from nemo_automodel.components.models.hy_v4.model import HyV4ForCausalLM


@pytest.fixture
def tiny_hy_v4_config() -> HyV4Config:
    return HyV4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=24,
        moe_intermediate_size=4,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.25,
        n_group=1,
        topk_group=1,
        mlp_layer_types=["dense", "sparse", "sparse"],
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_topk=3,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full", "full", "shared"],
        hc_mult=2,
        num_nextn_predict_layers=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype=torch.bfloat16,
    )


@pytest.fixture
def tiny_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="cudnn",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        gate_precision=torch.float32,
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
    )


@pytest.fixture
def tiny_hy_v4_model(tiny_hy_v4_config, tiny_backend) -> HyV4ForCausalLM:
    torch.manual_seed(1234)
    model = HyV4ForCausalLM(tiny_hy_v4_config, backend=tiny_backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    return model
