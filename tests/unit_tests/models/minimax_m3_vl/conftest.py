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
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig

# Tiny config (~layers 0 dense, 1-2 MoE) used across the Stage-1 M3 unit tests.
TINY_CFG = dict(
    hidden_size=64,
    intermediate_size=32,
    dense_intermediate_size=48,
    shared_intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rotary_dim=8,
    partial_rotary_factor=0.5,
    vocab_size=128,
    max_position_embeddings=256,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    num_local_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    moe_layer_freq=[0, 1, 1],
    use_gemma_norm=True,
    use_qk_norm=True,
    qk_norm_type="per_head",
    scoring_func="sigmoid",
    use_routing_bias=True,
    routed_scaling_factor=2.0,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
)


@pytest.fixture
def text_config():
    return MiniMaxM3VLTextConfig(torch_dtype="float32", **TINY_CFG)


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


@pytest.fixture
def model(text_config, backend):
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM

    m = MiniMaxM3SparseForCausalLM(text_config, backend=backend).eval()
    m.initialize_weights(dtype=torch.float32)
    return m
