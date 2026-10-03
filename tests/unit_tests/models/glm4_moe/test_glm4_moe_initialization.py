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

import torch
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm4_moe.layers import Glm4MoeAttention


def test_attention_initializes_projection_biases_after_meta_materialization() -> None:
    config = Glm4MoeConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        use_qk_norm=True,
        partial_rotary_factor=0.5,
        attention_bias=True,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )

    with torch.device("meta"):
        attention = Glm4MoeAttention(config, backend)

    attention.to_empty(device="cpu")
    projections = [attention.q_proj, attention.k_proj, attention.v_proj]
    with torch.no_grad():
        for projection in projections:
            assert projection.bias is not None
            projection.bias.fill_(float("nan"))

    attention.init_weights(torch.device("cpu"), init_std=0.02)

    assert all(torch.isfinite(parameter).all() for parameter in attention.parameters())
    for projection in projections:
        assert projection.bias is not None
        assert torch.count_nonzero(projection.bias).item() == 0
