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

from unittest.mock import patch

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.hy_v3.config import HYV3Config
from nemo_automodel.components.models.hy_v3.model import HYV3ForCausalLM


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        gate_precision="float32",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        enable_fsdp_optimizations=False,
    )


def _config() -> HYV3Config:
    return HYV3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=0,
        first_k_dense_replace=1,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
    )


def test_cast_bf16_keeps_router_correction_bias_fp32():
    with patch.object(torch.cuda, "current_device", return_value=0):
        model = HYV3ForCausalLM(_config(), backend=_backend())

    expected = {}
    for name, buf in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            values = torch.linspace(0.00123, 0.00456, buf.numel(), dtype=torch.float32).reshape_as(buf)
            buf.copy_(values)
            expected[name] = values

    assert expected, "HY V3 should create router correction-bias buffers"

    cast_model_to_dtype(model, torch.bfloat16)

    buffers = dict(model.named_buffers())
    for name, values in expected.items():
        assert buffers[name].dtype == torch.float32
        torch.testing.assert_close(buffers[name], values)
    assert model.lm_head.weight.dtype == torch.bfloat16
