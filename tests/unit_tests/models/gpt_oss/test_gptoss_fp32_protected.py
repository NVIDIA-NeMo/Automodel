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
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.model import GptOssForCausalLM


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


def _config() -> GptOssConfig:
    return GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "sliding_attention"],
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 4096,
        },
        torch_dtype=torch.bfloat16,
    )


def test_cast_bf16_keeps_flex_attention_sinks_fp32():
    with patch.object(torch.cuda, "current_device", return_value=0):
        model = GptOssForCausalLM(_config(), backend=_backend())

    expected = {}
    for name, param in model.named_parameters():
        if name.endswith("self_attn.sinks"):
            values = torch.linspace(0.00123, 0.00456, param.numel(), dtype=torch.float32).reshape_as(param)
            param.data.copy_(values)
            expected[name] = values

    assert expected, "GPT-OSS flex attention should create sink parameters"

    cast_model_to_dtype(model, torch.bfloat16)

    params = dict(model.named_parameters())
    for name, values in expected.items():
        assert params[name].dtype == torch.float32
        torch.testing.assert_close(params[name], values)
    assert model.lm_head.weight.dtype == torch.bfloat16
