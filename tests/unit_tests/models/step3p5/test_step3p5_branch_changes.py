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

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.layers import Step3p5Attention, Step3p5RotaryEmbedding
from nemo_automodel.components.models.step3p5.model import (
    Step3p5Model,
    _keep_step_router_bias_fp32,
    parse_moe_layers_enum,
)


@dataclass
class TinyStepConfig:
    vocab_size: int = 16
    hidden_size: int = 8
    intermediate_size: int = 16
    num_hidden_layers: int = 0
    num_attention_heads: int = 2
    num_attention_groups: int = 1
    max_position_embeddings: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict | None = None
    partial_rotary_factors: list | None = None
    layer_types: list | None = None
    attention_other_setting: dict | None = None
    sliding_window: int | None = None
    use_head_wise_attn_gate: bool = False
    use_rope_layers: list | None = None
    head_dim: int = 4
    attention_bias: bool = False
    torch_dtype: str = "float32"
    moe_layers_enum: tuple = ()
    moe_num_experts: int = 2
    moe_top_k: int = 1
    moe_intermediate_size: int = 4
    moe_router_activation: str = "softmax"
    moe_router_scaling_factor: float = 1.0
    use_moe_router_bias: bool = False
    share_expert_dims: int = 4
    swiglu_limits: list | None = None
    swiglu_limits_shared: list | None = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}
        if self.layer_types is None:
            self.layer_types = ["full_attention"]
        if self.attention_other_setting is None:
            self.attention_other_setting = {"num_attention_heads": 2, "num_attention_groups": 1}
        if self.swiglu_limits is None:
            self.swiglu_limits = [None]
        if self.swiglu_limits_shared is None:
            self.swiglu_limits_shared = [None]


def tiny_backend(**kwargs):
    values = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    values.update(kwargs)
    return BackendConfig(**values)


def test_parse_moe_layers_enum_accepts_int_and_rejects_unknown():
    assert parse_moe_layers_enum(2, num_hidden_layers=4) == {2}
    with pytest.raises(ValueError, match="Unsupported"):
        parse_moe_layers_enum(object(), num_hidden_layers=4)


def test_keep_step_router_bias_fp32_updates_bias_and_master():
    class GateLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.e_score_correction_bias = nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
            self.e_score_correction_bias_master = nn.Parameter(
                torch.ones(2, dtype=torch.bfloat16),
                requires_grad=False,
            )

    module = nn.Sequential(GateLike())
    _keep_step_router_bias_fp32(module)

    assert module[0].e_score_correction_bias.dtype == torch.float32
    assert module[0].e_score_correction_bias_master.dtype == torch.float32


def test_rotary_embedding_apply_recomputes_inv_freq_on_target_device():
    rotary = Step3p5RotaryEmbedding(TinyStepConfig(), layer_idx=0)
    old_inv_freq = rotary.inv_freq.clone()
    result = rotary.to(dtype=torch.bfloat16)
    assert result is rotary
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, old_inv_freq)


def test_attention_accepts_position_ids_instead_of_freqs_and_errors_without_either():
    config = TinyStepConfig(num_hidden_layers=1)
    attention = Step3p5Attention(config, layer_idx=0, backend=tiny_backend())
    x = torch.randn(1, 3, config.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(3).unsqueeze(0)
    out = attention(x, position_ids=position_ids)
    assert out.shape == x.shape

    with pytest.raises(ValueError, match="requires freqs_cis or position_ids"):
        attention(x)


def test_step3p5_model_accepts_inputs_embeds_and_float_input_ids_without_embeddings():
    model = Step3p5Model(TinyStepConfig(), tiny_backend())
    model.norm = nn.Identity()
    inputs_embeds = torch.randn(1, 3, 8)
    out = model(inputs_embeds=inputs_embeds)
    torch.testing.assert_close(out, inputs_embeds)

    model_with_embeddings = Step3p5Model(TinyStepConfig(), tiny_backend())
    model_with_embeddings.norm = nn.Identity()
    out_from_ids = model_with_embeddings(input_ids=torch.tensor([[1, 2, 3]]))
    assert out_from_ids.shape == (1, 3, 8)

    model.embed_tokens = None
    out_from_float_ids = model(input_ids=inputs_embeds)
    torch.testing.assert_close(out_from_float_ids, inputs_embeds)

    with pytest.raises(ValueError, match="requires input_ids or inputs_embeds"):
        model()
    with pytest.raises(ValueError, match="inputs_embeds must be provided"):
        model(input_ids=torch.ones(1, 3, dtype=torch.long))


def test_step3p5_model_router_bias_config_uses_sigmoid_with_bias_and_force_flag():
    config = TinyStepConfig(
        num_hidden_layers=1,
        moe_layers_enum=(0,),
        use_moe_router_bias=True,
        moe_router_activation="sigmoid",
    )
    model = Step3p5Model(config, tiny_backend())

    assert model.moe_config.score_func == "sigmoid_with_bias"
    assert model.moe_config.router_bias is False
    assert model.moe_config.force_e_score_correction_bias is True
