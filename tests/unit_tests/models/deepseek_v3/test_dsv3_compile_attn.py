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

import copy
from unittest.mock import Mock, patch

import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.layers import MLA


def _model_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        max_position_embeddings=16,
    )


@patch("nemo_automodel.components.models.deepseek_v3.layers.torch.compile")
@patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
@patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
@patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
def test_compile_attn_binds_after_pipeline_copy(
    mock_init_attn: Mock,
    mock_init_rms: Mock,
    mock_init_linear: Mock,
    mock_compile: Mock,
) -> None:
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        compile_attn=True,
    )
    mock_init_linear.return_value = torch.nn.Identity()
    mock_init_rms.return_value = torch.nn.Identity()
    mock_init_attn.return_value = (None, Mock())

    mla = MLA(_model_config(), backend)
    stage_mla = copy.deepcopy(mla)
    expected = torch.tensor(1.0)
    compiled_forward = Mock(return_value=expected)
    mock_compile.return_value = compiled_forward

    assert mock_compile.call_count == 0
    assert stage_mla(torch.empty(1), torch.empty(1)) is expected
    assert mock_compile.call_count == 1
    assert mock_compile.call_args.args[0].__self__ is stage_mla

    assert stage_mla(torch.empty(1), torch.empty(1)) is expected
    assert mock_compile.call_count == 1
