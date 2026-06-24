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
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_omni_moe.model import Qwen3OmniMoeThinkerTextModel


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _config() -> Qwen3MoeConfig:
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=5000.0,
        router_aux_loss_coef=0.0,
        use_sliding_window=False,
    )
    cfg.torch_dtype = "float32"
    return cfg


def test_cast_bf16_keeps_rotary_frequency_buffers_fp32():
    model = Qwen3OmniMoeThinkerTextModel(_config(), backend=_backend())
    expected = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if name.endswith(("inv_freq", "original_inv_freq"))
    }

    assert expected, "Qwen3-Omni text rotary should create frequency buffers"

    cast_model_to_dtype(model, torch.bfloat16)

    buffers = dict(model.named_buffers())
    for name, values in expected.items():
        assert buffers[name].dtype == torch.float32
        torch.testing.assert_close(buffers[name], values)
    assert model.embed_tokens.weight.dtype == torch.bfloat16
