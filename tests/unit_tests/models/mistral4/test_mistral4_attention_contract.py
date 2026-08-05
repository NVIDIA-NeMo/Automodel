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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
from nemo_automodel.components.models.mistral4.model import Mistral4MLA, _get_llama_4_attn_scale


def _tiny_long_context_config() -> Mistral4Config:
    return Mistral4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        kv_lora_rank=2,
        q_lora_rank=None,
        qk_rope_head_dim=2,
        v_head_dim=2,
        qk_nope_head_dim=2,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        max_position_embeddings=1048576,
        torch_dtype=torch.float32,
    )


def _torch_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def test_mistral4_uses_standard_qk_attention_scale() -> None:
    config = _tiny_long_context_config()
    attention = Mistral4MLA(config, _torch_backend())
    expected_scale = config.qk_head_dim**-0.5

    assert attention.softmax_scale == pytest.approx(expected_scale)

    query = torch.zeros(1, 1, 1, config.qk_head_dim)
    key = torch.zeros_like(query)
    value = torch.zeros(1, 1, 1, config.v_head_dim)
    with patch("nemo_automodel.components.attention.utils.F.scaled_dot_product_attention") as sdpa:
        sdpa.return_value = torch.zeros_like(value)
        attention.attn_func(query, key, value)

    assert sdpa.call_args.kwargs["scale"] == pytest.approx(expected_scale)


def test_mistral4_long_context_scale_applies_to_full_query() -> None:
    config = _tiny_long_context_config()
    attention = Mistral4MLA(config, _torch_backend())
    captured_queries: list[torch.Tensor] = []

    def capture_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Capture SDPA inputs while returning a shape-compatible result.

        Args:
            query: Tensor of shape [batch, heads, sequence, qk_head_dim].
            key: Tensor of shape [batch, heads, sequence, qk_head_dim].
            value: Tensor of shape [batch, heads, sequence, v_head_dim].
            **kwargs: Attention metadata unused by this test stand-in.

        Returns:
            Zero tensor of shape [batch, heads, sequence, v_head_dim].
        """
        del key, kwargs
        captured_queries.append(query.detach().clone())
        return torch.zeros_like(value)

    attention.attn_func = capture_attention
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.q_proj.weight[: config.qk_head_dim, : config.qk_head_dim].copy_(torch.eye(config.qk_head_dim))

    hidden_states = torch.ones(1, 1, config.hidden_size)
    freqs_cis = torch.ones(1, 1, config.qk_rope_head_dim // 2, dtype=torch.complex64)
    original_max_position = config.rope_parameters["original_max_position_embeddings"]
    low_position_ids = torch.zeros(1, 1, dtype=torch.long)
    high_position_ids = torch.full((1, 1), original_max_position, dtype=torch.long)

    attention(hidden_states, freqs_cis, position_ids=low_position_ids)
    attention(hidden_states, freqs_cis, position_ids=high_position_ids)

    expected_scale = _get_llama_4_attn_scale(
        high_position_ids,
        config.rope_parameters["llama_4_scaling_beta"],
        original_max_position,
    ).item()
    torch.testing.assert_close(captured_queries[1], captured_queries[0] * expected_scale)
