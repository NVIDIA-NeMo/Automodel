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

"""Reference parity for the dense Qwen3 implementation."""

import pytest
import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as HFQwen3ForCausalLM

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3.model import Qwen3Attention, Qwen3ForCausalLM


def _tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return config


def test_qwen3_bshd_matches_huggingface_logits_and_gradients():
    """The custom BSHD path must preserve HuggingFace forward and backward behavior."""
    torch.manual_seed(1234)
    config = _tiny_config()
    reference_model = HFQwen3ForCausalLM(config).train()
    model = Qwen3ForCausalLM(
        config,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", rope_fusion=False),
    ).train()
    model.load_state_dict(reference_model.state_dict())

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    reference_logits = reference_model(input_ids, position_ids=position_ids, use_cache=False).logits
    logits = model(input_ids, position_ids=position_ids, use_cache=False).logits
    reference_logits.square().sum().backward()
    logits.square().sum().backward()

    torch.testing.assert_close(logits, reference_logits, atol=1e-6, rtol=1e-6)
    reference_parameters = dict(reference_model.named_parameters())
    for name, parameter in model.named_parameters():
        reference_gradient = reference_parameters[name].grad
        assert reference_gradient is not None, name
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_gradient, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_qwen3_respects_tied_embedding_config(tie_word_embeddings):
    """The LM head must share storage with input embeddings only when configured."""
    config = _tiny_config()
    config.tie_word_embeddings = tie_word_embeddings

    model = Qwen3ForCausalLM(config, backend=BackendConfig(attn="sdpa"))

    assert (model.lm_head.weight is model.model.embed_tokens.weight) is tie_word_embeddings


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_qwen3_bshd_bf16_fp32_rope_forward_backward_and_cache(attn_implementation: str) -> None:
    """FP32 RoPE tables must preserve BF16 attention and cached decoding."""
    torch.manual_seed(1234)
    config = _tiny_config()
    config._attn_implementation = attn_implementation
    attention = Qwen3Attention(config, 0, BackendConfig(attn="sdpa", rope_fusion=False)).to(torch.bfloat16)
    hidden_states = torch.randn(1, 5, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    angles = torch.outer(torch.arange(257, 262, dtype=torch.float32), torch.tensor([1.0, 0.01]))
    cos, sin = angles.cos().repeat(1, 2).unsqueeze(0), angles.sin().repeat(1, 2).unsqueeze(0)
    attention_mask = torch.full((1, 1, 5, 5), float("-inf"), dtype=torch.bfloat16).triu(1)

    output, _ = attention(hidden_states, (cos, sin), attention_mask)
    assert output.dtype == torch.bfloat16
    output.backward(torch.randn_like(output))
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name

    cache = DynamicCache(config=config)
    with torch.no_grad():
        decoded = torch.cat(
            [
                attention(
                    hidden_states[:, index : index + 1],
                    (cos[:, index : index + 1], sin[:, index : index + 1]),
                    None,
                    past_key_values=cache,
                )[0]
                for index in range(hidden_states.shape[1])
            ],
            dim=1,
        )
    assert cache.get_seq_length() == hidden_states.shape[1]
    assert cache.layers[0].keys.dtype == cache.layers[0].values.dtype == torch.bfloat16
    torch.testing.assert_close(decoded, output)
