# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Backbone construction, single-pass residual behavior and training checks."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection, DeepseekV41Mix
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM


def _tiny_config() -> DeepseekV41Config:
    return DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=6,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0, 0, 2, 2, 1, 1],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            candidate_source_layer_id=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[],
            engram_num_embeddings=[],
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
            dtype="float32",
        ),
    )


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")


def test_custom_moe_must_preserve_released_combine_precision() -> None:
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
        assert model.moe_config.combine_in_fp32
        with pytest.raises(ValueError, match="combine_in_fp32=True"):
            DeepseekV41ForCausalLM(
                _tiny_config(), backend=_backend(), moe_config=replace(model.moe_config, combine_in_fp32=False)
            )


def test_mhc_combination_orientation_and_explicit_predecessor_mix() -> None:
    streams = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    previous_pre = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    current_pre = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    mix = DeepseekV41Mix(
        current_pre,
        torch.tensor([[[2.0, 3.0]]]),
        torch.tensor([[[[0.1, 0.9], [0.6, 0.4]]]]),
    )
    collapsed = DeepseekV41HyperConnection.collapse(streams, previous_pre)
    torch.testing.assert_close(collapsed, torch.tensor([[[1.0, 2.0]]]))
    result = DeepseekV41HyperConnection.expand(collapsed, streams, mix)
    # Residual output0 = .1*stream0 + .6*stream1; output1 = .9*stream0 + .4*stream1.
    expected = torch.tensor([[[[3.9, 6.6], [5.1, 9.4]]]])
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert previous_pre.grad is not None
    assert current_pre.grad is None  # It belongs to the following sublayer's input.


def test_mhc_coefficients_and_gradients_are_finite() -> None:
    torch.manual_seed(42)
    module = DeepseekV41HyperConnection(_tiny_config().text_config)
    streams = torch.randn(2, 3, 4, 16, requires_grad=True)
    mix = module(streams)
    assert torch.all(mix.pre > 0)
    assert torch.all((mix.post > 0) & (mix.post < 2))
    torch.testing.assert_close(mix.comb.sum(-1), torch.ones(2, 3, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(mix.comb.sum(-2), torch.ones(2, 3, 4), atol=1e-4, rtol=0)
    (mix.pre.square().sum() + mix.post.square().sum() + mix.comb.square().sum()).backward()
    assert torch.isfinite(streams.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_full_tiny_ced_model_trains_after_meta_initialization() -> None:
    torch.manual_seed(8)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    inputs = torch.randint(0, 64, (2, 12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = model(inputs, labels=inputs, output_hidden_states=True)
    assert initial.logits.shape == (2, 12, 64)
    assert len(initial.hidden_states) == 6
    initial_loss = initial.loss.item()
    initial.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    optimizer.zero_grad()
    assert model(inputs, labels=inputs).loss.item() < initial_loss


def test_tied_embeddings_are_rejected() -> None:
    config = _tiny_config()
    config.tie_word_embeddings = True
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        DeepseekV41ForCausalLM(config, backend=_backend())
