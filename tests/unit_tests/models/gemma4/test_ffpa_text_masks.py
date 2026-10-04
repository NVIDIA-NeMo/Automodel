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

"""CPU mask semantics and decoder dispatch for Gemma4 MoE FFPA."""

from unittest.mock import Mock

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from nemo_automodel.components.attention.ffpa_attention import register_ffpa_attention
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4MoETextModelBackend,
    _build_unpacked_gemma4_causal_mask_mapping,
)


@pytest.fixture(autouse=True)
def _build_cpu_masks_eagerly(monkeypatch):
    from torch.nn.attention.flex_attention import create_block_mask
    from transformers import masking_utils

    def create_eager_block_mask(mask_mod, **kwargs):
        # Exercise real predicates and compression without CPU JIT startup.
        kwargs["_compile"] = False
        return create_block_mask(mask_mod, **kwargs)

    monkeypatch.setattr(masking_utils, "create_block_mask", create_eager_block_mask)


def _config(attn_implementation="ffpa"):
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_experts=2,
        top_k_experts=1,
        enable_moe_block=True,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=512,
        attention_k_eq_v=True,
        sliding_window=2,
        layer_types=["sliding_attention", "full_attention"],
        use_bidirectional_attention="vision",
        torch_dtype="float32",
    )
    config._attn_implementation = attn_implementation
    register_ffpa_attention()
    return config


@pytest.mark.parametrize("token_types", [None, [0] * 8, [0, 1, 1, 0, 0, 0, 0, 0]])
@pytest.mark.parametrize("padded", [False, True])
def test_decoder_layers_receive_ffpa_full_mask_and_sliding_block_mask(monkeypatch, token_types, padded):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def attention_spy(module, query, key, value, attention_mask, **kwargs):
        """Record dispatch without running CUDA kernels.

        Args:
            module: The real Gemma4 attention layer.
            query: Query tensor of shape [batch, query_heads, sequence, head_dim].
            key: Key tensor of shape [batch, kv_heads, sequence, head_dim].
            value: Value tensor of shape [batch, kv_heads, sequence, head_dim].
            attention_mask: None, a [batch, sequence] padding mask, or a BlockMask.
            **kwargs: Attention scaling and backend options.

        Returns:
            Zero output of shape [batch, sequence, query_heads, head_dim] and no weights.
        """
        return torch.zeros_like(query.transpose(1, 2)), None

    config = _config()
    spy = Mock(side_effect=attention_spy)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "ffpa", spy)
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")
    model = Gemma4MoETextModelBackend(config, backend)
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.02)
    # The expert activation compiles even on CPU. Isolate that unrelated
    # arithmetic while exercising real decoder layers and attention dispatch.
    for layer in model.layers.values():
        monkeypatch.setattr(layer.moe, "forward", Mock(return_value=torch.zeros(1, 8, 32)))
    padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0] if padded else [1] * 8])
    types = None if token_types is None else torch.tensor([token_types])
    result = model(input_ids=torch.arange(8).unsqueeze(0), attention_mask=padding, mm_token_type_ids=types)
    assert result.last_hidden_state.shape == (1, 8, 32)
    assert spy.call_count == 2
    sliding_call, full_call = spy.call_args_list
    assert full_call.args[0] is model.layers["1"].self_attn
    assert full_call.args[1].shape == (1, 4, 8, 512)
    full_mask = full_call.args[4]
    if padded:
        torch.testing.assert_close(full_mask, padding.bool())
    else:
        assert full_mask is None
    assert sliding_call.args[0] is model.layers["0"].self_attn
    assert isinstance(sliding_call.args[4], BlockMask)


@pytest.mark.parametrize("backend", ["eager", "sdpa", "ffpa"])
@pytest.mark.parametrize("vision_type", [1, 2])
def test_vision_full_mask_is_causal_and_sliding_image_edges_stay_in_window(backend, vision_type):
    types = torch.tensor([[0, vision_type, vision_type, vision_type, vision_type, 0]])
    masks = _build_unpacked_gemma4_causal_mask_mapping(
        _config(backend), torch.zeros(1, 6, 32), None, None, torch.arange(6).unsqueeze(0), types
    )
    q = torch.arange(6).view(6, 1)
    k = torch.arange(6).view(1, 6)
    causal = k <= q
    image_edges = ((q >= 1) & (q <= 4)) & ((k >= 1) & (k <= 4))
    expected_sliding = (causal | image_edges) & (k > q - 2)
    if backend in ("ffpa", "sdpa"):
        assert masks["full_attention"] is None  # Implicit causal attention.
    else:
        torch.testing.assert_close(masks["full_attention"][0, 0] == 0, causal)
    if backend == "ffpa":
        actual_sliding = masks["sliding_attention"].mask_mod(torch.tensor(0), torch.tensor(0), q, k)
    elif backend == "sdpa":
        actual_sliding = masks["sliding_attention"][0, 0]
    else:
        actual_sliding = masks["sliding_attention"][0, 0] == 0
    torch.testing.assert_close(actual_sliding, expected_sliding)
    assert actual_sliding[1, 2]  # Future token in the same image.
    assert not actual_sliding[4, 1]  # Same image, outside the backward window.


def test_position_resets_preserve_document_boundaries():
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    masks = _build_unpacked_gemma4_causal_mask_mapping(_config(), torch.zeros(1, 8, 32), None, None, positions, None)
    q = torch.arange(8).view(8, 1)
    k = torch.arange(8).view(1, 8)
    actual = masks["full_attention"].mask_mod(torch.tensor(0), torch.tensor(0), q, k)
    expected = (k <= q) & ((q // 4) == (k // 4))
    torch.testing.assert_close(actual, expected)


def test_prebuilt_attention_mask_is_preserved():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :, 3] = torch.finfo(torch.float32).min
    actual = _build_unpacked_gemma4_causal_mask_mapping(_config(), torch.zeros(1, 8, 32), mask, None, None, None)
    for layer_mask in actual.values():
        torch.testing.assert_close(layer_mask, mask)
