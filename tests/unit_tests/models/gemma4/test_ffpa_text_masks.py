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

"""CPU mask semantics for text-only Gemma4 MoE FFPA dispatch."""

from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from nemo_automodel.components.attention.ffpa_attention import (
    register_ffpa_attention,
    setup_ffpa_backend,
)
from nemo_automodel.components.models.gemma4_moe.model import (
    _build_unpacked_gemma4_causal_mask_mapping,
)


@pytest.fixture(autouse=True)
def _build_cpu_masks_eagerly(monkeypatch):
    from torch.nn.attention.flex_attention import create_block_mask
    from transformers import masking_utils

    def create_eager_block_mask(mask_mod, **kwargs):
        # Exercise real mask predicates and compression without paying a CPU JIT
        # startup for an eight-token unit test. GPU tests use stock compilation.
        kwargs["_compile"] = False
        return create_block_mask(mask_mod, **kwargs)

    monkeypatch.setattr(masking_utils, "create_block_mask", create_eager_block_mask)


def _config():
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=512,
        sliding_window=2,
        layer_types=["sliding_attention", "full_attention"],
        use_bidirectional_attention="vision",
    )
    config._attn_implementation = "ffpa"
    register_ffpa_attention()
    return config


@pytest.mark.parametrize("token_types", [None, [0] * 8, [3] * 8])
@pytest.mark.parametrize("padded", [False, True])
def test_text_only_full_layers_reach_ffpa_and_sliding_mask_remains_causal(token_types, padded):
    config = _config()
    embeddings = torch.zeros(1, 8, 32)
    positions = torch.arange(8).unsqueeze(0)
    padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0] if padded else [1] * 8])
    types = None if token_types is None else torch.tensor([token_types])
    masks = _build_unpacked_gemma4_causal_mask_mapping(
        config, embeddings, padding, None, positions, types, None, is_training=True
    )
    if padded:
        torch.testing.assert_close(masks["full_attention"], padding.bool())
    else:
        assert masks["full_attention"] is None
    # The plain full mask is eligible for CuTe; local layers retain their sparse mask.
    q_indices = torch.arange(8).view(8, 1)
    k_indices = torch.arange(8).view(1, 8)
    actual = masks["sliding_attention"].mask_mod(torch.tensor(0), torch.tensor(0), q_indices, k_indices)
    expected = (k_indices <= q_indices) & ((q_indices - k_indices) < 2) & padding.bool()
    torch.testing.assert_close(actual, expected)
    assert config.use_bidirectional_attention == "vision"


@pytest.mark.parametrize("token_types,pixels", [([0, 1, 1, 0], None), ([0, 2, 2, 0], None), ([0] * 4, True)])
def test_vision_inputs_keep_model_owned_mask_builder(monkeypatch, token_types, pixels):
    from transformers.models.gemma4 import modeling_gemma4

    # Both legacy and current Transformers must retain their original vision route.
    expected = {"full_attention": object(), "sliding_attention": object()}
    legacy = MagicMock(return_value=expected)
    monkeypatch.setattr(modeling_gemma4, "create_causal_mask_mapping", legacy, raising=False)
    types = torch.tensor([token_types])
    image = torch.zeros(1, 3, 2, 2) if pixels else None
    actual = _build_unpacked_gemma4_causal_mask_mapping(
        _config(),
        torch.zeros(1, 4, 32),
        None,
        None,
        None,
        types,
        image,
        is_training=True,
    )
    assert actual is expected
    assert legacy.call_args.kwargs["mm_token_type_ids"] is types
    assert legacy.call_args.kwargs["pixel_values"] is image


def test_position_resets_do_not_drop_document_mask():
    config = _config()
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    masks = _build_unpacked_gemma4_causal_mask_mapping(
        config,
        torch.zeros(1, 8, 32),
        None,
        None,
        positions,
        None,
        None,
        is_training=True,
    )
    q_indices = torch.arange(8).view(8, 1)
    k_indices = torch.arange(8).view(1, 8)
    actual = masks["full_attention"].mask_mod(torch.tensor(0), torch.tensor(0), q_indices, k_indices)
    expected = (k_indices <= q_indices) & ((q_indices // 4) == (k_indices // 4))
    torch.testing.assert_close(actual, expected)


def test_packed_sequence_configuration_still_rejected():
    with pytest.raises(ValueError, match="packed sequences"):
        setup_ffpa_backend(cp_size=1, has_packed_sequence=True)


def test_prebuilt_attention_mask_is_preserved():
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :, 3] = torch.finfo(torch.float32).min
    actual = _build_unpacked_gemma4_causal_mask_mapping(
        _config(), torch.zeros(1, 8, 32), mask, None, None, None, None, is_training=True
    )
    for layer_mask in actual.values():
        torch.testing.assert_close(layer_mask, mask)


def test_current_transformers_vision_mask_keeps_bidirectional_image_edges(monkeypatch):
    from transformers.models.gemma4 import modeling_gemma4

    monkeypatch.setattr(modeling_gemma4, "create_causal_mask_mapping", None, raising=False)
    types = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]])
    masks = _build_unpacked_gemma4_causal_mask_mapping(
        _config(),
        torch.zeros(1, 8, 32),
        None,
        None,
        torch.arange(8).unsqueeze(0),
        types,
        None,
        is_training=True,
    )
    mask = masks["sliding_attention"].mask_mod
    assert bool(mask(torch.tensor(0), torch.tensor(0), torch.tensor(1), torch.tensor(2)))
    assert not bool(mask(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(2)))
