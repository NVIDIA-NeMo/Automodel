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

"""Tests for DFlash support of Qwen3.5-family targets (``Qwen/Qwen3.8-27B``).

Those targets ship as ``Qwen3_5ForConditionalGeneration``, which keeps its decoder
hyper-parameters on a nested ``text_config`` and its decoder blocks in a
``ModuleDict``. These cover the three seams that make it work: the registry entry,
the text-config unwrap, and the draft attention-shape overrides.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.components.speculative.dflash.target import HFDFlashTargetModel, resolve_text_config


def test_qwen3_5_targets_resolve_to_the_qwen3_shaped_drafts():
    """``Qwen/Qwen3.8-27B`` reports ``Qwen3_5ForConditionalGeneration``.

    Its drafter is still a plain Qwen3-shaped stack (the published checkpoint
    declares ``model_type: qwen3``), so both the DFlash and DFlash 2 classes apply.
    """
    for architecture in (
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ):
        spec = resolve_dflash_draft_spec([architecture])
        assert spec.draft_cls.__name__ == "Qwen3DFlashDraftModel"
        assert spec.draft2_cls.__name__ == "Qwen3DFlash2DraftModel"


def test_resolve_text_config_unwraps_only_when_nested():
    nested = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64, vocab_size=248320))
    assert resolve_text_config(nested).num_hidden_layers == 64

    flat = SimpleNamespace(num_hidden_layers=36, vocab_size=151936)
    assert resolve_text_config(flat) is flat

    # A wrapper config that declares ``text_config = None`` must not blank the config.
    explicit_none = SimpleNamespace(text_config=None, num_hidden_layers=8)
    assert resolve_text_config(explicit_none) is explicit_none


class _ModuleDictTarget(nn.Module):
    """Minimal stand-in for a ``*ForConditionalGeneration`` target.

    Mirrors the two structural traits that break naive introspection: the decoder
    blocks live in a ``ModuleDict`` under ``model.layers``, and ``num_hidden_layers``
    is only reachable through ``config.text_config``.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_layers))
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(i): nn.Identity() for i in range(num_layers)})


def test_target_wrapper_reads_layer_count_through_text_config():
    """Layer-id validation must not look for ``num_hidden_layers`` on the outer config.

    A multimodal wrapper config has no top-level ``num_hidden_layers``; reading it
    there raises before training starts.
    """
    target = _ModuleDictTarget(num_layers=64)
    wrapper = HFDFlashTargetModel(target, target_layer_ids=[5, 19, 33, 47, 61])
    assert wrapper.target_layer_ids == [5, 19, 33, 47, 61]

    # The ModuleDict is still resolved in integer order for hook registration.
    layers = wrapper._get_transformer_layers()
    assert len(layers) == 64
    assert layers[0] is target.model.layers["0"]

    with pytest.raises(ValueError, match="out of bounds"):
        HFDFlashTargetModel(target, target_layer_ids=[64])


def _draft_config(**overrides):
    """Draft config shaped like the published Qwen3.8-27B drafter (5 layers, tiny dims)."""
    kwargs = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
    }
    kwargs.update(overrides)
    cfg = Qwen3Config(**kwargs)
    cfg.num_target_layers = 64
    cfg.block_size = 8
    cfg.dflash_config = {
        "mask_token_id": 255,
        "target_layer_ids": [1, 3, 5],
        "conv_group_size": 16,
        "selector_rank": 32,
        "selector_top_k": 8,
    }
    cfg._attn_implementation = "sdpa"
    return cfg


def test_draft_attention_shape_can_diverge_from_the_target():
    """The published drafters size attention independently of their target.

    Qwen3.8-27B is 24 heads / 4 kv / head_dim 256 while its drafter is 32 / 8 / 128,
    so the draft's q/k/v projections must follow the override rather than the
    target's own shape.
    """
    heads, kv_heads, head_dim, hidden = 8, 4, 32, 64
    draft = Qwen3DFlash2DraftModel(
        _draft_config(num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim)
    )
    attn = draft.layers[0].self_attn
    assert attn.q_proj.weight.shape == (heads * head_dim, hidden)
    assert attn.k_proj.weight.shape == (kv_heads * head_dim, hidden)
    assert attn.o_proj.weight.shape == (hidden, heads * head_dim)

    # And the stack still runs with the diverged shape.
    n_blocks, block_size = 2, 8
    out = draft(
        position_ids=torch.arange(6 + n_blocks * block_size).unsqueeze(0),
        attention_mask=None,
        noise_embedding=torch.randn(1, n_blocks * block_size, hidden),
        target_hidden=torch.randn(1, 6, 3 * hidden),
    )
    assert out.shape == (1, n_blocks * block_size, hidden)
    assert torch.isfinite(out).all()
