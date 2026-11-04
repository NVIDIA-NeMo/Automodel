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
from nemo_automodel.recipes.llm.train_dflash import _project_onto_qwen3_config_keys


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


def test_draft_config_projection_drops_qwen3_5_only_fields():
    """The draft config must not inherit fields a Qwen3 stack does not declare.

    A Qwen3.5 text config carries linear-attention shapes, MTP fields, gating,
    and ``partial_rotary_factor: 0.25`` (top-level and inside ``rope_parameters``).
    The published Qwen3.8-27B drafter ships with none of them and a full-width
    rotary table; a leaked ``partial_rotary_factor`` would make a serving runtime
    rebuild the rotary at a quarter width against full-rotary trained weights.
    """
    text_config = {
        "head_dim": 256,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "num_hidden_layers": 64,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "vocab_size": 248320,
        "tie_word_embeddings": False,
        "eos_token_id": 248044,
        "partial_rotary_factor": 0.25,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "output_gate_type": "swish",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    }
    projected = _project_onto_qwen3_config_keys(text_config)

    for leaked in (
        "partial_rotary_factor",
        "attn_output_gate",
        "full_attention_interval",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "mamba_ssm_dtype",
        "mtp_num_hidden_layers",
        "output_gate_type",
    ):
        assert leaked not in projected
    # Matches the published drafter's rope_parameters exactly.
    assert projected["rope_parameters"] == {"rope_theta": 10000000, "rope_type": "default"}
    # The inherited decoder shape survives the projection.
    for kept in ("head_dim", "hidden_size", "num_attention_heads", "rms_norm_eps", "vocab_size", "eos_token_id"):
        assert projected[kept] == text_config[kept]

    # And the built config yields a full-width rotary table (head_dim / 2 freqs).
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    cfg = Qwen3Config.from_dict({**projected, "num_hidden_layers": 5})
    assert Qwen3RotaryEmbedding(cfg).inv_freq.shape[0] * 2 == cfg.head_dim


def test_saved_draft_config_declares_it_is_not_causal():
    """The saved config must say ``is_causal: false``, as the published drafters do.

    The DFlash draft is non-causal by construction. A reader that infers
    causality from the layer type -- z-lab/dflash does exactly this:

        is_causal = getattr(config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)

    defaults a ``sliding_attention`` layer to CAUSAL. With ``draft_sliding_window``
    set (the shipped Qwen3.8-27B recipe sets 2048) every draft layer is
    ``sliding_attention``, so omitting the key serves the draft causally after it
    was trained non-causally.
    """
    from nemo_automodel.recipes.llm.train_dflash import TrainDFlashRecipe

    # Drive the recipe's own derivation rather than grepping its source, so the
    # assertion survives the code moving between methods.
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.block_size = 8
    recipe.mask_token_id = 248070
    recipe.draft_sliding_window = 2048
    built = recipe._build_qwen3_draft_config(
        {},
        target_text_config=Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=64),
        draft_cls=Qwen3DFlash2DraftModel,
        draft_num_hidden_layers=5,
        num_target_layers=64,
        target_layer_ids=[5, 19, 33, 47, 61],
        attention_backend="sdpa",
    ).to_dict()
    assert built["is_causal"] is False
    assert built["layer_types"] == ["sliding_attention"] * 5

    def reference_is_causal(config: dict, layer_idx: int) -> bool:
        """Verbatim rule from z-lab/dflash's Qwen3DFlashAttention.__init__."""
        layer_types = config.get("layer_types")
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_causal = config.get("is_causal")
        return layer_type == "sliding_attention" if is_causal is None else bool(is_causal)

    sliding = {"layer_types": ["sliding_attention"] * 5, "is_causal": False}
    assert not any(reference_is_causal(sliding, i) for i in range(5))
    # Without the key the same config would be read as causal.
    assert all(reference_is_causal({"layer_types": ["sliding_attention"] * 5}, i) for i in range(5))


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
