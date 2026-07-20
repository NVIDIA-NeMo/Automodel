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

"""Unit tests for the Gemma4 EAGLE-3 draft model and its target-side plumbing."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.draft_gemma import (
    Gemma4Eagle3DraftModel,
    _extract_global_rope_theta,
    _normalize_gemma4_draft_config,
)
from nemo_automodel.components.speculative.eagle.registry import (
    EAGLE3_DRAFT_REGISTRY,
    resolve_eagle3_draft_spec,
)
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

Gemma4TextConfig = pytest.importorskip("transformers").Gemma4TextConfig

# The full-attention (global) base the draft must adopt; the sliding base (1e4)
# must NOT leak through.
_GLOBAL_ROPE_THETA = 1_000_000.0
_LOCAL_ROPE_THETA = 10_000.0


def _tiny_gemma4_text_config() -> Gemma4TextConfig:
    """A small ``Gemma4TextConfig`` carrying Gemma4's nested per-type RoPE layout."""
    config = Gemma4TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=64,
        hidden_activation="gelu_pytorch_tanh",
    )
    # Gemma4TextConfig populates the nested rope_parameters by default, but pin it
    # explicitly so the test is independent of transformers' defaults.
    config.rope_parameters = {
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": _GLOBAL_ROPE_THETA,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": _LOCAL_ROPE_THETA},
    }
    # Fields the recipe injects when deriving the draft config from the target.
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = config.hidden_size
    return config


# ── Registry wiring ──────────────────────────────────────────────────────


def test_registry_contains_gemma4():
    assert "Gemma4ForConditionalGeneration" in EAGLE3_DRAFT_REGISTRY
    assert EAGLE3_DRAFT_REGISTRY["Gemma4ForConditionalGeneration"].draft_cls is Gemma4Eagle3DraftModel


def test_resolve_eagle3_gemma4():
    spec = resolve_eagle3_draft_spec(["Gemma4ForConditionalGeneration"])
    assert spec.draft_cls is Gemma4Eagle3DraftModel


# ── Config normalization ─────────────────────────────────────────────────


def test_extract_global_rope_theta_from_nested():
    config = _tiny_gemma4_text_config()
    assert _extract_global_rope_theta(config) == _GLOBAL_ROPE_THETA


def test_extract_global_rope_theta_flat_fallback():
    config = SimpleNamespace(rope_parameters=None, rope_theta=1234.0)
    assert _extract_global_rope_theta(config) == 1234.0


def test_extract_global_rope_theta_default():
    config = SimpleNamespace(rope_parameters=None)
    assert _extract_global_rope_theta(config) == _GLOBAL_ROPE_THETA


def test_normalize_sets_hidden_act_and_flattens_rope():
    config = _tiny_gemma4_text_config()
    _normalize_gemma4_draft_config(config)
    assert config.hidden_act == "gelu_pytorch_tanh"
    assert config.rope_parameters is None
    assert config.rope_scaling is None
    assert config.rope_theta == _GLOBAL_ROPE_THETA
    assert config.partial_rotary_factor == 1.0


def test_normalize_honors_explicit_hidden_act():
    config = _tiny_gemma4_text_config()
    config.hidden_act = "silu"
    _normalize_gemma4_draft_config(config)
    assert config.hidden_act == "silu"


def test_normalize_dense_leaves_intermediate_size():
    # Dense Gemma4 (no MoE block): intermediate_size is the real FFN width, untouched.
    config = _tiny_gemma4_text_config()
    original = config.intermediate_size
    _normalize_gemma4_draft_config(config)
    assert config.intermediate_size == original


def test_normalize_moe_expands_intermediate_to_active_width():
    # MoE target: config intermediate_size is per-expert (here 20, below hidden 32).
    # The draft MLP must be sized to the active FFN width (top_k_experts * per-expert),
    # not the contracting per-expert width.
    config = _tiny_gemma4_text_config()
    config.intermediate_size = 20
    config.enable_moe_block = True
    config.top_k_experts = 4
    _normalize_gemma4_draft_config(config)
    assert config.intermediate_size == 80  # 20 * 4, now well above hidden_size (32)


# ── Draft model ──────────────────────────────────────────────────────────


def test_gemma4_eagle3_draft_forward_shape():
    config = _tiny_gemma4_text_config()
    model = Gemma4Eagle3DraftModel(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    hidden_states = model(
        input_ids=input_ids,
        projected_hidden_states=model.project_hidden_states(aux_hidden_states),
        attention_mask=attention_mask,
    )
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)
    assert logits.shape == (batch_size, seq_len, config.draft_vocab_size)


def test_gemma4_draft_rope_uses_global_theta_not_local():
    """The draft's single full-attention layer must adopt the global base (1e6),
    a standard full-rotary Llama RoPE -- not the sliding base (1e4) nor a base-10000
    fallback from the unread nested ``rope_parameters``."""
    config = _tiny_gemma4_text_config()
    draft_rope = Gemma4Eagle3DraftModel(config).model.layers[0].self_attn.rotary_emb

    head_dim = config.head_dim
    expected = 1.0 / (_GLOBAL_ROPE_THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    torch.testing.assert_close(draft_rope.inv_freq.float(), expected, rtol=1e-6, atol=1e-6)


# ── Target-side layer extraction (nested language_model backbone) ─────────


def _make_stub_gemma_target(num_layers: int = 8) -> nn.Module:
    """A minimal stand-in for a loaded ``Gemma4ForConditionalGeneration``.

    Mirrors the module tree the target wrapper must walk:
    ``model.model.language_model.layers`` with the depth on the nested
    ``config.text_config.num_hidden_layers`` (a multimodal ``Gemma4Config`` has no
    top-level ``num_hidden_layers``).
    """
    language_model = nn.Module()
    language_model.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
    inner = nn.Module()
    inner.language_model = language_model
    model = nn.Module()
    model.model = inner
    model.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_layers))
    model.get_input_embeddings = lambda: None
    return model


def test_target_locates_nested_language_model_layers():
    stub = _make_stub_gemma_target(num_layers=8)
    wrapper = HFEagle3TargetModel(stub)
    layers = wrapper._get_transformer_layers()
    assert len(layers) == 8
    assert layers is not stub.model.language_model.layers  # returns a plain list
    assert all(layers[i] is stub.model.language_model.layers[i] for i in range(8))


def test_target_num_hidden_layers_from_text_config():
    stub = _make_stub_gemma_target(num_layers=10)
    wrapper = HFEagle3TargetModel(stub)
    assert wrapper._num_hidden_layers() == 10
    # Default aux ids are the standard [1, n//2-1, n-4] offsets, in-bounds for n=10.
    assert wrapper.aux_layer_ids == [1, 4, 6]
