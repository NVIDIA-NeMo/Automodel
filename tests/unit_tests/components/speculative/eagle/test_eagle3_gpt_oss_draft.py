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

"""Unit tests for the gpt-oss EAGLE-3 draft model."""

import pytest
import torch
from transformers import GptOssConfig

from nemo_automodel.components.models.common.utils import get_rope_config
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding as GptOssRotaryEmbedding
from nemo_automodel.components.models.llama.rope_utils import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from nemo_automodel.components.speculative.eagle.draft_gpt_oss import GptOssEagle3DraftModel
from nemo_automodel.components.speculative.eagle.registry import (
    EAGLE3_DRAFT_REGISTRY,
    resolve_eagle3_draft_spec,
)

# gpt-oss carries YaRN RoPE: base 150000 inside the scaling dict, rope_type
# "yarn", factor 32 (extends the 4096 pre-YaRN context). The draft must
# reproduce this schedule, not a plain RoPE.
_GPT_OSS_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "truncate": False,
    "original_max_position_embeddings": 4096,
    "rope_theta": 150000.0,
}


def _tiny_gpt_oss_config() -> GptOssConfig:
    """A small ``GptOssConfig`` carrying gpt-oss's YaRN RoPE layout."""
    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
        rope_scaling=dict(_GPT_OSS_ROPE_SCALING),
    )
    # Fields the recipe injects when deriving the draft config from the target.
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = config.hidden_size
    return config


# ── Registry wiring ──────────────────────────────────────────────────────


def test_registry_contains_gpt_oss():
    assert "GptOssForCausalLM" in EAGLE3_DRAFT_REGISTRY
    assert EAGLE3_DRAFT_REGISTRY["GptOssForCausalLM"].draft_cls is GptOssEagle3DraftModel


def test_resolve_eagle3_gpt_oss():
    spec = resolve_eagle3_draft_spec(["GptOssForCausalLM"])
    assert spec.draft_cls is GptOssEagle3DraftModel


# ── Draft model ──────────────────────────────────────────────────────────


def test_gpt_oss_eagle3_draft_forward_shape():
    config = _tiny_gpt_oss_config()
    model = GptOssEagle3DraftModel(config)

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


def test_draft_rope_matches_gpt_oss_target_rope():
    """The draft's rotation must be bit-identical to the gpt-oss target's YaRN RoPE.

    Guards the core correctness claim: feeding the same Q/K through the draft's
    rotary (Llama ``(cos, sin)`` layout + ``rotate_half``) yields the same result
    as the gpt-oss target's own ``RotaryEmbedding`` (interleaved apply). If the
    draft silently fell back to plain/llama3 RoPE this diverges.
    """
    config = _tiny_gpt_oss_config()
    draft_rope = GptOssEagle3DraftModel(config).model.layers[0].self_attn.rotary_emb

    batch, seq_len = 1, 12
    n_heads, n_kv_heads, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    torch.manual_seed(0)
    # Draft attention operates in [batch, heads, seq, head_dim].
    q = torch.randn(batch, n_heads, seq_len, head_dim)
    k = torch.randn(batch, n_kv_heads, seq_len, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    cos, sin = draft_rope(torch.empty(batch, seq_len, 1), position_ids)
    q_draft, k_draft = llama_apply_rotary_pos_emb(q, k, cos, sin)

    # Reference: the target's own RotaryEmbedding, which takes [batch, seq, heads,
    # head_dim] and applies interleaved YaRN rope over positions arange(seq_len).
    base, rope_parameters, _ = get_rope_config(config)
    ref_rope = GptOssRotaryEmbedding(
        head_dim=head_dim,
        base=base,
        dtype=torch.float32,
        initial_context_length=rope_parameters.get("original_max_position_embeddings", 4096),
        scaling_factor=rope_parameters.get("factor", 1.0),
        ntk_alpha=rope_parameters.get("beta_slow", 1.0),
        ntk_beta=rope_parameters.get("beta_fast", 32.0),
        device=None,
    )
    q_ref, k_ref = ref_rope(q.transpose(1, 2), k.transpose(1, 2))

    torch.testing.assert_close(q_draft.transpose(1, 2), q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_draft.transpose(1, 2), k_ref, rtol=1e-5, atol=1e-5)


def test_draft_rope_engages_yarn_concentration():
    """YaRN must actually be active (factor > 1 -> concentration != 1), not bypassed."""
    config = _tiny_gpt_oss_config()
    draft_rope = GptOssEagle3DraftModel(config).model.layers[0].self_attn.rotary_emb
    concentration, _ = draft_rope._rope._compute_concentration_and_inv_freq()
    assert concentration != 1.0


def test_draft_rejects_partial_rotary():
    config = _tiny_gpt_oss_config()
    config.rope_scaling = {**_GPT_OSS_ROPE_SCALING, "partial_rotary_factor": 0.5}
    with pytest.raises(ValueError, match="full rotary"):
        GptOssEagle3DraftModel(config)
