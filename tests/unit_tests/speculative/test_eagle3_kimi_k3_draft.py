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

"""Unit tests for the Kimi K3 (NoPE MLA) EAGLE-3 draft model.

Validation is architecture-level, as for the DeepSeek MLA draft: the draft builds
from a real ``KimiK3TextConfig``, runs the EAGLE-3 forward and its ``cache_hidden``
TTT recurrence, isolates packed documents, trains through the shared
``Eagle3TrainerModule``, and is registered for both K3 architectures. The
K3-specific contracts (NoPE, output gate, SiTU MLP, K3 RMSNorm) are asserted
directly, since they are what separates this draft from the DeepSeek one.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel._transformers.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel._transformers.models.kimi_k3.model import KimiK3MLP, KimiRMSNorm
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_kimi_k3 import (
    KimiK3Eagle3DraftModel,
    KimiK3Eagle3MLAAttention,
)
from nemo_automodel.components.speculative.eagle.registry import resolve_eagle3_draft_spec

_VOCAB = 64
_DRAFT_VOCAB = 16
_HIDDEN = 32


def _text_config(**overrides) -> KimiK3TextConfig:
    """A tiny but structurally faithful K3 text config (4 layers, 1 MLA layer)."""
    kwargs = dict(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=6,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4]},
    )
    kwargs.update(overrides)
    return KimiK3TextConfig(**kwargs)


def _draft_config(**overrides) -> KimiK3TextConfig:
    """Build the draft config exactly as ``TrainEagle3Recipe`` does: target config + EAGLE-3 keys."""
    config = _text_config(**overrides).to_dict()
    config.update(
        target_hidden_size=_HIDDEN,
        draft_vocab_size=_DRAFT_VOCAB,
        num_aux_hidden_states=3,
        tie_word_embeddings=False,
    )
    return KimiK3TextConfig.from_dict(config)


def _build_draft(**overrides) -> KimiK3Eagle3DraftModel:
    torch.manual_seed(0)
    return KimiK3Eagle3DraftModel(_draft_config(**overrides)).to(torch.float32)


def test_draft_mirrors_the_k3_backbone():
    """The draft reuses K3's own MLA layout, SiTU MLP, fp32 RMSNorm and output gate."""
    draft = _build_draft()
    layer = draft.model.layers[0]
    assert isinstance(layer.mlp, KimiK3MLP)
    assert isinstance(layer.input_layernorm, KimiRMSNorm)
    assert isinstance(layer.self_attn, KimiK3Eagle3MLAAttention)
    assert layer.self_attn.use_output_gate  # mla_use_output_gate defaults to True
    # The fused first layer consumes [embed, hidden], so every input projection is 2H wide.
    assert layer.self_attn.q_a_proj.in_features == 2 * _HIDDEN
    assert layer.self_attn.kv_a_proj_with_mqa.in_features == 2 * _HIDDEN
    assert layer.self_attn.g_proj.in_features == 2 * _HIDDEN
    # NoPE: no rotary cache is registered, unlike the DeepSeek MLA draft.
    assert not hasattr(layer.self_attn, "rope_freqs")


@pytest.mark.parametrize("q_lora_rank", [16, None])
def test_forward_shapes(q_lora_rank):
    """project / forward / compute_logits produce the EAGLE-3 shapes for both Q paths."""
    draft = _build_draft(q_lora_rank=q_lora_rank).eval()
    batch, seq = 2, 5
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)

    projected = draft.project_hidden_states(torch.randn(batch, seq, _HIDDEN * 3))
    assert projected.shape == (batch, seq, _HIDDEN)
    hidden = draft(input_ids, projected, attn)
    assert hidden.shape == (batch, seq, _HIDDEN)
    logits = draft.compute_logits(hidden)
    assert logits.shape == (batch, seq, _DRAFT_VOCAB)  # draft (compressed) vocab


def test_forward_is_position_id_invariant():
    """K3's MLA is NoPE, so shifting position_ids must not change the draft output."""
    draft = _build_draft().eval()
    batch, seq = 1, 6
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    proj = draft.project_hidden_states(torch.randn(batch, seq, _HIDDEN * 3))
    attn = torch.ones(batch, seq, dtype=torch.long)
    positions = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    out = draft(input_ids, proj, attn, position_ids=positions, cache_hidden=[[], []])
    out_shifted = draft(input_ids, proj, attn, position_ids=positions + 17, cache_hidden=[[], []])
    torch.testing.assert_close(out, out_shifted)


def test_output_gate_is_applied():
    """Zeroing the gate projection must zero the attention contribution (residual only)."""
    draft = _build_draft().eval()
    batch, seq = 1, 4
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    proj = draft.project_hidden_states(torch.randn(batch, seq, _HIDDEN * 3))
    attn = torch.ones(batch, seq, dtype=torch.long)
    attention = draft.model.layers[0].self_attn

    with torch.no_grad():
        # sigmoid(0) = 0.5 everywhere; halving the gate must change the output.
        gated = draft(input_ids, proj, attn, cache_hidden=[[], []])
        attention.g_proj.weight.zero_()
        neutral = draft(input_ids, proj, attn, cache_hidden=[[], []])
    assert not torch.allclose(gated, neutral)


def test_ttt_cache_recurrence_grows_and_runs_diagonal_path():
    """Reusing one cache_hidden across steps grows it and exercises the diagonal-extension block."""
    draft = _build_draft().eval()
    batch, seq = 2, 5
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    proj = draft.project_hidden_states(torch.randn(batch, seq, _HIDDEN * 3))
    attn = torch.ones(batch, seq, dtype=torch.long)

    cache = [[], []]
    draft(input_ids, proj, attn, cache_hidden=cache)  # step 0: plain causal
    out1 = draft(input_ids, proj, attn, cache_hidden=cache)  # step 1: diagonal extension
    assert len(cache[0]) == 2 and len(cache[1]) == 2
    assert out1.shape == (batch, seq, _HIDDEN)
    assert torch.isfinite(out1).all()


def test_packed_draft_attention_isolates_documents():
    """Block-causal packing isolates documents: perturbing doc B leaves doc A bit-identical."""
    torch.manual_seed(0)
    draft = _build_draft().eval()
    seq_len = 6
    seq_lens = torch.tensor([[3, 3]], dtype=torch.long)  # doc A: slots 0..2, doc B: slots 3..5
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    input_ids = torch.randint(0, _VOCAB, (1, seq_len))
    projected = draft.project_hidden_states(torch.randn(1, seq_len, _HIDDEN * 3))

    def run(ids, proj):
        return draft(
            input_ids=ids,
            projected_hidden_states=proj,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long),
            position_ids=position_ids,
            cache_hidden=[[], []],
            seq_lens=seq_lens,
        )

    out_ref = run(input_ids, projected)
    ids_b = input_ids.clone()
    ids_b[:, 3:] = torch.randint(0, _VOCAB, (1, 3))
    proj_b = projected.clone()
    proj_b[:, 3:] = torch.randn(1, 3, _HIDDEN)
    out_perturbed = run(ids_b, proj_b)

    torch.testing.assert_close(out_ref[:, :3], out_perturbed[:, :3])  # doc A unchanged
    assert not torch.allclose(out_ref[:, 3:], out_perturbed[:, 3:])  # doc B changed


def test_packing_requires_position_ids():
    """Packing without per-document position_ids must fail loud, naming this draft."""
    draft = _build_draft().eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    proj = draft.project_hidden_states(torch.randn(1, 6, _HIDDEN * 3))
    with pytest.raises(ValueError, match="KimiK3Eagle3DraftModel: sequence packing"):
        draft(ids, proj, torch.ones(1, 6, dtype=torch.long), seq_lens=torch.tensor([[3, 3]], dtype=torch.long))


def test_rejects_grouped_key_value_heads():
    """K3 MLA emits one K/V head per query head; a GQA-shaped config must fail early."""
    with pytest.raises(ValueError, match="one K/V head per query head"):
        _build_draft(num_key_value_heads=2)


def test_rejects_rope_mla():
    """The draft implements K3's NoPE MLA only."""
    with pytest.raises(ValueError, match="mla_use_nope"):
        _build_draft(mla_use_nope=False)


def test_freeze_embeddings_and_vocab_mapping():
    """The draft exposes the EAGLE-3 draft API the recipe drives it with."""
    draft = _build_draft()
    draft.freeze_embeddings()
    assert not draft.model.embed_tokens.weight.requires_grad

    selected = torch.arange(_DRAFT_VOCAB) * 2  # draft id i -> target id 2i
    draft.set_vocab_mapping(selected)
    torch.testing.assert_close(draft.d2t, selected - torch.arange(_DRAFT_VOCAB))
    assert int(draft.t2d.sum()) == _DRAFT_VOCAB


def test_copy_embeddings_from_target():
    draft = _build_draft()
    target_embedding = torch.nn.Embedding(_VOCAB, _HIDDEN)
    draft.copy_embeddings_from_target(target_embedding)
    torch.testing.assert_close(draft.model.embed_tokens.weight, target_embedding.weight)


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForCausalLM"], ["KimiK3ForConditionalGeneration"]],
)
def test_registry_resolves_kimi_k3(architectures):
    assert resolve_eagle3_draft_spec(architectures).draft_cls is KimiK3Eagle3DraftModel


def test_vision_language_config_exposes_the_text_decoder():
    """A K3 VL checkpoint nests the decoder config the draft is built from under text_config."""
    config = KimiK3Config(text_config=_text_config())
    assert config.get_text_config().model_type == "kimi_linear"
    assert config.architectures == ["KimiK3ForConditionalGeneration"]


def test_trainer_integration_trains_on_cpu():
    """The K3 draft plugs into the shared Eagle3TrainerModule and trains end to end."""
    draft = _build_draft()
    selected_token_ids = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
    selected_token_mask = torch.zeros(_VOCAB, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    module = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=2,
    )

    batch, seq = 2, 6
    out = module(
        input_ids=torch.randint(0, _VOCAB, (batch, seq)),
        attention_mask=torch.ones(batch, seq, dtype=torch.long),
        loss_mask=torch.ones(batch, seq, dtype=torch.long),
        aux_hidden_states=torch.randn(batch, seq, _HIDDEN * 3),
        target_logits=torch.randn(batch, seq, _VOCAB),
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in draft.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
