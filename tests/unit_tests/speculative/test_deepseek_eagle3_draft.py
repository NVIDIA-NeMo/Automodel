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

"""Unit tests for the DeepSeek (MLA) EAGLE-3 draft model.

Validation here is architecture-level: the MLA draft builds, runs the EAGLE-3
forward (including the ``cache_hidden`` TTT recurrence), trains (gradients flow),
integrates with the shared ``Eagle3TrainerModule``, and is registered. A real
acceptance benchmark needs a full DeepSeek target (too large for CI), so it is
out of scope; the math is anchored by reusing the onboarded DeepSeek target's MLA
projection layout and ``rope_utils`` (interleaved RoPE).
"""

from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_deepseek import DeepseekV3Eagle3DraftModel
from nemo_automodel.components.speculative.eagle.registry import resolve_eagle3_draft_spec

_VOCAB = 64
_DRAFT_VOCAB = 16
_HIDDEN = 32


def _build_config(q_lora_rank: int | None = 16) -> PretrainedConfig:
    config = PretrainedConfig()
    for key, value in dict(
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=6,
        vocab_size=_VOCAB,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        pad_token_id=0,
        target_hidden_size=_HIDDEN,
        draft_vocab_size=_DRAFT_VOCAB,
        num_aux_hidden_states=3,
    ).items():
        setattr(config, key, value)
    return config


def _build_draft(q_lora_rank: int | None = 16) -> DeepseekV3Eagle3DraftModel:
    torch.manual_seed(0)
    return DeepseekV3Eagle3DraftModel(_build_config(q_lora_rank)).to(torch.float32)


@pytest.mark.parametrize("q_lora_rank", [16, None])
def test_forward_shapes(q_lora_rank):
    """project / forward / compute_logits produce the EAGLE-3 shapes for both Q paths."""
    draft = _build_draft(q_lora_rank).eval()
    batch, seq = 2, 5
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    aux = torch.randn(batch, seq, _HIDDEN * 3)
    attn = torch.ones(batch, seq, dtype=torch.long)

    projected = draft.project_hidden_states(aux)
    assert projected.shape == (batch, seq, _HIDDEN)
    hidden = draft(input_ids, projected, attn)
    assert hidden.shape == (batch, seq, _HIDDEN)
    logits = draft.compute_logits(hidden)
    assert logits.shape == (batch, seq, _DRAFT_VOCAB)  # draft (compressed) vocab


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


def test_packing_not_supported_yet():
    draft = _build_draft().eval()
    ids = torch.randint(0, _VOCAB, (1, 4))
    proj = draft.project_hidden_states(torch.randn(1, 4, _HIDDEN * 3))
    with pytest.raises(NotImplementedError, match="sequence packing"):
        draft(ids, proj, torch.ones(1, 4, dtype=torch.long), seq_lens=torch.tensor([4]))


def test_set_vocab_mapping_builds_d2t_t2d():
    draft = _build_draft()
    selected = torch.arange(_DRAFT_VOCAB) * 2  # draft id i -> target id 2i
    draft.set_vocab_mapping(selected)
    # d2t is the offset (target_id - draft_id) vLLM/SGLang expect.
    torch.testing.assert_close(draft.d2t, selected - torch.arange(_DRAFT_VOCAB))
    assert int(draft.t2d.sum()) == _DRAFT_VOCAB
    assert draft.t2d[selected].all()


def test_set_vocab_mapping_rejects_wrong_length():
    draft = _build_draft()
    with pytest.raises(ValueError, match="draft_vocab_size"):
        draft.set_vocab_mapping(torch.arange(_DRAFT_VOCAB + 1))


def test_registry_resolves_deepseek_v3():
    spec = resolve_eagle3_draft_spec(["DeepseekV3ForCausalLM"])
    assert spec.draft_cls is DeepseekV3Eagle3DraftModel


def test_trainer_integration_trains_on_cpu():
    """The MLA draft plugs into the shared Eagle3TrainerModule and trains end to end."""
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


def test_rope_freqs_stays_fp32_after_bf16_cast():
    """``draft.to(bf16)`` must not round the MLA RoPE frequencies.

    The EAGLE-3 training build casts the draft with a raw ``.to(dtype=compute_dtype)``;
    a bf16-rounded ``rope_freqs`` dephases with absolute position (a bf16 round-trip
    cannot be undone by a later fp32 upcast), so the draft's RoPE would drift from
    the fp32 target and erode draft acceptance. Every ``rope_freqs`` buffer must
    stay fp32 (and exact) after the cast.
    """
    draft = _build_draft()  # fp32
    freq_names = [n for n, _ in draft.named_buffers() if n.endswith("rope_freqs")]
    assert freq_names, "expected at least one rope_freqs buffer on the MLA draft"
    ref = {n: draft.get_buffer(n).detach().clone().float() for n in freq_names}

    draft = draft.to(torch.bfloat16)
    for n in freq_names:
        rope_freqs = draft.get_buffer(n)
        # Without the fp32 pin this buffer would be bf16 (its frequencies rounded).
        assert rope_freqs.dtype == torch.float32, n
        # Recomputed fresh in fp32, not a bf16 round-trip (bf16 rel. error ~1e-2).
        assert torch.allclose(rope_freqs, ref[n], rtol=1e-6, atol=1e-8), n
    # The pin is rope-only: the rest of the draft still casts to bf16.
    assert any(p.dtype == torch.bfloat16 for p in draft.parameters())
