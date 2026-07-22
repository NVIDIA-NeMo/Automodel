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

from nemo_automodel.components.models.deepseek_v3.rope_utils import precompute_freqs_cis
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule
from nemo_automodel.components.speculative.eagle.draft_deepseek import (
    DeepseekV3Eagle3DraftModel,
    _resolve_rope_theta,
)
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


def test_packed_single_doc_matches_unpacked_forward():
    """A packed row that is one full-width document must match the plain causal forward.

    ``seq_lens=[[T]]`` with ``position_ids=arange`` builds a block-causal mask that is
    exactly the lower-triangular causal mask, so the packed path must be bit-identical
    to the unpacked path (which also uses ``arange`` positions and an all-ones mask).
    """
    draft = _build_draft().eval()
    batch, seq = 1, 6
    input_ids = torch.randint(0, _VOCAB, (batch, seq))
    proj = draft.project_hidden_states(torch.randn(batch, seq, _HIDDEN * 3))
    attn = torch.ones(batch, seq, dtype=torch.long)
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    out_unpacked = draft(input_ids, proj, attn, position_ids=position_ids, cache_hidden=[[], []])
    out_packed = draft(
        input_ids,
        proj,
        attn,
        position_ids=position_ids,
        cache_hidden=[[], []],
        seq_lens=torch.tensor([[seq]], dtype=torch.long),
    )
    torch.testing.assert_close(out_packed, out_unpacked)


def test_packed_draft_attention_isolates_documents():
    """Block-causal packing must isolate documents: perturbing only doc B's inputs
    leaves doc A's output bit-identical (cross-document attention is masked)."""
    torch.manual_seed(0)
    draft = _build_draft().eval()
    seq_len = 6
    seq_lens = torch.tensor([[3, 3]], dtype=torch.long)  # doc A: slots 0..2, doc B: slots 3..5
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)  # reset per document
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
    # Perturb only document B (slots 3..5).
    ids_b = input_ids.clone()
    ids_b[:, 3:] = torch.randint(0, _VOCAB, (1, 3))
    proj_b = projected.clone()
    proj_b[:, 3:] = torch.randn(1, 3, _HIDDEN)
    out_perturbed = run(ids_b, proj_b)

    torch.testing.assert_close(out_ref[:, :3], out_perturbed[:, :3])  # doc A unchanged
    assert not torch.allclose(out_ref[:, 3:], out_perturbed[:, 3:])  # doc B changed


def test_packing_requires_position_ids():
    """Packing without per-document position_ids must fail loud, not silently use arange."""
    draft = _build_draft().eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    proj = draft.project_hidden_states(torch.randn(1, 6, _HIDDEN * 3))
    with pytest.raises(ValueError, match="per-document position_ids"):
        draft(ids, proj, torch.ones(1, 6, dtype=torch.long), seq_lens=torch.tensor([[3, 3]], dtype=torch.long))


def test_packing_rejects_seq_lens_not_summing_to_t():
    """A seq_lens row that does not sum to T must raise, not silently misbucket documents."""
    draft = _build_draft().eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    proj = draft.project_hidden_states(torch.randn(1, 6, _HIDDEN * 3))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="sum to seq_length"):
        draft(
            ids,
            proj,
            torch.ones(1, 6, dtype=torch.long),
            position_ids=position_ids,
            seq_lens=torch.tensor([[3, 2]], dtype=torch.long),  # sums to 5, not 6
        )


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


def test_packing_matches_padding_loss_and_grads():
    """Golden parity: packing N docs into one row == N padded single-doc rows.

    Target-free draft-level check: identical per-document aux hidden states and
    target logits are fed in two layouts through the shared Eagle3TrainerModule --
      * padding: ``[D, T]`` with each doc's ``L`` real tokens padded to ``T``
        (per-row causal + padding mask, ``loss_mask``-shift TTT gating);
      * packing: the same docs as ``[1, T]`` (``D * L == T``) with per-document
        position_ids, the MLA block-causal mask, and ``doc_remaining`` gating.
    Both supervise the identical (doc, position, TTT step) triples against
    identical targets, so loss and every gradient match (CPU/fp32, tight tol).
    """
    num_docs, doc_len = 3, 8
    total = num_docs * doc_len  # packed row width T
    config = _build_config()
    config.draft_vocab_size = _VOCAB  # full vocab so every position is supervised
    config.max_position_embeddings = total

    def build_trainer():
        torch.manual_seed(123)  # identical draft init for both layouts
        draft = DeepseekV3Eagle3DraftModel(config).to(torch.float32)
        selected_token_ids = torch.arange(_VOCAB, dtype=torch.long)
        selected_token_mask = torch.ones(_VOCAB, dtype=torch.bool)
        return Eagle3TrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=3,
        )

    torch.manual_seed(7)
    doc_ids = [torch.randint(0, _VOCAB, (doc_len,)) for _ in range(num_docs)]
    doc_aux = [torch.randn(doc_len, _HIDDEN * 3) for _ in range(num_docs)]
    doc_logits = [torch.randn(doc_len, _VOCAB) for _ in range(num_docs)]

    # Layout A: D padded single-document rows.
    ids_a = torch.zeros(num_docs, total, dtype=torch.long)
    aux_a = torch.zeros(num_docs, total, _HIDDEN * 3)
    logits_a = torch.zeros(num_docs, total, _VOCAB)
    loss_a = torch.zeros(num_docs, total, dtype=torch.long)
    attn_a = torch.zeros(num_docs, total, dtype=torch.long)
    for d in range(num_docs):
        ids_a[d, :doc_len] = doc_ids[d]
        aux_a[d, :doc_len] = doc_aux[d]
        logits_a[d, :doc_len] = doc_logits[d]
        # Supervise every real token except the doc's last one, which has no
        # in-document next-token label. In the packed layout that same boundary
        # is enforced by ``doc_remaining`` (== 0 at each doc's last slot), so this
        # keeps both layouts on the identical supervised (doc, slot, step) triples.
        loss_a[d, : doc_len - 1] = 1
        attn_a[d, :doc_len] = 1
    trainer_a = build_trainer()
    metrics_a = trainer_a(
        input_ids=ids_a,
        attention_mask=attn_a,
        loss_mask=loss_a,
        aux_hidden_states=aux_a,
        target_logits=logits_a,
    )
    metrics_a.loss.backward()
    grads_a = {n: p.grad.clone() for n, p in trainer_a.named_parameters() if p.grad is not None}

    # Layout B: the same documents packed into one row (no padding).
    ids_b = torch.cat(doc_ids).unsqueeze(0)
    aux_b = torch.cat(doc_aux).unsqueeze(0)
    logits_b = torch.cat(doc_logits).unsqueeze(0)
    loss_b = torch.ones(1, total, dtype=torch.long)
    attn_b = torch.ones(1, total, dtype=torch.long)
    position_ids = torch.cat([torch.arange(doc_len) for _ in range(num_docs)]).unsqueeze(0)
    doc_remaining = torch.cat([torch.arange(doc_len - 1, -1, -1) for _ in range(num_docs)]).unsqueeze(0)
    seq_lens = torch.tensor([[doc_len] * num_docs], dtype=torch.long)
    trainer_b = build_trainer()
    metrics_b = trainer_b(
        input_ids=ids_b,
        attention_mask=attn_b,
        loss_mask=loss_b,
        aux_hidden_states=aux_b,
        target_logits=logits_b,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    metrics_b.loss.backward()
    grads_b = {n: p.grad.clone() for n, p in trainer_b.named_parameters() if p.grad is not None}

    assert metrics_a.valid_tokens.item() == metrics_b.valid_tokens.item()
    torch.testing.assert_close(metrics_a.loss, metrics_b.loss, rtol=1e-4, atol=1e-5)
    assert set(grads_a) == set(grads_b)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=1e-4, atol=1e-5, msg=f"grad mismatch: {name}")


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


def test_resolve_rope_theta_prefers_rope_parameters():
    """transformers 5.x drops the top-level ``rope_theta``; it lives under ``rope_parameters``."""
    config = PretrainedConfig()
    config.rope_parameters = {"rope_theta": 123456.0}
    config.rope_theta = 10000.0  # a stale top-level value must lose to rope_parameters
    assert _resolve_rope_theta(config) == 123456.0


def test_resolve_rope_theta_falls_back_to_top_level_then_default():
    """Older configs with only a top-level ``rope_theta`` (or none at all) still resolve."""
    config = PretrainedConfig()
    config.rope_parameters = None
    config.rope_theta = 5000.0
    assert _resolve_rope_theta(config) == 5000.0

    bare = PretrainedConfig()
    bare.rope_parameters = None
    assert _resolve_rope_theta(bare) == 10000.0


def test_rope_freqs_follow_rope_parameters_theta():
    """The draft's rotary table must use the target's ``rope_parameters`` base.

    A real ``DeepseekV3Config`` carries ``rope_theta`` only inside
    ``rope_parameters`` (no top-level attribute), so reading it with a bare
    ``getattr(config, "rope_theta", 10000.0)`` silently pins the draft's RoPE
    base to 10000 and dephases it from any target with a different base. Build
    the draft from such a config and check the frequency table end to end,
    including the fp32 recompute that follows a bf16 cast.
    """
    config = _build_config()
    del config.rope_theta
    config.rope_parameters = {"rope_theta": 123456.0}
    torch.manual_seed(0)
    draft = DeepseekV3Eagle3DraftModel(config).to(torch.float32)

    expected = precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings, 123456.0, None)
    default_theta = precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings, 10000.0, None)
    assert not torch.allclose(expected, default_theta)

    freq_names = [n for n, _ in draft.named_buffers() if n.endswith("rope_freqs")]
    assert freq_names, "expected at least one rope_freqs buffer on the MLA draft"
    for n in freq_names:
        assert torch.allclose(draft.get_buffer(n), expected, rtol=1e-6, atol=1e-8), n

    # The post-cast fp32 recompute must resolve the same theta, not the default.
    draft = draft.to(torch.bfloat16)
    for n in freq_names:
        assert torch.allclose(draft.get_buffer(n), expected, rtol=1e-6, atol=1e-8), n
