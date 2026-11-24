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

from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution
from nemo_automodel.components.speculative.eagle.parallel_reconsideration import (
    LlamaParallelReconsiderationDraftModel,
    ParallelReconsiderationConfig,
    ReconsiderationKV,
    halt_probabilities,
    reconsideration_token_loss,
)


def _draft() -> LlamaParallelReconsiderationDraftModel:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        draft_vocab_size=16,
        max_position_embeddings=64,
        fc_norm=True,
        norm_output=True,
    )
    return LlamaParallelReconsiderationDraftModel(config).float()


def test_reconsideration_gate_depth_cap_and_gradients() -> None:
    probabilities = torch.tensor([0.25, 0.4])
    torch.testing.assert_close(halt_probabilities(torch.logit(probabilities)), torch.tensor([0.25, 0.30, 0.45]))
    ce = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mass = torch.tensor([0.25, 0.30, 0.45], requires_grad=True)
    loss = reconsideration_token_loss(ce, mass)
    torch.testing.assert_close(loss, torch.tensor(5.15))
    d_ce, d_mass = torch.autograd.grad(loss, (ce, mass))
    torch.testing.assert_close(d_ce, torch.tensor([1.0, 0.30, 0.45]))
    torch.testing.assert_close(d_mass, ce.detach())


def test_reconsideration_replaces_the_same_rope_slot() -> None:
    torch.manual_seed(13)
    model = _draft()
    ids = torch.tensor([[1, 2, 3]])
    features = torch.randn(1, 3, 16)
    positions = torch.arange(3)[None]
    floor = torch.finfo(features.dtype).min
    inclusive = torch.triu(torch.full((3, 3), floor), diagonal=1)[None, None]
    strict = torch.triu(torch.full((3, 3), floor), diagonal=0)[None, None]
    base = model.draft_step(model.embed_input_ids(ids), features, inclusive, positions)
    reference = model(ids, features, torch.ones_like(ids), positions)
    torch.testing.assert_close(base.hidden, reference)

    embeddings = model.self_query(base.hidden)
    refined = model.draft_step(embeddings, base.hidden, strict, positions, prefix=(base.kv,))
    key, value = base.kv.key.clone(), base.kv.value.clone()
    key[:, :, 0] += 100
    value[:, :, 0] += 100
    modified = model.draft_step(embeddings, base.hidden, strict, positions, prefix=(ReconsiderationKV(key, value),))
    torch.testing.assert_close(refined.hidden[:, 0], modified.hidden[:, 0])
    assert not torch.allclose(refined.hidden[:, 1], modified.hidden[:, 1])


def test_live_labels_use_the_cache_order_for_frequency_ranked_vocab() -> None:
    model = _draft()
    selected = (torch.arange(16) * 2).roll(5)
    selected_mask = torch.zeros(32, dtype=torch.bool)
    selected_mask[selected] = True
    trainer = ParallelReconsiderationConfig().build(
        model, selected_token_ids=selected, selected_token_mask=selected_mask, ttt_steps=2
    )
    # The last teacher token is outside the draft vocabulary.
    logits = torch.nn.functional.one_hot(torch.tensor([[selected[0], selected[9], 1]]), 32).float()
    loss_mask = torch.ones((1, 3), dtype=torch.bool)
    probs, mask = _compute_target_distribution(logits, selected, selected_mask, loss_mask)
    live_labels, live_mask = trainer._teacher_labels(
        loss_mask, target_logits=logits, target_probs=None, position_mask=None
    )
    cached_labels, cached_mask = trainer._teacher_labels(
        loss_mask, target_logits=None, target_probs=probs, position_mask=mask
    )
    torch.testing.assert_close(live_mask, cached_mask)
    torch.testing.assert_close(live_labels[live_mask], torch.tensor([0, 9]))
    torch.testing.assert_close(live_labels[live_mask], cached_labels[cached_mask])


def test_native_draft_training_and_serialization(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = _draft()
    selected = torch.arange(16) * 2
    selected_mask = torch.zeros(32, dtype=torch.bool)
    selected_mask[selected] = True
    model.set_vocab_mapping(selected)
    model.gradient_checkpointing_enable()
    trainer = ParallelReconsiderationConfig().build(
        model, selected_token_ids=selected, selected_token_mask=selected_mask, ttt_steps=2
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = torch.nn.functional.one_hot(torch.tensor([[2, 4, 6, 8]]), 32).float()
    out = trainer(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.tensor([[1, 1, 1, 0]]),
        aux_hidden_states=torch.randn(1, 4, 48),
        target_logits=logits,
    )
    assert torch.isfinite(out.loss)
    assert out.valid_tokens.item() == 7
    out.loss.backward()
    params = (model.self_query.weight, model.halt_gate.weight, model.model.layers[0].self_attn.q_proj.weight)
    before = [p.detach().clone() for p in params]
    for parameter in params:
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert all(not torch.equal(p, previous) for p, previous in zip(params, before))

    model.save_pretrained(tmp_path)
    loaded = LlamaParallelReconsiderationDraftModel.from_pretrained(tmp_path)
    assert loaded.config.architectures == ["LlamaParallelReconsiderationDraftModel"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
