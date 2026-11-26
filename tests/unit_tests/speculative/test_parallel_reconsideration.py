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

import math
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution
from nemo_automodel.components.speculative.eagle.parallel_reconsideration import (
    LlamaParallelReconsiderationDraftModel,
    ParallelReconsiderationConfig,
    ParallelReconsiderationTrainerModule,
    ReconsiderationKV,
    ReconsiderationStep,
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


def _trainer(*, ttt_steps: int = 2) -> ParallelReconsiderationTrainerModule:
    model = _draft()
    selected = torch.arange(16) * 2
    model.set_vocab_mapping(selected)
    return ParallelReconsiderationConfig().build(
        model, selected_token_ids=selected, selected_token_mask=torch.arange(32) % 2 == 0, ttt_steps=ttt_steps
    )


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
    # Prefix length changes the visible keys, but never the projected key's RoPE slot.
    uncached = model.draft_step(embeddings, base.hidden, inclusive, positions)
    torch.testing.assert_close(refined.kv.key, uncached.kv.key, rtol=0, atol=0)
    key, value = base.kv.key.clone(), base.kv.value.clone()
    key[:, :, 0] += 100
    value[:, :, 0] += 100
    modified = model.draft_step(embeddings, base.hidden, strict, positions, prefix=(ReconsiderationKV(key, value),))
    torch.testing.assert_close(refined.hidden[:, 0], modified.hidden[:, 0])
    assert not torch.allclose(refined.hidden[:, 1], modified.hidden[:, 1])


@pytest.mark.cpu_only
def test_two_step_rollout_matches_recorded_tokens_and_reference_loss() -> None:
    torch.manual_seed(19)
    trainer = _trainer()
    draft = trainer.draft_model
    with torch.no_grad():
        draft.halt_gate.weight.zero_()
        draft.halt_gate.bias.zero_()
    # Odd recorded IDs cannot be sampled from the compressed draft vocabulary.
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    labels = torch.tensor([[0, 2, 4, 6, 8]])
    depth_by_origin = [0, 1, 2, 1, 2]
    passes: list[ReconsiderationStep] = []
    original_step = draft.draft_step
    original_multinomial = torch.multinomial

    def record_step(
        embeds: Tensor,
        hidden: Tensor,
        mask: Tensor,
        positions: Tensor,
        *,
        prefix: tuple[ReconsiderationKV, ...] = (),
    ) -> ReconsiderationStep:
        """Record the real decoder output and its uncommitted cache.

        Args:
            embeds: Tensor of shape [batch, origin, hidden].
            hidden: Tensor of shape [batch, origin, hidden].
            mask: Tensor of shape [batch, 1, origin, origin].
            positions: Tensor of shape [batch, origin].
            prefix: Prior blocks with keys and values of shape [batch, heads, origin, head_dim].

        Returns:
            A pass with features [batch, origin, hidden] and KV [batch, heads, origin, head_dim].
        """
        result = original_step(embeds, hidden, mask, positions, prefix=prefix)
        passes.append(result)
        return result

    def draw_halt(probs: Tensor, num_samples: int) -> Tensor:
        """Choose fixed halt depths for each origin.

        Args:
            probs: Tensor of shape [batch * origin, depth] for a halt draw, or
                [batch * origin, vocab] for a token draw.
            num_samples: Number of samples per row.

        Returns:
            Sampled indices of shape [batch * origin, num_samples].
        """
        if probs.size(-1) != 3:
            return original_multinomial(probs, num_samples)
        torch.testing.assert_close(probs, torch.tensor([[0.5, 0.25, 0.25]]).expand_as(probs))
        return torch.tensor(depth_by_origin)[:, None]

    with (
        patch.object(draft, "draft_step", side_effect=record_step) as trace,
        patch.object(torch, "multinomial", side_effect=draw_halt),
    ):
        result = trainer(
            input_ids=ids,
            attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
            loss_mask=torch.tensor([[0, 1, 1, 1, 0]]),
            aux_hidden_states=torch.randn(1, 5, 48),
            target_logits=F.one_hot(2 * labels, 32).float(),
        )

    assert len(passes) == 7
    steps = trace.call_args_list
    shifted = draft.embed_input_ids(torch.tensor([[3, 5, 7, 9, 0]]))
    for advance in (steps[3], steps[4]):
        torch.testing.assert_close(advance.args[0], shifted, rtol=0, atol=0)
    visible = torch.ones(5, 5, dtype=torch.bool).tril()
    visible[:, 3:] = False
    for index, call in enumerate(steps):
        expected = visible if index in (0, 3) else visible.tril(-1)
        torch.testing.assert_close(call.args[2][0, 0] == 0, expected)

    recurrent = steps[4].args[1]
    committed = steps[4].kwargs["prefix"][-1]
    for origin, depth in enumerate(depth_by_origin):
        expected = passes[depth]
        torch.testing.assert_close(recurrent[:, origin], expected.hidden[:, origin])
        torch.testing.assert_close(committed.key[:, :, origin], expected.kv.key[:, :, origin])
        torch.testing.assert_close(committed.value[:, :, origin], expected.kv.value[:, :, origin])

    # The supervision is at origins 1,2 on step zero and 0,1,2 on step one.
    # With masses (1/2, 1/4, 1/4), Eq. (6) has CE coefficients (3/2, 1/2, 1/2).
    logits = [draft.compute_logits(step.hidden)[0] for step in passes]
    first = [F.cross_entropy(logits[i][1:3], labels[0, 1:3], reduction="sum") for i in (0, 1, 2)]
    second = [F.cross_entropy(logits[i][:3], labels[0, 1:4], reduction="sum") for i in (3, 4, 5, 6)]
    numerator = 1.5 * (first[0] + second[1]) + 0.5 * sum(first[1:] + second[2:]) + second[0]
    expected_loss = numerator / 8 - trainer.settings.balance_beta * 1.5 * math.log(2)
    assert result.valid_tokens.item() == 8
    torch.testing.assert_close(result.loss, expected_loss, rtol=1e-6, atol=1e-6)


@pytest.mark.cpu_only
def test_future_ids_and_target_taps_do_not_affect_supervised_loss() -> None:
    torch.manual_seed(23)
    trainer = _trainer()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    taps = torch.randn(1, 5, 48, requires_grad=True)
    labels = F.one_hot(torch.tensor([[2, 4, 6, 8, 10]]), 32).float()
    attention_mask = torch.ones_like(ids)
    loss_mask = torch.tensor([[0, 0, 1, 0, 0]])
    changed_ids = torch.tensor([[1, 2, 3, 19, 20]])
    changed_taps = torch.cat((taps.detach()[:, :3], torch.randn(1, 2, 48)), dim=1)

    torch.manual_seed(29)
    expected = trainer(ids, attention_mask, loss_mask, taps, labels)
    torch.manual_seed(29)
    actual = trainer(changed_ids, attention_mask, loss_mask, changed_taps, labels)
    assert expected.valid_tokens.item() == 3
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(expected.loss, taps)
    assert torch.count_nonzero(gradient[:, :3]) > 0
    torch.testing.assert_close(gradient[:, 3:], torch.zeros_like(gradient[:, 3:]), rtol=0, atol=0)


def _ddp_loss_parity_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    trainer = _trainer(ttt_steps=1)
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    batch = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]]),
        aux_hidden_states=torch.randn(2, 4, 48),
        target_logits=F.one_hot(2 * ids, 32).float(),
    )
    expected = trainer(**batch)
    expected.loss.backward()
    reference_grads = {name: p.grad.clone() for name, p in trainer.named_parameters() if p.requires_grad}
    trainer.zero_grad(set_to_none=True)

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        ddp = DistributedDataParallel(trainer)
        local = ddp(**{name: tensor[rank : rank + 1] for name, tensor in batch.items()})
        assert local.valid_tokens.item() == (1, 3)[rank]
        local.loss.backward()
        summed_loss = local.loss.detach().clone()
        dist.all_reduce(summed_loss)
        torch.testing.assert_close(summed_loss / 2, expected.loss.detach(), rtol=1e-6, atol=1e-6)
        for name, parameter in trainer.named_parameters():
            if name in reference_grads:
                torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=1e-5, atol=1e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_two_gloo_ranks_match_combined_batch(tmp_path: Path) -> None:
    mp.spawn(_ddp_loss_parity_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


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
    trainer = _trainer()
    model = trainer.draft_model
    model.gradient_checkpointing_enable()
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
