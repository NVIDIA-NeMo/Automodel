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

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.target_v12 import (
    HFEagleTargetModel,
    _shift_left_with_zero,
)


def _build_tiny_target(num_hidden_layers: int = 4) -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=32,
    )
    config.torch_dtype = torch.float32
    return LlamaForCausalLM(config).to(torch.float32).eval()


def _build_tiny_draft_model() -> LlamaEagleDraftModel:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=32,
    )
    config.torch_dtype = torch.float32
    config.draft_num_hidden_layers = 1
    return LlamaEagleDraftModel(config).to(torch.float32)


def test_llama_eagle_draft_forward_shape():
    model = _build_tiny_draft_model()
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    target_hidden_states = torch.randn(batch_size, seq_len, model.config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    hidden_states = model(
        input_ids=input_ids,
        target_hidden_states=target_hidden_states,
        attention_mask=attention_mask,
    )
    assert hidden_states.shape == (batch_size, seq_len, model.config.hidden_size)


def test_eagle_target_batch_shifts_hidden_states_logits_and_ids():
    torch.manual_seed(0)
    model = _build_tiny_target()
    target = HFEagleTargetModel(model)

    input_ids = torch.randint(1, model.config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.long)

    batch = target.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)

    torch.testing.assert_close(batch.input_ids, _shift_left_with_zero(input_ids))
    torch.testing.assert_close(batch.loss_mask, _shift_left_with_zero(loss_mask))
    torch.testing.assert_close(batch.attention_mask, attention_mask)
    torch.testing.assert_close(batch.target_hidden_states, _shift_left_with_zero(batch.input_hidden_states))

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        expected_logits = _shift_left_with_zero(model.lm_head(outputs[0]))
    torch.testing.assert_close(batch.target_logits, expected_logits)


def test_eagle_trainer_runs_and_backprops():
    torch.manual_seed(0)
    target_model = _build_tiny_target()
    target_model.requires_grad_(False)
    draft_model = _build_tiny_draft_model()
    trainer = EagleTrainerModule(draft_model, target_lm_head=target_model.lm_head)

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, draft_model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    input_hidden_states = torch.randn(batch_size, seq_len, draft_model.config.hidden_size)
    target_hidden_states = torch.randn(batch_size, seq_len, draft_model.config.hidden_size)
    target_logits = torch.randn(batch_size, seq_len, target_model.config.vocab_size)

    metrics = trainer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        input_hidden_states=input_hidden_states,
        target_hidden_states=target_hidden_states,
        target_logits=target_logits,
    )

    assert metrics.loss.dim() == 0
    assert metrics.hidden_loss.dim() == 0
    assert metrics.token_loss.dim() == 0
    assert torch.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy.item() <= 1.0
    assert metrics.valid_tokens.item() == batch_size * seq_len

    metrics.loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0
        for p in draft_model.parameters()
        if p.requires_grad
    )
    assert has_grad, "expected at least one parameter to receive a non-zero gradient"


def test_compute_logits_gathers_sharded_dtensor_target_lm_head():
    """The token-loss head must work when the target lm_head is an FSDP2 ``DTensor``.

    The EAGLE-1/2 recipe can FSDP2-shard the target (its ``distributed:`` path for
    targets that do not fit on one GPU), making ``lm_head.weight`` a ``DTensor``
    while the DDP draft produces plain ``hidden_states``. ``compute_logits`` must
    gather the weight first; otherwise ``F.linear`` raises a mixed Tensor/DTensor
    error.
    """
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    if not dist.is_available():
        import pytest

        pytest.skip("torch.distributed is not available")

    already_initialized = dist.is_initialized()
    if not already_initialized:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        torch.manual_seed(0)
        target_model = _build_tiny_target()
        target_model.requires_grad_(False)
        draft_model = _build_tiny_draft_model()

        mesh = init_device_mesh("cpu", (1,))
        full_weight = target_model.lm_head.weight.detach().clone()
        sharded_weight = distribute_tensor(target_model.lm_head.weight.detach(), mesh, [Shard(0)])
        # A ``DTensor`` is a ``torch.Tensor`` subclass, so it can back a Parameter;
        # this reproduces what FSDP2 leaves on ``lm_head.weight`` after sharding.
        target_model.lm_head.weight = nn.Parameter(sharded_weight, requires_grad=False)
        assert hasattr(target_model.lm_head.weight, "full_tensor")

        trainer = EagleTrainerModule(draft_model, target_lm_head=target_model.lm_head)

        hidden_states = torch.randn(2, 6, draft_model.config.hidden_size)
        logits = trainer.compute_logits(hidden_states)

        assert torch.isfinite(logits).all()
        torch.testing.assert_close(logits, torch.nn.functional.linear(hidden_states, full_weight))
    finally:
        if not already_initialized:
            dist.destroy_process_group()


class TestOptionalRankLoss:
    """ViSpec's stage 1 is EAGLE plus a ListMLE ranking term.

    The reference optimizes ``v_w * vloss + p_w * (ploss + 0.1 * rloss)`` with
    ``v_w=1.0`` and ``p_w=0.1``, so the ranking term carries an effective weight
    of 0.01. EAGLE-1/2 itself does not use it, so it must stay off by default.
    """

    @staticmethod
    def _inputs(loss_mask=None):
        torch.manual_seed(0)
        return dict(
            input_ids=torch.randint(0, 64, (1, 6)),
            attention_mask=torch.ones(1, 6, dtype=torch.long),
            loss_mask=torch.tensor([[0, 0, 1, 1, 1, 0]]) if loss_mask is None else loss_mask,
            input_hidden_states=torch.randn(1, 6, 16),
            target_hidden_states=torch.randn(1, 6, 16),
            target_logits=torch.randn(1, 6, 64),
        )

    @staticmethod
    def _trainer(**kwargs):
        torch.manual_seed(0)
        return EagleTrainerModule(
            _build_tiny_draft_model(), target_lm_head=_build_tiny_target().lm_head, **kwargs
        ).eval()

    def test_off_by_default_leaves_the_objective_untouched(self):
        trainer = self._trainer()
        metrics = trainer(**self._inputs())

        assert metrics.rank_loss is None
        expected = trainer.hidden_loss_weight * metrics.hidden_loss + trainer.token_loss_weight * metrics.token_loss
        torch.testing.assert_close(metrics.loss, expected)

    def test_enabled_adds_the_weighted_ranking_term(self):
        trainer = self._trainer(rank_loss_weight=0.01, rank_loss_topk=4)
        metrics = trainer(**self._inputs())

        assert metrics.rank_loss is not None
        expected = (
            trainer.hidden_loss_weight * metrics.hidden_loss
            + trainer.token_loss_weight * metrics.token_loss
            + 0.01 * metrics.rank_loss
        )
        torch.testing.assert_close(metrics.loss, expected)

    def test_empty_loss_mask_keeps_the_ranking_term_finite(self):
        """topk over an empty selection would otherwise produce NaN."""
        trainer = self._trainer(rank_loss_weight=0.01)
        metrics = trainer(**self._inputs(loss_mask=torch.zeros(1, 6, dtype=torch.long)))

        assert torch.isfinite(metrics.loss)
        assert metrics.rank_loss is not None and float(metrics.rank_loss.detach()) == 0.0
