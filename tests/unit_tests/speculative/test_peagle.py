# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for P-EAGLE (parallel-drafting EAGLE-3) training.

These pin the two things a P-EAGLE checkpoint must satisfy to load into
vLLM's parallel-drafting runtime (https://github.com/vllm-project/vllm/pull/32887):

1. the draft exposes a single learnable ``mask_hidden`` tensor of shape
   ``[1, num_aux_hidden_states * target_hidden_size]`` under exactly that key;
2. the masked multi-token-prediction depths are conditioned on the fixed
   ``ptd_token_id`` token and the ``mask_hidden`` placeholder (no recurrence),
   while the next-token-prediction depth is byte-identical to EAGLE-3 step 0.
"""

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule, PEagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel


def _build_tiny_draft_model(*, parallel_drafting: bool = False, ptd_token_id: int = 0) -> LlamaEagle3DraftModel:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    config.parallel_drafting = parallel_drafting
    if parallel_drafting:
        config.ptd_token_id = ptd_token_id
    return LlamaEagle3DraftModel(config).to(torch.float32)


def _vocab_mapping(config: LlamaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.ones(config.vocab_size, dtype=torch.bool)
    return selected_token_ids, selected_token_mask


def _make_trainer(draft: LlamaEagle3DraftModel, *, num_draft_tokens: int, ptd_token_id: int = 0) -> PEagleTrainerModule:
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    return PEagleTrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        num_draft_tokens=num_draft_tokens,
        ptd_token_id=ptd_token_id,
    )


def _random_batch(config: LlamaConfig, *, batch_size: int = 2, seq_len: int = 8) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(0, config.draft_vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "aux_hidden_states": torch.randn(batch_size, seq_len, config.hidden_size * 3),
        "target_logits": torch.randn(batch_size, seq_len, config.vocab_size),
    }


def test_parallel_drafting_registers_mask_hidden_under_exact_key():
    """``parallel_drafting=True`` must register a learnable ``mask_hidden`` of shape ``[1, 3 * H]``.

    vLLM's loader keys on the substring ``mask_hidden`` and does
    ``self.mask_hidden.copy_(w.view(1, -1))`` against a buffer sized
    ``[1, (3 if use_aux else 1) * hidden_size]``. EAGLE-3 here always uses 3
    aux hidden states, so the on-disk tensor must be ``[1, 3 * hidden_size]``
    (== ``model.fc.in_features``) under exactly the key ``mask_hidden``.
    """
    draft = _build_tiny_draft_model(parallel_drafting=True)
    num_aux = getattr(draft.config, "num_aux_hidden_states", 3)
    expected_shape = (1, num_aux * draft.config.target_hidden_size)

    assert isinstance(draft.mask_hidden, torch.nn.Parameter)
    assert draft.mask_hidden.requires_grad
    assert draft.mask_hidden.shape == expected_shape
    assert draft.mask_hidden.shape == (1, draft.model.fc.in_features)

    state_keys = set(draft.state_dict().keys())
    assert "mask_hidden" in state_keys
    assert draft.state_dict()["mask_hidden"].shape == expected_shape


def test_default_draft_has_no_mask_hidden_key():
    """Without ``parallel_drafting`` the state dict must be free of ``mask_hidden``.

    Guards EAGLE-3 / EAGLE-3.1 checkpoints from gaining an unexpected key
    (which would otherwise break their strict-load round-trip).
    """
    draft = _build_tiny_draft_model(parallel_drafting=False)
    assert not hasattr(draft, "mask_hidden")
    assert "mask_hidden" not in set(draft.state_dict().keys())


def test_masked_projected_hidden_shape_and_path():
    """``masked_projected_hidden`` projects the ``[1, 3H]`` placeholder to ``[1, H]`` via ``fc``."""
    draft = _build_tiny_draft_model(parallel_drafting=True)
    out = draft.masked_projected_hidden()
    assert out.shape == (1, draft.config.hidden_size)
    # It must be exactly the placeholder run through ``project_hidden_states``.
    torch.testing.assert_close(out, draft.project_hidden_states(draft.mask_hidden))


def test_peagle_trainer_requires_mask_hidden_parameter():
    """Constructing the trainer on a non-parallel draft must fail loudly."""
    draft = _build_tiny_draft_model(parallel_drafting=False)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    with pytest.raises(ValueError, match="mask_hidden"):
        PEagleTrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_draft_tokens=3,
            ptd_token_id=0,
        )


def test_peagle_trainer_rejects_non_positive_num_draft_tokens():
    draft = _build_tiny_draft_model(parallel_drafting=True)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    for bad in (0, -1, -4):
        with pytest.raises(ValueError, match="num_draft_tokens"):
            PEagleTrainerModule(
                draft,
                selected_token_ids=selected_token_ids,
                selected_token_mask=selected_token_mask,
                num_draft_tokens=bad,
                ptd_token_id=0,
            )
    with pytest.raises(ValueError, match="num_draft_tokens"):
        PEagleTrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_draft_tokens=2.0,  # type: ignore[arg-type]
            ptd_token_id=0,
        )


def test_peagle_trainer_runs_and_backprops_to_mask_hidden():
    """Parallel forward yields a finite loss and a non-zero gradient on ``mask_hidden``.

    The gradient on ``mask_hidden`` is the headline check: it is the one new
    learnable that distinguishes P-EAGLE from EAGLE-3, so it must be wired
    into the autograd graph at the masked depths.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_draft_tokens=3)
    batch = _random_batch(draft.config)

    metrics = trainer(**batch)

    assert metrics.loss.dim() == 0
    assert torch.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy.item() <= 1.0
    assert metrics.valid_tokens.item() >= 0

    metrics.loss.backward()
    grad = draft.mask_hidden.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum().item() > 0, (
        "mask_hidden received no gradient -- the masked-depth placeholder is not in the autograd graph."
    )


def test_peagle_single_token_uses_no_mask_hidden_gradient():
    """With ``num_draft_tokens=1`` only the NTP depth runs, so ``mask_hidden`` is unused.

    The masked placeholder is consumed only at depths ``>= 1``; a single-token
    draft must therefore leave ``mask_hidden`` out of the graph (no gradient).
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_draft_tokens=1)
    batch = _random_batch(draft.config)

    trainer(**batch).loss.backward()
    assert draft.mask_hidden.grad is None or draft.mask_hidden.grad.abs().sum().item() == 0


def test_peagle_masked_depths_use_fixed_mask_inputs_not_recurrence():
    """Depth 0 sees the real inputs; depths >= 1 see the fixed mask token + ``mask_hidden``.

    Records the ``input_ids`` and ``projected_hidden_states`` the trainer
    passes to the draft at each depth. The discriminating property of P-EAGLE
    (vs EAGLE-3 TTT) is that the masked depths do **not** consume the previous
    depth's output -- their inputs are constant across all masked depths.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True, ptd_token_id=7)
    trainer = _make_trainer(draft, num_draft_tokens=4, ptd_token_id=7)
    batch = _random_batch(draft.config)

    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_forward = draft.forward

    def _recording_forward(*args, **kwargs):
        calls.append((kwargs["input_ids"].clone(), kwargs["projected_hidden_states"].clone()))
        return original_forward(*args, **kwargs)

    draft.forward = _recording_forward
    try:
        trainer(**batch)
    finally:
        draft.forward = original_forward

    assert len(calls) == 4

    # Depth 0: the real next-token-prediction inputs.
    torch.testing.assert_close(calls[0][0], batch["input_ids"])
    torch.testing.assert_close(calls[0][1], draft.project_hidden_states(batch["aux_hidden_states"]))

    # Depths >= 1: the masked token id everywhere and a constant projected
    # ``mask_hidden`` -- identical across all masked depths (no recurrence).
    expected_ids = torch.full_like(batch["input_ids"], 7)
    expected_hidden = draft.masked_projected_hidden().view(1, 1, -1).expand_as(calls[1][1])
    for depth in range(1, 4):
        torch.testing.assert_close(calls[depth][0], expected_ids)
        torch.testing.assert_close(calls[depth][1], expected_hidden)


def test_peagle_step0_matches_eagle3_step0():
    """P-EAGLE with ``num_draft_tokens=1`` equals EAGLE-3 with ``ttt_steps=1``.

    The next-token-prediction depth consumes the real token and real aux
    hidden states with no mask involvement, so the single-depth P-EAGLE loss
    must match EAGLE-3 step 0 exactly when the shared weights are identical.
    """
    torch.manual_seed(0)
    eagle3_draft = _build_tiny_draft_model(parallel_drafting=False)

    torch.manual_seed(0)
    peagle_draft = _build_tiny_draft_model(parallel_drafting=True)
    # Copy the shared backbone weights; ``mask_hidden`` is unused at K=1.
    peagle_draft.load_state_dict(eagle3_draft.state_dict(), strict=False)

    selected_token_ids, selected_token_mask = _vocab_mapping(eagle3_draft.config)
    batch = _random_batch(eagle3_draft.config)

    eagle3 = Eagle3TrainerModule(
        eagle3_draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=1,
    )
    peagle = _make_trainer(peagle_draft, num_draft_tokens=1)

    with torch.no_grad():
        loss_eagle3 = eagle3(**batch).loss
        loss_peagle = peagle(**batch).loss

    torch.testing.assert_close(loss_peagle, loss_eagle3, atol=1e-6, rtol=1e-6)


def test_peagle_loss_decreases_over_optimizer_steps():
    """A few AdamW steps on a fixed batch must reduce the P-EAGLE loss.

    End-to-end trainability check: every trainable parameter -- including the
    new ``mask_hidden`` placeholder -- updates and the loss drops.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_draft_tokens=3)
    batch = _random_batch(draft.config)

    optimizer = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=1e-2)

    initial_loss = trainer(**batch).loss.item()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = trainer(**batch).loss
        loss.backward()
        optimizer.step()
    final_loss = trainer(**batch).loss.item()

    assert final_loss < initial_loss, f"loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_mask_hidden_state_dict_round_trip():
    """``mask_hidden`` survives a state-dict save / strict reload into a fresh parallel draft."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    with torch.no_grad():
        draft.mask_hidden.normal_()

    state = draft.state_dict()
    reloaded = _build_tiny_draft_model(parallel_drafting=True)
    missing, unexpected = reloaded.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    torch.testing.assert_close(reloaded.mask_hidden, draft.mask_hidden)
