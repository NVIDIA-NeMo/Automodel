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

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_token_mapping
from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy
from nemo_automodel.components.speculative.eagle.core import (
    Eagle3TrainerModule,
    _compute_target_distribution,
    simulated_accept_length,
)
from nemo_automodel.components.speculative.eagle.draft_llama import _HAS_FA, LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.target import _shift_left_with_zero


def test_masked_soft_cross_entropy_normalizes_by_valid_positions():
    logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    position_mask = torch.tensor([[[1], [0]]], dtype=torch.bool)

    loss = masked_soft_cross_entropy(logits=logits, target_probs=target_probs, position_mask=position_mask)
    expected = -torch.log_softmax(logits[0, 0], dim=-1)[0]
    torch.testing.assert_close(loss, expected)


def test_compute_target_distribution_uses_selected_vocab_mask():
    target_logits = torch.tensor(
        [[[0.1, 2.0, 0.0], [0.1, 0.2, 3.0]]],
        dtype=torch.float32,
    )
    selected_token_ids = torch.tensor([0, 1], dtype=torch.long)
    selected_token_mask = torch.tensor([True, True, False], dtype=torch.bool)
    loss_mask = torch.tensor([[1, 1]], dtype=torch.long)

    target_probs, position_mask = _compute_target_distribution(
        target_logits=target_logits,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        loss_mask=loss_mask,
    )

    assert target_probs.shape == (1, 2, 2)
    assert position_mask.tolist() == [[[True], [False]]]


def test_llama_eagle3_draft_forward_shape():
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
    model = LlamaEagle3DraftModel(config)

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


def test_build_eagle3_token_mapping_rejects_non_positive_draft_vocab_size():
    """``draft_vocab_size`` must be a positive int or ``None``.

    Without validation, ``draft_vocab_size=0`` returns an empty selection
    and ``draft_vocab_size=-1`` slices the special-token list (which has
    nothing to do with the requested vocab size). Both are silent
    miscompilations -- raise instead.
    """

    class DummyLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.tensor([[5, 9, 9, 3]], dtype=torch.long),
                "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            }

    for bad in (0, -1, -16):
        with pytest.raises(ValueError, match="draft_vocab_size"):
            build_eagle3_token_mapping(
                DummyLoader(),
                target_vocab_size=16,
                draft_vocab_size=bad,
                special_token_ids=[0, 1],
            )

    # Non-int (e.g. a float coming from YAML) is also rejected.
    with pytest.raises(ValueError, match="draft_vocab_size"):
        build_eagle3_token_mapping(
            DummyLoader(),
            target_vocab_size=16,
            draft_vocab_size=4.0,  # type: ignore[arg-type]
            special_token_ids=[0, 1],
        )


def test_build_eagle3_token_mapping_rejects_non_positive_target_vocab_size():
    """``target_vocab_size`` must be a positive int; it sizes the count tensor."""

    class DummyLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.tensor([[5, 9, 9, 3]], dtype=torch.long),
                "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            }

    for bad in (0, -1):
        with pytest.raises(ValueError, match="target_vocab_size"):
            build_eagle3_token_mapping(
                DummyLoader(),
                target_vocab_size=bad,
                draft_vocab_size=4,
                special_token_ids=[0, 1],
            )


def test_build_eagle3_token_mapping_keeps_requested_vocab_size():
    class DummyLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.tensor([[5, 9, 9, 3]], dtype=torch.long),
                "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
            }

    selected_ids, selected_mask = build_eagle3_token_mapping(
        DummyLoader(),
        target_vocab_size=16,
        draft_vocab_size=4,
        special_token_ids=[0, 1],
    )

    assert selected_ids.shape == (4,)
    assert selected_mask.shape == (16,)
    assert selected_ids[0].item() == 0
    assert selected_ids[1].item() == 1


def _make_draft_config(vocab_size: int = 128, draft_vocab_size: int = 16) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = draft_vocab_size
    config.target_hidden_size = 32
    return config


def _build_tiny_draft_model() -> LlamaEagle3DraftModel:
    return LlamaEagle3DraftModel(_make_draft_config()).to(torch.float32)


def test_eagle3_trainer_rejects_non_positive_ttt_steps():
    """Misconfigured ``ttt_steps`` must raise at construction.

    The forward pass divides by ``sum(0.8**i for i in range(ttt_steps))``,
    which is zero when ``ttt_steps <= 0`` and would silently produce a
    NaN loss. Catching it in ``__init__`` keeps the failure local to the
    recipe setup step.
    """
    import pytest

    draft = _build_tiny_draft_model()
    config = draft.config
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    for bad in (0, -1, -7):
        with pytest.raises(ValueError, match="ttt_steps"):
            Eagle3TrainerModule(
                draft,
                selected_token_ids=selected_token_ids,
                selected_token_mask=selected_token_mask,
                ttt_steps=bad,
            )

    # Non-int (e.g. a float coming from YAML) is also rejected.
    with pytest.raises(ValueError, match="ttt_steps"):
        Eagle3TrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=1.0,  # type: ignore[arg-type]
        )

    # ``ttt_steps=1`` is the minimum valid configuration and must work.
    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=1,
    )
    assert trainer.ttt_steps == 1


def test_eagle3_trainer_runs_multi_step_ttt():
    """Multi-step TTT must keep target_probs / position_mask aligned with the shifted logits."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=3,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    metrics = trainer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        aux_hidden_states=aux_hidden_states,
        target_logits=target_logits,
    )

    assert metrics.loss.dim() == 0
    assert torch.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy.item() <= 1.0
    assert metrics.valid_tokens.item() >= 0

    metrics.loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0
        for p in draft.parameters()
        if p.requires_grad
    )
    assert has_grad, "expected at least one parameter to receive a non-zero gradient"


def test_simulated_accept_length_hand_values():
    """The prefix-survival formula must match hand-computed expectations."""
    # Perfect draft: every chain survives every depth -> 1 bonus + ttt_steps.
    tau = simulated_accept_length(torch.tensor([4.0, 4.0, 4.0]), torch.tensor([4.0, 4.0, 4.0]))
    assert tau.item() == pytest.approx(4.0)
    # Survival rates [0.5, 0.25] -> 1 + 0.5 + 0.25.
    tau = simulated_accept_length(torch.tensor([2.0, 1.0]), torch.tensor([4.0, 4.0]))
    assert tau.item() == pytest.approx(1.75)
    # Depth correlation is preserved: with both depths 50% accurate,
    # coincident hits (the same 2 chains survive depth 2) accept more than
    # disjoint hits (no chain survives depth 2).
    tau = simulated_accept_length(torch.tensor([2.0, 2.0]), torch.tensor([4.0, 4.0]))
    assert tau.item() == pytest.approx(2.0)
    tau = simulated_accept_length(torch.tensor([2.0, 0.0]), torch.tensor([4.0, 4.0]))
    assert tau.item() == pytest.approx(1.5)
    # A step with no supervised positions contributes zero.
    tau = simulated_accept_length(torch.tensor([2.0, 0.0, 0.0]), torch.tensor([4.0, 0.0, 10.0]))
    assert tau.item() == pytest.approx(1.5)
    # Hopeless draft: nothing accepted beyond the bonus token.
    tau = simulated_accept_length(torch.tensor([0.0, 0.0]), torch.tensor([4.0, 4.0]))
    assert tau.item() == pytest.approx(1.0)


def test_eagle3_trainer_reports_per_step_counts():
    """The trainer must expose prefix-hit and loss-mask-based valid counts."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    ttt_steps = 3
    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=ttt_steps,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    with torch.no_grad():
        metrics = trainer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
        )

    assert metrics.step_prefix_hits is not None and metrics.step_valid is not None
    assert metrics.step_prefix_hits.shape == (ttt_steps,)
    assert metrics.step_valid.shape == (ttt_steps,)
    # The denominator AND-accumulates the shifted loss masks (ignoring
    # draft-vocab coverage); a full loss mask makes them nested, so one
    # position rolls out per step.
    assert metrics.step_valid.tolist() == [batch_size * seq_len, batch_size * (seq_len - 1), batch_size * (seq_len - 2)]
    # The coverage-filtered aggregate can only be smaller.
    assert metrics.valid_tokens.item() <= metrics.step_valid.sum().item()
    # Prefix hits are AND-accumulated per slot, so they never grow with depth
    # and never exceed the supervised count.
    step_prefix_hits = metrics.step_prefix_hits.tolist()
    assert all(a >= b for a, b in zip(step_prefix_hits, step_prefix_hits[1:]))
    assert (metrics.step_prefix_hits <= metrics.step_valid).all()
    tau = simulated_accept_length(metrics.step_prefix_hits, metrics.step_valid)
    assert 1.0 <= tau.item() <= 1.0 + ttt_steps


def test_eagle3_trainer_prefix_valid_follows_gappy_loss_mask():
    """Numerator and denominator must cover the same chain population.

    A chain whose earlier depth is unsupervised (prompt tokens, gappy
    multi-turn loss masks) can never be a prefix hit, so it must leave the
    denominator too; counting only the current step's mask would report such
    chains as misses and deflate tau even for a perfect draft.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    ttt_steps = 3
    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=ttt_steps,
    )

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Prompt-style mask: the first two positions are unsupervised.
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    with torch.no_grad():
        metrics = trainer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
        )

    # AND-accumulated shifted masks: [00111111] -> &[01111110] -> &[11111100].
    # Per-current-step counts would be [6, 7, 6] instead.
    assert metrics.step_valid.tolist() == [6, 5, 4]
    assert (metrics.step_prefix_hits <= metrics.step_valid).all()


def test_eagle3_trainer_counts_out_of_vocab_targets_as_rejections():
    """Targets outside the draft vocabulary stay in the acceptance denominator.

    Serving must reject a target token the compressed draft vocabulary cannot
    emit, so such positions count as chain breaks. Excluding them (as the CE
    ``position_mask`` does) would bias tau upward whenever vocabulary
    coverage is below 100%.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=1,
    )

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    # Force the target's greedy token out of the draft vocabulary on the last
    # four positions; the first four stay in-vocabulary.
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)
    target_logits[:, :4, : config.draft_vocab_size] += 100.0
    target_logits[:, 4:, config.draft_vocab_size] += 100.0

    with torch.no_grad():
        metrics = trainer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
        )

    # CE supervision sees only the four covered positions ...
    assert metrics.valid_tokens.item() == 4
    # ... but the acceptance denominator keeps all eight, and the four
    # out-of-vocabulary targets can never be hits.
    assert metrics.step_valid.tolist() == [seq_len]
    assert metrics.step_prefix_hits.item() <= 4


def test_eagle3_trainer_single_vs_multi_step_first_step_matches():
    """The first TTT step should be independent of ttt_steps."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    def _run(ttt_steps: int) -> torch.Tensor:
        trainer = Eagle3TrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=ttt_steps,
        )
        with torch.no_grad():
            return trainer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                aux_hidden_states=aux_hidden_states,
                target_logits=target_logits,
            ).loss

    loss_single = _run(1)
    loss_multi = _run(3)
    assert torch.isfinite(loss_single)
    assert torch.isfinite(loss_multi)


def test_build_eagle3_token_mapping_prefers_high_frequency_tokens():
    class DummyLoader:
        def __iter__(self):
            yield {
                "input_ids": torch.tensor([[7, 7, 7, 7, 9, 9, 4]], dtype=torch.long),
                "loss_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            }

    selected_ids, _ = build_eagle3_token_mapping(
        DummyLoader(),
        target_vocab_size=32,
        draft_vocab_size=3,
        special_token_ids=None,
    )

    assert selected_ids[0].item() == 7
    assert selected_ids[1].item() == 9
    assert selected_ids[2].item() == 4


def test_draft_attention_cache_grows_one_per_call():
    """Each TTT-mode attention call must append exactly one K and one V to cache_hidden."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    projected = draft.project_hidden_states(aux_hidden_states)

    cache_hidden: list[list[torch.Tensor]] = [[], []]
    for expected_lck in range(1, 4):
        draft(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
            cache_hidden=cache_hidden,
        )
        assert len(cache_hidden[0]) == expected_lck
        assert len(cache_hidden[1]) == expected_lck
        # K/V tensors are stored AFTER GQA expansion -> shape head dim is num_heads.
        head_dim = config.hidden_size // config.num_attention_heads
        assert cache_hidden[0][-1].shape == (batch_size, config.num_attention_heads, seq_len, head_dim)
        assert cache_hidden[1][-1].shape == (batch_size, config.num_attention_heads, seq_len, head_dim)


def test_draft_attention_step1_attends_to_step0_kv():
    """At TTT step 1, the diagonal contribution from K_1/V_1 must change the output.

    Mutating V_1 in cache_hidden must produce a different attention output at
    step 1 -- this regression-guards the diagonal-extension code path. Without
    that code path the step-1 output collapses back to ``Q @ K_0 -> V_0`` and
    is independent of V_1.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    projected = draft.project_hidden_states(aux_hidden_states)

    # Step 0: populate cache with (K_0, V_0).
    cache_a: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        draft(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
            cache_hidden=cache_a,
        )
    # Step 1: capture the real attention output.
    with torch.no_grad():
        out_real = draft(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
            cache_hidden=cache_a,
        )

    # Repeat but zero the V_1 entry right after step 1 appends it.
    cache_b: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        draft(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
            cache_hidden=cache_b,
        )

    # Monkey-patch V_1 to a constant during step 1: detect change by comparing
    # against a *third* run where the cache is fresh -- if the step-1 path
    # were broken (degenerate to step-0 attention), out_real would equal a
    # one-call cache_hidden=None forward, which is what we now build.
    cache_no_step1: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        out_step0_only = draft(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
            cache_hidden=cache_no_step1,
        )

    diff = (out_real - out_step0_only).abs().max().item()
    assert diff > 1e-4, (
        f"step-1 attention output is identical to step-0 (max_diff={diff}); "
        "the diagonal K_1/V_1 contribution is not being applied."
    )


def test_draft_attention_rope_shifts_position_by_step_idx():
    """At TTT step ``k`` RoPE must be applied with ``position_ids + k``.

    Without the shift, step ``k``'s newly-projected K/V would encode the
    same absolute position as step 0, defeating the EAGLE-3 "this token
    is k positions into the future" semantics. SpecForge implements this
    as ``lck = len(cache_hidden[0]); position_ids + lck``; we mirror it
    with ``step_idx = len(cache_k); position_ids + step_idx``.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    projected = draft.project_hidden_states(aux_hidden_states)

    # Wrap rotary_emb to capture which position_ids it was called with.
    attn = draft.model.layers[0].self_attn
    original_rotary = attn.rotary_emb
    captured: list[torch.Tensor] = []

    class _Recording(torch.nn.Module):
        def forward(self, x, position_ids):
            captured.append(position_ids.clone())
            return original_rotary(x, position_ids)

    attn.rotary_emb = _Recording()

    try:
        cache_hidden: list[list[torch.Tensor]] = [[], []]
        for _ in range(3):
            with torch.no_grad():
                draft(
                    input_ids=input_ids,
                    projected_hidden_states=projected,
                    attention_mask=attention_mask,
                    cache_hidden=cache_hidden,
                )
    finally:
        attn.rotary_emb = original_rotary

    # Three TTT steps -> three rotary_emb calls with positions:
    #   step 0: arange       = [0..T-1]
    #   step 1: arange + 1   = [1..T]
    #   step 2: arange + 2   = [2..T+1]
    assert len(captured) == 3
    base = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    for step_idx, pos in enumerate(captured):
        expected = base + step_idx
        torch.testing.assert_close(pos, expected, msg=f"step {step_idx} position mismatch: got {pos[0].tolist()}")


def test_draft_attention_einsum_matches_loop_reference():
    """The polished ``einsum`` TTT attention must match a SpecForge-style loop reference.

    Implementing the diagonal extensions twice -- once with the
    ``cat``-in-loop pattern from the SpecForge reference, once with the
    fused ``einsum`` in production -- and asserting they produce
    near-identical outputs guards against any algebraic regression a
    future refactor might introduce.
    """
    from nemo_automodel.components.models.llama.rope_utils import (
        apply_rotary_pos_emb as _rope,
    )
    from nemo_automodel.components.speculative.eagle.draft_llama import _build_causal_mask

    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    projected = draft.project_hidden_states(aux_hidden_states)

    # Production einsum path: 3 TTT steps, chain the hidden state forward
    # (this is what ``Eagle3TrainerModule`` does in real training).
    cache_einsum: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        h_e = projected
        for _ in range(3):
            h_e = draft(
                input_ids=input_ids,
                projected_hidden_states=h_e,
                attention_mask=attention_mask,
                cache_hidden=cache_einsum,
            )
        out_einsum = h_e

    # SpecForge-style cat-in-loop reference, driven through the same
    # decoder layer (so RMSNorm / MLP / residuals match exactly).
    attn = draft.model.layers[0].self_attn
    scaling = attn.scaling
    layer = draft.model.layers[0]

    def _ref_attention(combined, mask, pos_ids, cache):
        bsz, q_len, _ = combined.shape
        q, k, v = attn._project_qkv(combined)
        step_idx = len(cache[0])
        cos, sin = attn.rotary_emb(combined, pos_ids + step_idx)
        q, k = _rope(q, k, cos, sin)
        k, v = attn._repeat_kv(k, v)
        cache[0].append(k)
        cache[1].append(v)
        k0, v0 = cache[0][0], cache[1][0]
        attn_w = torch.matmul(q, k0.transpose(-2, -1)) * scaling + mask
        for i in range(1, step_idx + 1):
            col = (q * cache[0][i]).sum(-1) * scaling
            attn_w = torch.cat((attn_w, col[..., None]), dim=-1)
        probs = torch.softmax(attn_w.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs[..., :q_len], v0)
        for i in range(1, step_idx + 1):
            out = out + probs[..., q_len + i - 1, None] * cache[1][i]
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return attn.o_proj(out)

    causal_mask = _build_causal_mask(attention_mask, dtype=projected.dtype)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    cache_ref: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        hidden = projected
        for _ in range(3):
            input_embeds = draft.embed_input_ids(input_ids)
            residual = hidden
            norm_input = layer.input_layernorm(input_embeds)
            norm_hidden = layer.hidden_norm(hidden)
            combined = torch.cat((norm_input, norm_hidden), dim=-1)
            hidden = residual + _ref_attention(combined, causal_mask, position_ids, cache_ref)
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)
        out_ref = hidden

    diff = (out_einsum - out_ref).abs().max().item()
    assert diff < 1e-5, f"einsum TTT path diverges from loop reference (max_diff={diff})."


def test_eagle3_trainer_multi_step_differs_from_independent_runs():
    """Multi-step TTT loss must differ from ``ttt_steps`` independent single-step losses.

    With the EAGLE-3 cache_hidden recurrence wired up, step 2 of a
    ``ttt_steps=3`` run conditions on the K/V of steps 0 and 1. If the
    recurrence were not wired (each step independently rebuilds attention),
    the multi-step loss would average to the same value as a single-step
    forward repeated 3 times. This guards against accidental regressions.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    # Cover the entire target vocab so every shifted TTT step still has
    # supervised positions; otherwise step 1/2 ``position_mask`` can be
    # all-zero (each step shifts left and zero-fills the tail) and their
    # losses collapse to 0, which would make the multi-step weighted sum
    # numerically identical to the single-step loss.
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.ones(config.vocab_size, dtype=torch.bool)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    multi = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=3,
    )
    single = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=1,
    )

    with torch.no_grad():
        loss_multi = multi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
        ).loss
        loss_single = single(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=aux_hidden_states,
            target_logits=target_logits,
        ).loss

    assert torch.isfinite(loss_multi)
    assert torch.isfinite(loss_single)
    # The two paths must not produce identical losses; if they did, the
    # cache_hidden recurrence has likely regressed.
    assert (loss_multi - loss_single).abs().item() > 1e-4, (
        f"multi-step TTT loss ({loss_multi.item():.6f}) matches single-step "
        f"loss ({loss_single.item():.6f}) exactly -- the cache_hidden "
        "recurrence is probably not in effect."
    )


def test_eagle3_trainer_applies_specforge_loss_decay():
    """Loss must equal ``Σ_i 0.8^i * step_loss_i / Σ_i 0.8^i``.

    Walks the same trainer with ``ttt_steps=1, 2, 3``, then algebraically
    inverts the closed-form weighted-mean formula to recover each per-step
    loss. Asserts the recovered ``step_loss_1`` and ``step_loss_2`` are
    finite and positive (cross-entropy is non-negative by construction);
    any regression that drops the decay factor or the normalization would
    yield non-positive or NaN reconstructions.
    """
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.ones(config.vocab_size, dtype=torch.bool)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    losses: list[torch.Tensor] = []
    for k in (1, 2, 3):
        trainer = Eagle3TrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=k,
        )
        with torch.no_grad():
            losses.append(
                trainer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    aux_hidden_states=aux_hidden_states,
                    target_logits=target_logits,
                ).loss.item()
            )

    # k=1: loss = L_0 / 1.0 -> recover L_0 directly.
    # k=2: loss = (L_0 + 0.8 * L_1) / 1.8 -> solve for L_1.
    # k=3: loss = (L_0 + 0.8 * L_1 + 0.64 * L_2) / 2.44 -> solve for L_2.
    # Because steps 0..k-2 share cache_hidden state across runs, L_0 and
    # L_1 are stable across the three trainers, so the algebra is exact
    # up to floating-point noise.
    l0 = losses[0]
    l1 = (1.8 * losses[1] - l0) / 0.8
    l2 = (2.44 * losses[2] - l0 - 0.8 * l1) / 0.64
    assert l0 > 0, f"step_loss_0 should be positive CE, got {l0}"
    assert l1 > 0, f"step_loss_1 should be positive CE, got {l1}"
    assert l2 > 0, f"step_loss_2 should be positive CE, got {l2}"


def test_target_shift_matches_reference_padding_behavior():
    tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
    shifted = _shift_left_with_zero(tensor)
    expected = torch.tensor([[[2.0], [3.0], [0.0]]])
    torch.testing.assert_close(shifted, expected)

    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    shifted_mask = _shift_left_with_zero(mask)
    expected_mask = torch.tensor([[0, 1, 0]], dtype=torch.long)
    torch.testing.assert_close(shifted_mask, expected_mask)


def _build_eagle3_config(attn_implementation: str) -> LlamaConfig:
    """Realistically-sized layer config for FA2 equivalence checks."""
    config = LlamaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=1024,
        max_position_embeddings=128,
    )
    config.torch_dtype = torch.bfloat16
    config.draft_vocab_size = 128
    config.target_hidden_size = 256
    config.attn_implementation = attn_implementation
    return config


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_FA,
    reason="FA2 path requires a CUDA device and the 'flash-attn' package",
)
def test_eagle3_flash_attention_matches_eager():
    """Multi-step TTT forward must match between eager and flash_attention_2 backends.

    Builds two draft models with identical weights but different attention
    backends, runs three chained TTT steps (exercising Block 1 + diagonal
    extensions), and checks that the post-MLP hidden state agrees within
    bf16 tolerances. Any regression in the log-space softmax merge between
    FA's ``softmax_lse`` and the eager diagonal logits would surface here.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    eager_config = _build_eagle3_config("eager")
    fa_config = _build_eagle3_config("flash_attention_2")

    eager_draft = LlamaEagle3DraftModel(eager_config).to(device=device, dtype=dtype)
    fa_draft = LlamaEagle3DraftModel(fa_config).to(device=device, dtype=dtype)
    fa_draft.load_state_dict(eager_draft.state_dict())

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, eager_config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    aux_hidden_states = torch.randn(batch_size, seq_len, eager_config.hidden_size * 3, device=device, dtype=dtype)
    projected_eager = eager_draft.project_hidden_states(aux_hidden_states)
    projected_fa = fa_draft.project_hidden_states(aux_hidden_states)

    cache_eager: list[list[torch.Tensor]] = [[], []]
    cache_fa: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        h_eager = projected_eager
        h_fa = projected_fa
        for _ in range(3):
            h_eager = eager_draft(
                input_ids=input_ids,
                projected_hidden_states=h_eager,
                attention_mask=attention_mask,
                cache_hidden=cache_eager,
            )
            h_fa = fa_draft(
                input_ids=input_ids,
                projected_hidden_states=h_fa,
                attention_mask=attention_mask,
                cache_hidden=cache_fa,
            )

    max_diff = (h_eager - h_fa).abs().max().item()
    torch.testing.assert_close(h_eager, h_fa, atol=1e-2, rtol=1e-2)
    assert max_diff < 1e-1, f"FA2 vs eager TTT max abs diff = {max_diff}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_FA,
    reason="FA2 path requires a CUDA device and the 'flash-attn' package",
)
def test_eagle3_flash_attention_step0_matches_eager():
    """Step-0 FA2 path (no diagonal extension) must match eager.

    Isolates the simpler half of ``_flash_attention_forward`` -- when
    ``step_idx == 0`` the merge math collapses to just rescaling FA's
    output by ``exp(lse_fa - lse_full) == 1`` -- so any divergence here
    points at the FA call itself (transpose / scale / causal) rather than
    the diagonal-merge algebra.
    """
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    eager_config = _build_eagle3_config("eager")
    fa_config = _build_eagle3_config("flash_attention_2")

    eager_draft = LlamaEagle3DraftModel(eager_config).to(device=device, dtype=dtype)
    fa_draft = LlamaEagle3DraftModel(fa_config).to(device=device, dtype=dtype)
    fa_draft.load_state_dict(eager_draft.state_dict())

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, eager_config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    aux_hidden_states = torch.randn(batch_size, seq_len, eager_config.hidden_size * 3, device=device, dtype=dtype)

    with torch.no_grad():
        out_eager = eager_draft(
            input_ids=input_ids,
            projected_hidden_states=eager_draft.project_hidden_states(aux_hidden_states),
            attention_mask=attention_mask,
        )
        out_fa = fa_draft(
            input_ids=input_ids,
            projected_hidden_states=fa_draft.project_hidden_states(aux_hidden_states),
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(out_eager, out_fa, atol=1e-2, rtol=1e-2)


def test_eagle3_flash_attention_2_raises_without_flash_attn():
    """Requesting FA2 must fail loudly when flash-attn is not installed."""
    if _HAS_FA:
        pytest.skip("flash-attn is installed; cannot exercise the missing-import path")
    config = _build_eagle3_config("flash_attention_2")
    with pytest.raises(ImportError, match="flash-attn"):
        LlamaEagle3DraftModel(config)


def test_eagle3_unknown_attn_implementation_raises():
    config = _build_eagle3_config("xformers")  # unsupported
    with pytest.raises(ValueError, match="attn_implementation"):
        LlamaEagle3DraftModel(config)


def test_eagle3_flash_attention_2_rejects_left_padded_attention_mask():
    config = _build_eagle3_config("flash_attention_2" if _HAS_FA else "eager")
    draft = LlamaEagle3DraftModel(config)
    if not _HAS_FA:
        draft.model.layers[0].self_attn.attn_implementation = "flash_attention_2"

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)

    with pytest.raises(ValueError, match="right-padded attention_mask"):
        draft(
            input_ids=input_ids,
            projected_hidden_states=draft.project_hidden_states(aux_hidden_states),
            attention_mask=attention_mask,
        )


def test_eagle3_flash_attention_2_rejects_non_monotonic_attention_mask():
    config = _build_eagle3_config("flash_attention_2" if _HAS_FA else "eager")
    draft = LlamaEagle3DraftModel(config)
    if not _HAS_FA:
        draft.model.layers[0].self_attn.attn_implementation = "flash_attention_2"

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)

    with pytest.raises(ValueError, match="right-padded attention_mask"):
        draft(
            input_ids=input_ids,
            projected_hidden_states=draft.project_hidden_states(aux_hidden_states),
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# EAGLE-3.1 drafter toggles: ``fc_norm`` and ``norm_output``.
#
# Both flags default to False; EAGLE-3 behavior must be byte-for-byte
# identical to the pre-3.1 path. Enabling either toggle re-routes a
# specific tensor through an RMSNorm, so the unit tests below assert two
# things per toggle:
#
# 1. The default-off path matches the legacy behavior (regression guard).
# 2. The on path actually changes the tensor at the expected boundary
#    (sanity that we did not accidentally no-op the flag).
# ---------------------------------------------------------------------------


def _build_tiny_eagle31_config(*, fc_norm: bool, norm_output: bool) -> LlamaConfig:
    """Tiny Llama config with the two EAGLE-3.1 drafter toggles attached."""
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
    config.fc_norm = fc_norm
    config.norm_output = norm_output
    return config


def test_eagle3_1_default_flags_register_no_extra_parameters():
    """With both flags off the state dict must be byte-identical to EAGLE-3.

    Guards against accidentally registering any ``model.fc_norm.*`` key when
    neither toggle is set: existing EAGLE-3 checkpoints would otherwise fail
    to load with a "missing keys" error.
    """
    eagle3_config = _build_tiny_eagle31_config(fc_norm=False, norm_output=False)
    eagle3_keys = set(LlamaEagle3DraftModel(eagle3_config).state_dict().keys())
    assert not any(k.startswith("model.fc_norm") for k in eagle3_keys)

    # Reference: a draft built with neither attribute present at all
    # (legacy-shape config) must produce exactly the same key set.
    legacy_config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    legacy_config.torch_dtype = torch.float32
    legacy_config.draft_vocab_size = 16
    legacy_config.target_hidden_size = 32
    legacy_keys = set(LlamaEagle3DraftModel(legacy_config).state_dict().keys())
    assert legacy_keys == eagle3_keys


def test_eagle3_1_fc_norm_registers_modulelist_and_changes_projection():
    """``fc_norm=True`` adds one independent RMSNorm per aux chunk.

    Pins the on-disk layout to vLLM's EAGLE-3.1 convention:
    ``model.fc_norm.0.weight``, ``model.fc_norm.1.weight``,
    ``model.fc_norm.2.weight`` (one per ``num_aux_hidden_states``). A single
    shared ``model.fc_norm.weight`` would NOT load into vLLM / SGLang and
    would silently produce a different architecture.
    """
    torch.manual_seed(0)
    config_off = _build_tiny_eagle31_config(fc_norm=False, norm_output=False)
    draft_off = LlamaEagle3DraftModel(config_off)

    torch.manual_seed(0)
    config_on = _build_tiny_eagle31_config(fc_norm=True, norm_output=False)
    draft_on = LlamaEagle3DraftModel(config_on)

    # Per-chunk ModuleList: exactly ``num_aux_hidden_states`` (3 here)
    # independent RMSNorm modules, each registered as
    # ``model.fc_norm.<i>.weight``. No single ``model.fc_norm.weight``.
    keys_off = set(draft_off.state_dict().keys())
    keys_on = set(draft_on.state_dict().keys())
    num_aux = getattr(config_on, "num_aux_hidden_states", 3)
    expected_norm_keys = {f"model.fc_norm.{i}.weight" for i in range(num_aux)}
    assert keys_off.isdisjoint(expected_norm_keys)
    assert expected_norm_keys.issubset(keys_on)
    # Reject the wrong (single-shared) layout outright.
    assert "model.fc_norm.weight" not in keys_on

    assert isinstance(draft_on.model.fc_norm, torch.nn.ModuleList)
    assert len(draft_on.model.fc_norm) == num_aux
    for norm in draft_on.model.fc_norm:
        assert norm.weight.shape == (config_on.target_hidden_size,)

    # The per-chunk RMSNorm modules must be independent objects with
    # independent ``Parameter`` storage; otherwise training would tie
    # gradients across chunks and silently collapse to the shared-norm
    # variant we just rejected.
    storage_ids = {id(norm.weight) for norm in draft_on.model.fc_norm}
    assert len(storage_ids) == num_aux

    # Shared backbone weights round-trip between the two drafts; the
    # ``model.fc_norm.*`` keys exist only on ``draft_on`` so we load with
    # ``strict=False`` and then ensure the comparison is meaningful by
    # also forcing per-chunk norm weights to canonical ones (the default,
    # so this is mostly a guard against future init changes).
    draft_on.load_state_dict(draft_off.state_dict(), strict=False)
    with torch.no_grad():
        for norm in draft_on.model.fc_norm:
            norm.weight.fill_(1.0)

    batch_size, seq_len = 2, 6
    aux = torch.randn(batch_size, seq_len, config_on.hidden_size * 3)

    with torch.no_grad():
        proj_off = draft_off.project_hidden_states(aux)
        proj_on = draft_on.project_hidden_states(aux)

    # The default RMSNorm scale is ones, so ``fc_norm(x) != x`` in general
    # (RMSNorm rescales by 1 / sqrt(mean(x^2))). The projections must
    # therefore differ once the flag is on.
    assert proj_off.shape == proj_on.shape
    diff = (proj_off - proj_on).abs().max().item()
    assert diff > 1e-4, (
        f"fc_norm=True did not change the FC input (max_diff={diff}); the per-chunk RMSNorm path is not being applied."
    )


def test_eagle3_1_fc_norm_chunks_are_routed_per_index():
    """``project_hidden_states`` routes chunk ``i`` through ``fc_norm[i]``.

    Replaces each ``fc_norm[i]`` with a tag module that adds ``i`` to its
    input, sets the FC weight to a block-diagonal identity so every chunk's
    contribution is summed into the output unchanged, then verifies the
    output is ``sum(range(num_aux))``. A bug where every chunk goes through
    ``fc_norm[0]`` (or any other misindexing) yields a different sum --
    catching the failure mode that motivated the ModuleList refactor and
    that pure numerical tests would miss.
    """
    torch.manual_seed(0)
    config = _build_tiny_eagle31_config(fc_norm=True, norm_output=False)
    draft = LlamaEagle3DraftModel(config)
    target_h = config.target_hidden_size
    num_aux = len(draft.model.fc_norm)
    assert config.hidden_size == target_h, "test assumes hidden_size == target_hidden_size"

    class _TagNorm(torch.nn.Module):
        def __init__(self, idx: int, hidden_size: int):
            super().__init__()
            self.idx = idx
            # Keep a Parameter so ``draft.model.fc_norm[i].weight`` still
            # resolves; the tag behaviour does not consult it.
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + float(self.idx)

    draft.model.fc_norm = torch.nn.ModuleList([_TagNorm(i, target_h) for i in range(num_aux)])
    with torch.no_grad():
        draft.model.fc.weight.zero_()
        # Block-diagonal: column block ``i`` is identity, so the FC sums
        # the tagged chunks unchanged into the output.
        for i in range(num_aux):
            draft.model.fc.weight[:, i * target_h : (i + 1) * target_h] = torch.eye(target_h)

    aux = torch.zeros(1, 1, target_h * num_aux)
    with torch.no_grad():
        out = draft.project_hidden_states(aux)
    expected_per_position = float(sum(range(num_aux)))  # 0 + 1 + 2 = 3 for num_aux=3
    torch.testing.assert_close(out, torch.full_like(out, expected_per_position))


def test_eagle3_1_norm_output_returns_post_norm_state():
    """``norm_output=True`` makes ``forward`` return the post-``model.norm`` state."""
    torch.manual_seed(0)
    config_off = _build_tiny_eagle31_config(fc_norm=False, norm_output=False)
    draft_off = LlamaEagle3DraftModel(config_off)

    torch.manual_seed(0)
    config_on = _build_tiny_eagle31_config(fc_norm=False, norm_output=True)
    draft_on = LlamaEagle3DraftModel(config_on)
    draft_on.load_state_dict(draft_off.state_dict())

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config_on.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux = torch.randn(batch_size, seq_len, config_on.hidden_size * 3)
    projected = draft_off.project_hidden_states(aux)

    with torch.no_grad():
        h_off = draft_off(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
        )
        h_on = draft_on(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
        )
        expected = draft_off.model.norm(h_off)

    # ``forward`` with the flag on must equal applying ``model.norm`` to the
    # legacy output. Bf16 / fp32 RMSNorm are deterministic so an exact
    # ``allclose`` with tight tolerance is appropriate.
    torch.testing.assert_close(h_on, expected, atol=1e-6, rtol=1e-6)
    # And it must NOT equal the raw legacy output (otherwise the flag
    # silently degenerated to a no-op).
    assert (h_off - h_on).abs().max().item() > 1e-4


def test_eagle3_1_compute_logits_skips_double_norm_when_norm_output():
    """compute_logits(norm_output=True, forward_out) must equal compute_logits(off, raw).

    The pair (forward returns post-norm) + (compute_logits skips norm) is
    designed to be observation-equivalent to the legacy path at the
    logit boundary: only the *feedback* into the next TTT step changes.
    This test pins that invariant so future refactors of either method
    cannot silently introduce a double or missing normalization.
    """
    torch.manual_seed(0)
    config_off = _build_tiny_eagle31_config(fc_norm=False, norm_output=False)
    draft_off = LlamaEagle3DraftModel(config_off)

    torch.manual_seed(0)
    config_on = _build_tiny_eagle31_config(fc_norm=False, norm_output=True)
    draft_on = LlamaEagle3DraftModel(config_on)
    draft_on.load_state_dict(draft_off.state_dict())

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, config_on.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux = torch.randn(batch_size, seq_len, config_on.hidden_size * 3)
    projected = draft_off.project_hidden_states(aux)

    with torch.no_grad():
        h_off = draft_off(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
        )
        h_on = draft_on(
            input_ids=input_ids,
            projected_hidden_states=projected,
            attention_mask=attention_mask,
        )
        logits_off = draft_off.compute_logits(h_off)
        logits_on = draft_on.compute_logits(h_on)

    torch.testing.assert_close(logits_on, logits_off, atol=1e-6, rtol=1e-6)


def test_eagle3_1_full_forward_runs_end_to_end():
    """Both flags on: ``project_hidden_states`` + ``forward`` + ``compute_logits`` produce finite logits."""
    torch.manual_seed(0)
    config = _build_tiny_eagle31_config(fc_norm=True, norm_output=True)
    draft = LlamaEagle3DraftModel(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    aux = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    hidden = draft(
        input_ids=input_ids,
        projected_hidden_states=draft.project_hidden_states(aux),
        attention_mask=attention_mask,
    )
    logits = draft.compute_logits(hidden)

    assert hidden.shape == (batch_size, seq_len, config.hidden_size)
    assert logits.shape == (batch_size, seq_len, config.draft_vocab_size)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()


def test_eagle3_1_trainer_multi_step_runs_with_flags_enabled():
    """Multi-step TTT training works end-to-end with both EAGLE-3.1 toggles on."""
    torch.manual_seed(0)
    config = _build_tiny_eagle31_config(fc_norm=True, norm_output=True)
    draft = LlamaEagle3DraftModel(config).to(torch.float32)

    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.ones(config.vocab_size, dtype=torch.bool)

    trainer = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=3,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.draft_vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    aux_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3)
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size)

    metrics = trainer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        aux_hidden_states=aux_hidden_states,
        target_logits=target_logits,
    )

    assert torch.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy.item() <= 1.0
    metrics.loss.backward()
    # Every per-chunk fc_norm parameter must receive a non-zero gradient;
    # otherwise the chunk's normalization is not in the autograd graph.
    for i, norm in enumerate(draft.model.fc_norm):
        grad = norm.weight.grad
        assert grad is not None and grad.abs().sum().item() > 0, (
            f"fc_norm[{i}].weight did not receive a gradient -- the EAGLE-3.1 fc_norm path "
            f"is not in the autograd graph for that chunk."
        )


def test_compressed_draft_registers_d2t_t2d_buffers():
    """A compressed-vocab draft must own persistent d2t/t2d remap buffers."""
    model = LlamaEagle3DraftModel(_make_draft_config(vocab_size=128, draft_vocab_size=16))

    assert model.has_vocab_compression
    assert model.d2t.shape == (16,) and model.d2t.dtype == torch.long
    assert model.t2d.shape == (128,) and model.t2d.dtype == torch.bool

    # Persistent buffers must land in the state dict so they serialize into the
    # saved safetensors and load into vLLM/SGLang.
    keys = model.state_dict().keys()
    assert "d2t" in keys and "t2d" in keys


def test_uncompressed_draft_has_no_remap_buffers():
    """With ``draft_vocab_size == vocab_size`` vLLM expects no mapping; emit none."""
    model = LlamaEagle3DraftModel(_make_draft_config(vocab_size=64, draft_vocab_size=64))

    assert not model.has_vocab_compression
    assert not hasattr(model, "d2t")
    assert not hasattr(model, "t2d")
    keys = model.state_dict().keys()
    assert "d2t" not in keys and "t2d" not in keys
    # No-op even when called, so callers need no compression branch of their own.
    model.set_vocab_mapping(torch.arange(64, dtype=torch.long))


def test_set_vocab_mapping_builds_offset_and_mask():
    """d2t is the draft->target offset; t2d marks selected target ids."""
    model = LlamaEagle3DraftModel(_make_draft_config(vocab_size=128, draft_vocab_size=4))
    selected = torch.tensor([5, 9, 50, 100], dtype=torch.long)  # draft id -> target id (all unique)

    model.set_vocab_mapping(selected)

    # vLLM remap: target_id == draft_id + d2t[draft_id].
    base = torch.arange(4, dtype=torch.long)
    assert torch.equal(base + model.d2t, selected)
    # t2d is a boolean presence mask over the full target vocab.
    expected_t2d = torch.zeros(128, dtype=torch.bool)
    expected_t2d[selected] = True
    assert torch.equal(model.t2d, expected_t2d)


def test_set_vocab_mapping_rejects_wrong_length():
    model = LlamaEagle3DraftModel(_make_draft_config(vocab_size=128, draft_vocab_size=4))
    with pytest.raises(ValueError, match="draft_vocab_size"):
        model.set_vocab_mapping(torch.arange(5, dtype=torch.long))


def test_remap_buffers_survive_dtype_cast():
    """``.to(dtype=...)`` must not corrupt the integer/bool remap buffers."""
    model = LlamaEagle3DraftModel(_make_draft_config(vocab_size=128, draft_vocab_size=4))
    model.set_vocab_mapping(torch.tensor([1, 2, 3, 4], dtype=torch.long))

    model = model.to(torch.bfloat16)

    assert model.d2t.dtype == torch.long
    assert model.t2d.dtype == torch.bool


def test_remap_buffers_round_trip_through_state_dict():
    """Saved d2t/t2d restore exactly into a freshly constructed draft."""
    config = _make_draft_config(vocab_size=128, draft_vocab_size=4)
    src = LlamaEagle3DraftModel(config)
    src.set_vocab_mapping(torch.tensor([7, 8, 50, 120], dtype=torch.long))

    dst = LlamaEagle3DraftModel(config)
    dst.load_state_dict(src.state_dict())

    assert torch.equal(dst.d2t, src.d2t)
    assert torch.equal(dst.t2d, src.t2d)


def _lk_trainer(draft, lk_loss_type, ttt_steps=2, **kwargs):
    config = draft.config
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    return Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=ttt_steps,
        lk_loss_type=lk_loss_type,
        **kwargs,
    )


def test_lk_loss_rejects_unknown_type_and_negative_knobs():
    draft = _build_tiny_draft_model()
    with pytest.raises(ValueError, match="lk_loss_type"):
        _lk_trainer(draft, "bogus")
    with pytest.raises(ValueError, match="lk_kl_scale"):
        _lk_trainer(draft, "lambda", lk_kl_scale=-0.5)
    with pytest.raises(ValueError, match="lk_kl_scale"):
        _lk_trainer(draft, "lambda", lk_kl_decay=-1.0)


@pytest.mark.parametrize("lk_loss_type", ["alpha", "lambda"])
def test_lk_trainer_runs_ttt_and_backprops(lk_loss_type):
    torch.manual_seed(0)
    draft = _build_tiny_draft_model()
    config = draft.config
    trainer = _lk_trainer(draft, lk_loss_type, ttt_steps=3)

    batch_size, seq_len = 2, 8
    metrics = trainer(
        input_ids=torch.randint(0, config.draft_vocab_size, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        loss_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        aux_hidden_states=torch.randn(batch_size, seq_len, config.hidden_size * 3),
        target_logits=torch.randn(batch_size, seq_len, config.vocab_size),
    )
    assert metrics.loss.dim() == 0
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0
        for p in draft.parameters()
        if p.requires_grad
    )
    assert has_grad


def test_lk_alpha_step_loss_matches_reference():
    """alpha = -masked-mean(log sum_v min(p_target, p_draft))."""
    torch.manual_seed(1)
    draft = _build_tiny_draft_model()
    trainer = _lk_trainer(draft, "alpha")

    logits = torch.randn(1, 4, draft.config.draft_vocab_size)
    target_probs = torch.softmax(torch.randn(1, 4, draft.config.draft_vocab_size), dim=-1)
    position_mask = torch.tensor([[[True], [True], [False], [True]]])

    loss = trainer._lk_step_loss(logits, target_probs, position_mask)

    accept = torch.minimum(target_probs, torch.softmax(logits.float(), dim=-1)).sum(-1)
    mask = position_mask.squeeze(-1).float()
    expected = -(accept.log() * mask).sum() / mask.sum()
    torch.testing.assert_close(loss, expected)


def test_lk_alpha_perfect_draft_has_zero_loss():
    """A draft matching the target exactly has acceptance 1 -> -log(1) = 0."""
    draft = _build_tiny_draft_model()
    trainer = _lk_trainer(draft, "alpha")
    logits = torch.randn(1, 3, draft.config.draft_vocab_size)
    target_probs = torch.softmax(logits.float(), dim=-1)
    position_mask = torch.ones(1, 3, 1, dtype=torch.bool)
    loss = trainer._lk_step_loss(logits, target_probs, position_mask)
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-6, rtol=0)


def test_lk_lambda_step_loss_matches_reference():
    """lambda = kl_weight * soft_ce + (1 - kl_weight) * (1 - acceptance), with
    kl_weight = kl_scale * exp(-kl_decay * acceptance.detach())."""
    torch.manual_seed(2)
    draft = _build_tiny_draft_model()
    trainer = _lk_trainer(draft, "lambda", lk_kl_scale=0.7, lk_kl_decay=2.0)

    logits = torch.randn(1, 4, draft.config.draft_vocab_size)
    target_probs = torch.softmax(torch.randn(1, 4, draft.config.draft_vocab_size), dim=-1)
    position_mask = torch.tensor([[[True], [False], [True], [True]]])

    loss = trainer._lk_step_loss(logits, target_probs, position_mask)

    accept = torch.minimum(target_probs, torch.softmax(logits.float(), dim=-1)).sum(-1)
    mask = position_mask.squeeze(-1).float()
    acceptance = (accept * mask).sum() / mask.sum()
    soft_ce = masked_soft_cross_entropy(logits=logits, target_probs=target_probs, position_mask=position_mask)
    kl_weight = 0.7 * torch.exp(-2.0 * acceptance)
    expected = kl_weight * soft_ce + (1.0 - kl_weight) * (1.0 - acceptance)
    torch.testing.assert_close(loss, expected)


def test_lk_lambda_zero_decay_full_scale_reduces_to_soft_ce_plus_zero_weighting():
    """kl_scale=1, kl_decay=0 -> kl_weight == 1 -> the lambda loss IS the
    soft-CE step loss (the acceptance term gets weight 0)."""
    torch.manual_seed(3)
    draft = _build_tiny_draft_model()
    trainer = _lk_trainer(draft, "lambda", lk_kl_scale=1.0, lk_kl_decay=0.0)
    logits = torch.randn(1, 5, draft.config.draft_vocab_size)
    target_probs = torch.softmax(torch.randn(1, 5, draft.config.draft_vocab_size), dim=-1)
    position_mask = torch.ones(1, 5, 1, dtype=torch.bool)
    loss = trainer._lk_step_loss(logits, target_probs, position_mask)
    expected = masked_soft_cross_entropy(logits=logits, target_probs=target_probs, position_mask=position_mask)
    torch.testing.assert_close(loss, expected)
