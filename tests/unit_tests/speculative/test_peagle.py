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

P-EAGLE follows the vLLM speculators implementation
(https://github.com/vllm-project/speculators/pull/480): the draft predicts all
``num_depths`` tokens in a single parallel forward over a flat, COD-subsampled
sequence with a flex-attention cross-depth mask -- NOT EAGLE-3's autoregressive
TTT recurrence. These tests pin:

1. the COD sampler (geometric decay, depth-0 fullness, loss-mask filtering);
2. the cross-depth flex mask (causal depth-0 context + own-rollout depth order);
3. the draft contract for vLLM (``mask_hidden`` shape/key, ``config.json`` keys);
4. trainability (finite count-normalized KL loss, gradient into ``mask_hidden``).
"""

import os

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.peagle_attention import create_peagle_mask_mod
from nemo_automodel.components.speculative.eagle.peagle_data import (
    assign_cod_segments,
    generate_cod_sample_indices,
)
from nemo_automodel.components.speculative.eagle.peagle_trainer import PEagleTrainerModule

# P-EAGLE's draft forward runs flex_attention, whose autograd is not implemented
# on CPU in the CI torch build (even the forward errors once the inputs require
# grad). Tests that drive the draft forward therefore run on CUDA and are skipped
# when it is unavailable; the pure-tensor / mask-mod / plan / config tests stay on
# CPU.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="P-EAGLE forward uses flex_attention, which has no CPU autograd support in the CI torch build.",
)


def _build_tiny_draft_model(
    *, parallel_drafting: bool = False, mask_token_id: int = 0, device: str = _DEVICE
) -> LlamaEagle3DraftModel:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # P-EAGLE compiles flex_attention on CUDA; Inductor's flex Triton kernel
        # refuses to lower head_dim < 16. hidden_size // num_attention_heads = 8
        # here, so set a decoupled head_dim >= 16 (the draft reads it via
        # getattr(config, "head_dim", ...)). Keeps hidden_size/target_hidden_size
        # at 32 so the rest of the tiny-config assertions are unchanged.
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = 16
    config.target_hidden_size = 32
    config.parallel_drafting = parallel_drafting
    if parallel_drafting:
        config.mask_token_id = mask_token_id
        config.num_depths = 8
    return LlamaEagle3DraftModel(config).to(device=device, dtype=torch.float32)


def _vocab_mapping(config: LlamaConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Realistic shrunk draft vocab: only the selected ids are marked in-vocab."""
    selected_token_ids = torch.arange(config.draft_vocab_size, dtype=torch.long)
    selected_token_mask = torch.zeros(config.vocab_size, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    return selected_token_ids, selected_token_mask


def _make_trainer(draft: LlamaEagle3DraftModel, *, num_depths: int, mask_token_id: int = 0) -> PEagleTrainerModule:
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    # ``.to(device)`` follows the draft so the registered vocab buffers and the
    # draft share a device when the draft was built on CUDA.
    return PEagleTrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        num_depths=num_depths,
        mask_token_id=mask_token_id,
    ).to(next(draft.parameters()).device)


def _random_batch(
    config: LlamaConfig,
    *,
    batch_size: int = 2,
    seq_len: int = 12,
    targets_in_draft_vocab: bool = True,
    device: str = _DEVICE,
) -> dict[str, torch.Tensor]:
    target_logits = torch.randn(batch_size, seq_len, config.vocab_size, device=device)
    if targets_in_draft_vocab:
        # Bias the target argmax into the draft vocab so supervised positions
        # survive the realistic (shrunk) selected_token_mask.
        target_logits[..., : config.draft_vocab_size] += 8.0
    return {
        "input_ids": torch.randint(0, config.draft_vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "aux_hidden_states": torch.randn(batch_size, seq_len, config.hidden_size * 3, device=device),
        "target_logits": target_logits,
    }


# --------------------------------------------------------------------------- #
# COD sampling
# --------------------------------------------------------------------------- #
def test_cod_depth0_keeps_all_positions_and_decays_deeper():
    torch.manual_seed(0)
    seq_len, num_depths = 16, 8
    loss_mask = torch.ones(1, seq_len, dtype=torch.long)
    anchor_pos, depth = generate_cod_sample_indices(
        seq_length=seq_len, loss_mask=loss_mask, num_depths=num_depths, down_sample_ratio=0.7
    )
    counts = torch.bincount(depth, minlength=num_depths)
    # Depth 0 keeps every position; deeper depths shrink monotonically.
    assert counts[0].item() == seq_len
    nonzero = counts[counts > 0]
    assert torch.all(nonzero[1:] <= nonzero[:-1])
    # Reference positions never leave the sequence.
    assert int((anchor_pos + depth).max()) < seq_len


def test_cod_filters_unsupervised_positions_from_deeper_depths():
    """Deeper-depth candidates come only from ``loss_mask == 1`` positions."""
    torch.manual_seed(0)
    seq_len = 16
    loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
    loss_mask[0, :8] = 1  # only the first half is supervised
    anchor_pos, depth = generate_cod_sample_indices(
        seq_length=seq_len, loss_mask=loss_mask, num_depths=8, down_sample_ratio=0.9
    )
    orig = anchor_pos + depth
    # Depth-0 spans the whole sequence; depths >= 1 must reference supervised ids.
    assert torch.all(orig[depth >= 1] < 8)


def test_cod_supervised_position_zero_does_not_leak_negative_anchor():
    """A supervised position 0 must not produce ``anchor_pos == -1``.

    The depth-1 pool is seeded straight from ``all_valid_indices``; if position 0
    is supervised it could be sampled, and its chain start ``anchor_pos = 0 - 1``
    is negative. The mask then indexes ``document_ids[-1]`` via Python's negative
    wraparound, silently breaking cross-document isolation. ``filter_position_zero``
    (default) must drop it from every depth>=1 pool, not just the deeper ones.
    """
    seq_len, num_depths = 16, 8
    loss_mask = torch.ones(1, seq_len, dtype=torch.long)  # position 0 IS supervised
    # Stress several seeds: with the bug, any draw that picks position 0 at
    # depth 1 leaks anchor_pos == -1. The fix makes min >= 0 a hard invariant.
    for seed in range(8):
        torch.manual_seed(seed)
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_len, loss_mask=loss_mask, num_depths=num_depths, down_sample_ratio=0.9
        )
        assert int(anchor_pos.min()) >= 0
        # Depth 0 still spans the full sequence (the filter only touches depth>=1).
        assert int((depth == 0).sum()) == seq_len


# --------------------------------------------------------------------------- #
# Sequence partitioning (Algorithm 1)
# --------------------------------------------------------------------------- #
def test_assign_cod_segments_phase1_position_and_phase2_dependency():
    """depths 0/1 bucket by position; depth >= 2 inherits its rollout's depth-1 bucket.

    With ``seq_len=8, S=2`` the boundary is at 4. A rollout anchored at position 3
    has its depth-1 element at orig 4 (bucket 1) -- so the whole deep chain lands
    in segment 1, NOT segment 0 (the bucket of anchor 3). This pins the Phase-2
    ``A^g[p] = A^{g-1}[p-1]`` propagation and the ``anchor + 1`` off-by-one.
    """
    seq_len, num_segments = 8, 2
    #                        depth-0 over all positions          | d1@a3 | d2@a3
    anchor_pos = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 3], dtype=torch.long)
    depth = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=torch.long)
    seg = assign_cod_segments(anchor_pos, depth, seq_len, num_segments)
    assert seg[:8].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]  # depth-0 by position
    assert seg[8].item() == 1  # depth-1 anchored at 3 -> bucket(orig=4) = 1
    assert seg[9].item() == 1  # depth-2 inherits the depth-1 bucket = 1


def test_assign_cod_segments_is_in_range_and_defensively_clamped():
    """Every assignment lands in ``[0, S)`` -- including a defensive clamp."""
    seq_len, num_segments = 12, 4
    anchor_pos = torch.arange(seq_len, dtype=torch.long)
    depth = torch.zeros(seq_len, dtype=torch.long)
    seg = assign_cod_segments(anchor_pos, depth, seq_len, num_segments)
    assert int(seg.min()) >= 0 and int(seg.max()) < num_segments
    # Out-of-range position (defensive): the right boundary must clamp to S-1.
    clamped = assign_cod_segments(torch.tensor([seq_len]), torch.tensor([0]), seq_len, num_segments)
    assert clamped.item() == num_segments - 1


def _drive_partitioned(trainer: PEagleTrainerModule, batch: dict[str, torch.Tensor]):
    """Drive the partitioned path exactly as the recipe does: build the plan, then
    one ``forward(peagle_segment=...)`` + ``backward()`` per segment. Returns the
    aggregated (detached) loss, valid count, and correct count."""
    plan = trainer.build_peagle_plan(batch["loss_mask"])
    device = batch["input_ids"].device
    loss_sum = torch.zeros((), device=device)
    valid_sum = torch.zeros((), device=device)
    correct_sum = torch.zeros((), device=device)
    for i in range(len(plan.units)):
        seg = trainer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            aux_hidden_states=batch["aux_hidden_states"],
            target_logits=batch["target_logits"],
            peagle_segment=(plan, i),
        )
        seg.loss.backward()
        loss_sum = loss_sum + seg.loss.detach()
        valid_sum = valid_sum + seg.valid_tokens
        correct_sum = correct_sum + seg.accuracy.detach() * seg.valid_tokens
    return loss_sum, valid_sum, correct_sum


@_gpu_only
def test_sequence_partitioning_matches_single_flat_forward():
    """Algorithm 1 partitioning is exact: driving the S>1 plan segment-by-segment
    (forward + backward each) reproduces the single flat S=1 forward's loss,
    valid-token count, and (up to float summation order) its gradients on every
    draft parameter. This is the linchpin -- if the per-segment key/value sets or
    the loss-charging partition were wrong, the gradients would diverge."""
    torch.manual_seed(0)
    mask_id = 20
    draft = _build_tiny_draft_model(parallel_drafting=True, mask_token_id=mask_id)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)

    def _trainer(sequence_partitions: int) -> PEagleTrainerModule:
        return PEagleTrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_depths=6,
            mask_token_id=mask_id,
            sequence_partitions=sequence_partitions,
        ).to(next(draft.parameters()).device)

    single = _trainer(1)
    partitioned = _trainer(4)
    batch = _random_batch(draft.config, batch_size=2, seq_len=16)

    # Seed identically before each path so COD sampling (the only RNG consumer,
    # in single forward / in build_peagle_plan) draws the same anchors/depths.
    draft.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    single_metrics = single(**batch)
    single_metrics.loss.backward()
    single_grads = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    draft.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    part_loss, part_valid, part_correct = _drive_partitioned(partitioned, batch)
    partitioned_grads = {n: p.grad.detach().clone() for n, p in draft.named_parameters() if p.grad is not None}

    torch.testing.assert_close(part_loss, single_metrics.loss, rtol=1e-4, atol=1e-5)
    assert part_valid.item() == single_metrics.valid_tokens.item()
    assert set(partitioned_grads) == set(single_grads)
    assert single_grads, "expected non-empty gradients"
    for name in single_grads:
        torch.testing.assert_close(partitioned_grads[name], single_grads[name], rtol=1e-3, atol=1e-4)


def test_peagle_plan_charges_every_supervised_token_once():
    """The plan partitions supervised tokens: ``total_den`` equals the single-pass
    supervised count, and every supervised sampled position is the loss-owner of
    exactly one segment (completion-context rows are never charged)."""
    torch.manual_seed(0)
    mask_id = 20
    draft = _build_tiny_draft_model(parallel_drafting=True, mask_token_id=mask_id)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    trainer = PEagleTrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        num_depths=6,
        mask_token_id=mask_id,
        sequence_partitions=4,
    ).to(next(draft.parameters()).device)
    # build_peagle_plan is flex-free (COD sampling + Algorithm 1 assignment), so
    # this test runs on CPU too -- no @_gpu_only guard.
    batch = _random_batch(draft.config, batch_size=2, seq_len=16)
    plan = trainer.build_peagle_plan(batch["loss_mask"])

    charged = sum(int(loss_positions.sum()) for _b, _a, _d, _o, loss_positions in plan.units)
    assert charged == int(plan.total_den.item()), "loss-charged tokens must equal total_den"
    # No completion-context (unowned) row is charged; charged count matches the
    # supervised sampled-token total, so each is owned by exactly one segment.
    assert charged > 0


# --------------------------------------------------------------------------- #
# COD flex mask
# --------------------------------------------------------------------------- #
def test_peagle_mask_mod_visibility_rules():
    """The mask enforces causal depth-0 context + own-rollout depth order, no cross-doc."""
    # One document of length 4: anchor/depth from a hand-built COD layout.
    anchor_pos = torch.tensor([0, 1, 2, 3, 0, 1, 0], dtype=torch.long)
    depth = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)
    mask_mod = create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len=4)

    def visible(q, kv):
        return bool(mask_mod(0, 0, torch.tensor(q), torch.tensor(kv)))

    # A depth-0 query attends causally over depth-0 keys.
    assert visible(2, 1) and visible(2, 2) and not visible(1, 2)
    # A depth-1 element (anchor 0) attends to depth-0 keys at anchor <= 0 and to
    # its own rollout at depth <= 1, but not to a different rollout's masked slot.
    assert visible(4, 0)  # depth1@anchor0 -> depth0@anchor0
    assert not visible(4, 5)  # depth1@anchor0 -/-> depth1@anchor1 (different rollout)
    # A deeper element sees its own shallower rollout slot.
    assert visible(6, 4)  # depth2@anchor0 -> depth1@anchor0


def test_peagle_mask_mod_excludes_padding():
    anchor_pos = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    depth = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    lengths = torch.tensor([2], dtype=torch.long)  # positions 2,3 are padding
    mask_mod = create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len=4)
    # A padded query attends to nothing; valid queries never attend to padding.
    assert not bool(mask_mod(0, 0, torch.tensor(3), torch.tensor(0)))
    assert not bool(mask_mod(0, 0, torch.tensor(1), torch.tensor(3)))


# --------------------------------------------------------------------------- #
# Draft contract (vLLM speculators format)
# --------------------------------------------------------------------------- #
def test_parallel_drafting_registers_mask_hidden_under_exact_key():
    """``parallel_drafting=True`` registers a learnable ``mask_hidden`` of shape ``[1, 1, 3 * H]``."""
    draft = _build_tiny_draft_model(parallel_drafting=True)
    num_aux = getattr(draft.config, "num_aux_hidden_states", 3)
    expected_shape = (1, 1, num_aux * draft.config.target_hidden_size)

    assert isinstance(draft.mask_hidden, torch.nn.Parameter)
    assert draft.mask_hidden.requires_grad
    assert draft.mask_hidden.shape == expected_shape
    assert draft.mask_hidden.shape[-1] == draft.model.fc.in_features

    state_keys = set(draft.state_dict().keys())
    assert "mask_hidden" in state_keys
    assert draft.state_dict()["mask_hidden"].shape == expected_shape


def test_default_draft_has_no_mask_hidden_key():
    draft = _build_tiny_draft_model(parallel_drafting=False)
    assert not hasattr(draft, "mask_hidden")
    assert "mask_hidden" not in set(draft.state_dict().keys())


def test_masked_projected_hidden_shape_and_path():
    """``masked_projected_hidden`` projects the ``[1, 1, 3H]`` placeholder to ``[1, H]`` via ``fc``."""
    draft = _build_tiny_draft_model(parallel_drafting=True)
    out = draft.masked_projected_hidden()
    assert out.shape == (1, draft.config.hidden_size)
    torch.testing.assert_close(out, draft.project_hidden_states(draft.mask_hidden.view(1, -1)))


def test_peagle_stacks_num_hidden_layers():
    """P-EAGLE builds ``num_hidden_layers`` draft layers; EAGLE-3 TTT stays single-layer.

    Layer 0 is the fused ``[embed, hidden]`` (2H) block; layers 1.. are vanilla
    Llama blocks on plain hidden (H), matching speculators' draft stack
    (the reference example trains 4 layers via ``--num-layers 4``).
    """
    from nemo_automodel.components.speculative.eagle.draft_llama import (
        Eagle3LlamaDecoderLayer,
        Eagle3LlamaPeagleLayer,
    )

    peagle = _build_tiny_draft_model(parallel_drafting=True)  # config.num_hidden_layers == 2
    assert len(peagle.model.layers) == peagle.config.num_hidden_layers == 2
    assert isinstance(peagle.model.layers[0], Eagle3LlamaDecoderLayer)
    assert isinstance(peagle.model.layers[1], Eagle3LlamaPeagleLayer)
    # The fused first layer takes 2H q/k/v input; deeper layers take H.
    assert peagle.model.layers[0].self_attn.fuse_input is True
    assert peagle.model.layers[1].self_attn.fuse_input is False

    # The non-parallel EAGLE-3 draft is always single-layer regardless of config.
    eagle3 = _build_tiny_draft_model(parallel_drafting=False)
    assert len(eagle3.model.layers) == 1


@_gpu_only
def test_peagle_deeper_layers_receive_gradient():
    """Stacked deeper layers must be in the autograd graph (their params update)."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)  # 2 layers
    trainer = _make_trainer(draft, num_depths=4)
    trainer(**_random_batch(draft.config)).loss.backward()

    deeper = draft.model.layers[1]
    grads = [p.grad for p in deeper.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0, "deeper P-EAGLE layer received no gradient"


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def test_peagle_trainer_requires_mask_hidden_parameter():
    draft = _build_tiny_draft_model(parallel_drafting=False)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    with pytest.raises(ValueError, match="mask_hidden"):
        PEagleTrainerModule(
            draft,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_depths=3,
            mask_token_id=0,
        )


def test_peagle_trainer_rejects_non_positive_num_depths():
    """``num_depths`` must be a positive integer (a non-int or < 1 must fail loudly)."""
    draft = _build_tiny_draft_model(parallel_drafting=True)
    selected_token_ids, selected_token_mask = _vocab_mapping(draft.config)
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError, match="num_depths"):
            PEagleTrainerModule(
                draft,
                selected_token_ids=selected_token_ids,
                selected_token_mask=selected_token_mask,
                num_depths=bad,  # type: ignore[arg-type]
                mask_token_id=0,
            )


@_gpu_only
def test_peagle_trainer_runs_and_backprops_to_mask_hidden():
    """Parallel forward yields a finite loss and a non-zero gradient on ``mask_hidden``."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_depths=4)
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


@_gpu_only
def test_peagle_single_depth_uses_no_mask_hidden_gradient():
    """With ``num_depths=1`` only depth 0 runs, so ``mask_hidden`` carries no gradient signal."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_depths=1)
    batch = _random_batch(draft.config)

    trainer(**batch).loss.backward()
    assert draft.mask_hidden.grad is None or draft.mask_hidden.grad.abs().sum().item() == 0


@_gpu_only
def test_peagle_flat_inputs_mask_depth_ge_1_slots():
    """Depth-0 slots carry real token/hidden; depth >= 1 slots carry mask token/``mask_hidden``."""
    torch.manual_seed(0)
    # Pick a mask id OUTSIDE the [0, draft_vocab_size) range that ``_random_batch``
    # draws real tokens from, so ``ids == mask_id`` selects exactly the masked
    # depth >= 1 slots (no collision with a real depth-0 token).
    mask_id = 20
    draft = _build_tiny_draft_model(parallel_drafting=True, mask_token_id=mask_id)
    trainer = _make_trainer(draft, num_depths=4, mask_token_id=mask_id)
    batch = _random_batch(draft.config, batch_size=1, seq_len=12)

    captured: dict[str, torch.Tensor] = {}
    original = draft.forward_peagle

    def _recording(*args, **kwargs):
        captured["ids"] = kwargs["sampled_input_ids"].clone()
        captured["hidden"] = kwargs["sampled_projected_hidden"].clone()
        captured["pos"] = kwargs["position_ids"].clone()
        return original(*args, **kwargs)

    draft.forward_peagle = _recording
    try:
        trainer(**batch)
    finally:
        draft.forward_peagle = original

    # Reconstruct depth from the recorded position layout is not direct, but the
    # invariant is: every masked slot equals mask_token_id and the masked hidden.
    ids = captured["ids"][0]  # [total]
    hidden = captured["hidden"][0]  # [total, H]
    mask_proj = draft.masked_projected_hidden()[0]  # [H]

    masked_slots = ids == mask_id
    assert masked_slots.any(), "expected some masked depth >= 1 slots"
    # Every masked-id slot must carry exactly the projected mask_hidden placeholder.
    for h in hidden[masked_slots]:
        torch.testing.assert_close(h, mask_proj)


@_gpu_only
def test_peagle_supervision_depends_only_on_loss_mask():
    """Supervised-token count is driven by loss_mask + COD, NOT by target vocab.

    Unlike EAGLE-3 TTT, speculators' P-EAGLE does not drop positions whose
    full-vocab argmax falls outside the draft vocab (its ``verifier_lm_head``
    is natively draft-vocab). So ``valid_tokens`` must be identical whether or
    not the target argmax lands in the draft vocab.
    """
    draft = _build_tiny_draft_model(parallel_drafting=True, mask_token_id=7)
    trainer = _make_trainer(draft, num_depths=2, mask_token_id=7)

    torch.manual_seed(0)
    in_vocab = trainer(**_random_batch(draft.config, targets_in_draft_vocab=True)).valid_tokens.item()
    torch.manual_seed(0)
    out_vocab = trainer(**_random_batch(draft.config, targets_in_draft_vocab=False)).valid_tokens.item()

    assert in_vocab > 0
    assert out_vocab == in_vocab, "P-EAGLE supervision must not depend on whether targets are in the draft vocab"


@_gpu_only
def test_peagle_loss_decreases_over_optimizer_steps():
    """A few AdamW steps on a fixed batch reduce the P-EAGLE loss."""
    torch.manual_seed(0)
    draft = _build_tiny_draft_model(parallel_drafting=True)
    trainer = _make_trainer(draft, num_depths=4)
    batch = _random_batch(draft.config, batch_size=1, seq_len=16)

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


def test_peagle_mvp_example_config_uses_cod_knobs_not_ttt():
    """The committed P-EAGLE example must enable parallel drafting with COD knobs and no ``ttt_steps``."""
    from nemo_automodel.components.config.loader import load_yaml_config

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "examples/speculative/p-eagle/llama_peagle_mvp.yaml"
    )
    recipe_args = load_yaml_config(cfg_path).recipe_args

    assert recipe_args.get("parallel_drafting", False) is True
    assert recipe_args.get("mask_token_id") is not None
    assert int(recipe_args.get("num_depths")) >= 1
    assert "ttt_steps" not in recipe_args.to_dict()
    # P-EAGLE trains the draft embeddings (speculators embed_requires_grad=True).
    assert recipe_args.get("freeze_embeddings") is False


def test_peagle_checkpoint_save_round_trip(tmp_path):
    """A saved draft carries the vLLM contract: config.json keys + mask_hidden tensor."""
    import json

    from safetensors.torch import load_file

    draft = _build_tiny_draft_model(parallel_drafting=True, mask_token_id=7)
    with torch.no_grad():
        draft.mask_hidden.normal_()
    draft.save_pretrained(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["parallel_drafting"] is True
    assert config["mask_token_id"] == 7

    sd = load_file(tmp_path / "model.safetensors")
    mask_keys = [key for key in sd if "mask_hidden" in key]
    assert mask_keys, f"mask_hidden missing from saved weights: {sorted(sd)}"
    num_aux = getattr(draft.config, "num_aux_hidden_states", 3)
    assert tuple(sd[mask_keys[0]].shape) == (1, 1, num_aux * draft.config.target_hidden_size)
