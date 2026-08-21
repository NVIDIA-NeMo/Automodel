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

"""Tests for the DFlash 2 draft model: in-block convolution, selector, decoding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import (
    CandidateSelector,
    GroupedDynamicCausalConv,
    Qwen3DFlash2DraftModel,
    dflash2_rejection_sample,
)

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
TOP_K = 4


def _draft_cfg(block_size=BLOCK_SIZE, **dflash_config):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1000000,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = block_size
    cfg.dflash_config = {
        "mask_token_id": MASK_ID,
        "target_layer_ids": TARGET_LAYER_IDS,
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": TOP_K,
        **dflash_config,
    }
    cfg._attn_implementation = "sdpa"
    return cfg


def _draft_inputs(bsz=2, seq_len=10, n_blocks=3, block_size=BLOCK_SIZE):
    torch.manual_seed(0)
    noise = torch.randn(bsz, n_blocks * block_size, HIDDEN)
    target_hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    position_ids = torch.arange(seq_len + n_blocks * block_size).unsqueeze(0).expand(bsz, -1)
    return noise, target_hidden, position_ids


# ---------------------------------------------------------------------------
# Two-tap dynamic convolution
# ---------------------------------------------------------------------------


def test_conv_taps_never_cross_a_block_boundary():
    """Block position 0 must read zero padding, not the previous block's tail.

    Training packs ``blocks * block_size`` draft positions into one sequence. If
    the predecessor tap shifted along that flat axis, block *i*'s first position
    would be conditioned on block *i-1*'s last drafted token -- a neighbour it
    never sees at decode time -- and the packed layout would stop being
    equivalent to drafting one block at a time.
    """
    torch.manual_seed(0)
    conv = GroupedDynamicCausalConv(HIDDEN, kernel_size=2, group_size=8)
    torch.nn.init.normal_(conv.kernel_projection.weight, std=0.02)
    hidden = torch.randn(1, 3 * BLOCK_SIZE, HIDDEN)

    packed, packed_dynamic = conv.prepare(hidden, BLOCK_SIZE)
    packed_out = conv.finish(packed, packed_dynamic, BLOCK_SIZE)

    for block in range(3):
        block_slice = hidden[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE]
        alone, alone_dynamic = conv.prepare(block_slice, BLOCK_SIZE)
        alone_out = conv.finish(alone, alone_dynamic, BLOCK_SIZE)
        torch.testing.assert_close(packed_out[:, block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE], alone_out)


def test_conv_predecessor_tap_reads_the_previous_in_block_position():
    """With only the predecessor tap active the conv is an in-block right shift."""
    conv = GroupedDynamicCausalConv(HIDDEN, kernel_size=2, group_size=8)
    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[:, 1, :] = 1.0  # predecessor tap only, no dynamic correction
    hidden = torch.randn(1, 2 * BLOCK_SIZE, HIDDEN)

    shifted, _ = conv.prepare(hidden, BLOCK_SIZE)

    for block in range(2):
        start = block * BLOCK_SIZE
        # The block's first position has no predecessor inside the block.
        torch.testing.assert_close(shifted[:, start], torch.zeros(1, HIDDEN))
        torch.testing.assert_close(
            shifted[:, start + 1 : start + BLOCK_SIZE], hidden[:, start : start + BLOCK_SIZE - 1]
        )


def test_conv_rejects_a_group_size_that_does_not_divide_hidden():
    with pytest.raises(ValueError, match="conv_group_size"):
        GroupedDynamicCausalConv(HIDDEN, kernel_size=2, group_size=7)


# ---------------------------------------------------------------------------
# Draft model
# ---------------------------------------------------------------------------


def test_forward_output_shape_and_finiteness():
    draft = Qwen3DFlash2DraftModel(_draft_cfg())
    noise, target_hidden, position_ids = _draft_inputs()
    out = draft(position_ids=position_ids, attention_mask=None, noise_embedding=noise, target_hidden=target_hidden)
    assert out.shape == noise.shape
    assert torch.isfinite(out).all()


def test_untrained_dflash2_matches_dflash_on_shared_weights():
    """The convolutions and the selector start at the identity.

    A fresh DFlash 2 draft must be numerically indistinguishable from plain
    DFlash so training starts from the DFlash optimum rather than from a random
    perturbation of it: the conv passes its input through (base tap 1, zero
    dynamic correction) and every selector score collapses to the draft's own
    logit (zero successor codebook).
    """
    torch.manual_seed(0)
    dflash = Qwen3DFlashDraftModel(_draft_cfg())
    dflash2 = Qwen3DFlash2DraftModel(_draft_cfg())
    missing, unexpected = dflash2.load_state_dict(dflash.state_dict(), strict=False)
    assert unexpected == []
    assert all(("conv" in key) or ("codebook" in key) or ("hidden_projection" in key) for key in missing)

    noise, target_hidden, position_ids = _draft_inputs()
    kwargs = dict(position_ids=position_ids, attention_mask=None, noise_embedding=noise, target_hidden=target_hidden)
    torch.testing.assert_close(dflash2(**kwargs), dflash(**kwargs))

    hidden = torch.randn(2, 3, HIDDEN)
    logits = torch.randn(2, 3, VOCAB)
    candidate_ids = torch.randint(0, VOCAB, (2, 3, TOP_K))
    predecessors = torch.randint(0, VOCAB, (2, 3))
    scores = dflash2.candidate_selector.pair_scores(
        hidden, logits.gather(-1, candidate_ids), candidate_ids, predecessors
    )
    torch.testing.assert_close(scores, logits.gather(-1, candidate_ids))


def test_state_dict_keys_match_the_published_dflash2_layout():
    """The extra tensors must be named exactly as the released drafters name them.

    The published DFlash 2 checkpoints (e.g. ``z-lab/Qwen3.8-27B-DFlash2``) store
    the convolutions as ``layers.N.{attention,mlp}_conv.{base_kernel,
    kernel_projection.weight}`` and the selector codebooks as bare
    ``candidate_selector.{predecessor,successor}_codebook`` tensors -- no
    ``.weight`` suffix, because they are parameters rather than ``nn.Embedding``
    modules. Keeping that layout is what lets a draft trained here load into the
    serving runtimes, and a released drafter load here, without a key map.
    """
    draft = Qwen3DFlash2DraftModel(_draft_cfg())
    keys = draft.state_dict()
    rank = draft.candidate_selector.rank

    assert keys["candidate_selector.predecessor_codebook"].shape == (VOCAB, rank)
    assert keys["candidate_selector.successor_codebook"].shape == (VOCAB, rank)
    assert keys["candidate_selector.hidden_projection.weight"].shape == (rank, HIDDEN)
    assert "candidate_selector.predecessor_codebook.weight" not in keys

    groups = HIDDEN // 8
    for layer in range(2):
        for conv in ("attention_conv", "mlp_conv"):
            # [pre/post sublayer, taps, hidden]
            assert keys[f"layers.{layer}.{conv}.base_kernel"].shape == (2, 2, HIDDEN)
            # 2 (pre/post) * taps * groups rows of dynamic corrections
            assert keys[f"layers.{layer}.{conv}.kernel_projection.weight"].shape == (2 * 2 * groups, HIDDEN)


def test_conv_block_size_defaults_and_validation():
    draft = Qwen3DFlash2DraftModel(_draft_cfg())
    # A whole number of blocks is the trainer's packed layout.
    assert draft.resolve_conv_block_size(3 * BLOCK_SIZE, None) == BLOCK_SIZE
    # A truncated final block at decode time is a single block.
    assert draft.resolve_conv_block_size(BLOCK_SIZE - 1, None) == BLOCK_SIZE - 1
    assert draft.resolve_conv_block_size(2 * BLOCK_SIZE, BLOCK_SIZE) == BLOCK_SIZE
    with pytest.raises(ValueError, match="conv_block_size"):
        draft.resolve_conv_block_size(2 * BLOCK_SIZE + 1, BLOCK_SIZE)


def test_domino_head_is_rejected():
    with pytest.raises(ValueError, match="projector_type"):
        Qwen3DFlash2DraftModel(_draft_cfg(projector_type="domino", emb_dim=8, gru_hidden_dim=8))


def _selector_objective(draft, noise, target_hidden, position_ids, lm_head):
    """Backbone forward plus a selector score, the shape of the real training loss.

    Args:
        draft: The DFlash 2 draft model under test.
        noise: Tensor of shape [batch, draft, hidden]; the embedded draft blocks.
        target_hidden: Tensor of shape [batch, context, layers * hidden]; the
            captured target context features.
        position_ids: Long tensor of shape [batch, context + draft].
        lm_head: Tensor of shape [vocab, hidden]; a stand-in output projection.

    Returns:
        Scalar tensor to call ``backward()`` on.
    """
    hidden = draft(position_ids=position_ids, attention_mask=None, noise_embedding=noise, target_hidden=target_hidden)
    logits = torch.nn.functional.linear(hidden, lm_head)
    candidate_ids = torch.randint(0, VOCAB, (*logits.shape[:-1], TOP_K))
    scores = draft.candidate_selector.pair_scores(
        hidden, logits.gather(-1, candidate_ids), candidate_ids, torch.randint(0, VOCAB, logits.shape[:-1])
    )
    return hidden.square().mean() + scores.square().mean()


def test_selector_leaves_its_identity_start_after_one_step():
    """The identity start costs one step of gradient on two selector tensors.

    ``S_t`` is bilinear in the two codebooks, so making it collapse to the draft's
    own logit forces one factor to zero -- here the successor codebook -- and the
    predecessor codebook and the context gate multiply that zero on the first
    backward. This asserts the warm-up is exactly one step: the successor codebook
    and both convolutions train immediately, and once the successor codebook has
    moved the remaining selector tensors receive gradient too.
    """
    torch.manual_seed(0)
    draft = Qwen3DFlash2DraftModel(_draft_cfg())
    noise, target_hidden, position_ids = _draft_inputs()
    lm_head = torch.randn(VOCAB, HIDDEN)

    _selector_objective(draft, noise, target_hidden, position_ids, lm_head).backward()
    trained_now = {
        name for name, param in draft.named_parameters() if param.grad is not None and param.grad.abs().sum() > 0
    }
    assert "candidate_selector.successor_codebook" in trained_now
    assert {name for name in trained_now if "conv" in name}, "the convolutions must train from the first step"
    assert "candidate_selector.predecessor_codebook" not in trained_now

    optimizer = torch.optim.SGD(draft.parameters(), lr=1.0)
    optimizer.step()
    optimizer.zero_grad()
    _selector_objective(draft, noise, target_hidden, position_ids, lm_head).backward()

    for name in ("predecessor_codebook", "successor_codebook", "hidden_projection.weight"):
        param = dict(draft.candidate_selector.named_parameters())[name]
        assert param.grad is not None and param.grad.abs().sum() > 0, name


# ---------------------------------------------------------------------------
# Candidate selector
# ---------------------------------------------------------------------------


def test_selector_walk_conditions_each_step_on_the_token_it_just_picked():
    """The walk is the only sequential part: step t's score uses step t-1's pick.

    Scoring is precomputed in parallel, so the walk is what turns independent
    candidate lists into one path; if it re-used a fixed predecessor the whole
    coherence gain would disappear silently.
    """
    torch.manual_seed(0)
    selector = CandidateSelector(vocab_size=VOCAB, hidden_size=HIDDEN, rank=16, top_k=TOP_K)
    torch.nn.init.normal_(selector.successor_codebook, std=1.0)
    hidden = torch.randn(1, 3, HIDDEN)
    logits = torch.randn(1, 3, VOCAB)
    anchor_ids = torch.tensor([7])

    path, candidate_ids, draft_probs = selector.walk(hidden, logits, anchor_ids, temperature=0.0)

    assert draft_probs is None
    assert path.shape == (1, 3)
    assert candidate_ids.shape == (1, 3, TOP_K)
    # Every selected token comes from that position's own candidate list.
    assert bool((candidate_ids == path.unsqueeze(-1)).any(dim=-1).all())
    # Replaying the scores by hand with the walk's own predecessors reproduces it.
    predecessor = anchor_ids
    for position in range(3):
        scores = selector.pair_scores(
            hidden[:, position],
            logits[:, position].gather(-1, candidate_ids[:, position]),
            candidate_ids[:, position],
            predecessor,
        )
        predecessor = candidate_ids[:, position].gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        assert int(predecessor[0]) == int(path[0, position])


def test_selector_walk_samples_within_candidates_and_returns_a_proposal():
    torch.manual_seed(0)
    selector = CandidateSelector(vocab_size=VOCAB, hidden_size=HIDDEN, rank=16, top_k=TOP_K)
    path, candidate_ids, draft_probs = selector.walk(
        torch.randn(1, 3, HIDDEN), torch.randn(1, 3, VOCAB), torch.tensor([7]), temperature=1.0
    )
    assert draft_probs.shape == (1, 3, TOP_K)
    torch.testing.assert_close(draft_probs.sum(-1), torch.ones(1, 3))
    assert bool((candidate_ids == path.unsqueeze(-1)).any(dim=-1).all())


def test_selector_rejects_a_top_k_larger_than_the_vocabulary():
    with pytest.raises(ValueError, match="selector_top_k"):
        CandidateSelector(vocab_size=8, hidden_size=HIDDEN, rank=16, top_k=16)


# ---------------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------------


def test_rejection_sampling_accepts_the_matching_prefix_of_a_deterministic_target():
    """A one-hot target accepts a drafted token only when it is that token.

    This is the lossless-decoding contract: with ``p`` concentrated on one token
    the emitted sequence must be that token repeated, whatever the draft
    proposed, and the bonus token must come from the residual rather than from
    the draft's own candidate list.
    """
    torch.manual_seed(0)
    forced = 5
    draft_tokens = torch.tensor([[forced, forced, 11]])
    candidate_ids = torch.tensor([[[forced, 1], [forced, 2], [11, 3]]])
    draft_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    target_probs = torch.zeros(1, 4, VOCAB)
    target_probs[..., forced] = 1.0

    accepted, bonus = dflash2_rejection_sample(draft_tokens, target_probs, draft_probs, candidate_ids)

    assert accepted == 2
    assert int(bonus) == forced


def test_rejection_sampling_accepts_the_whole_block_and_draws_a_bonus_token():
    forced = 5
    draft_tokens = torch.tensor([[forced, forced]])
    candidate_ids = torch.tensor([[[forced, 1], [forced, 2]]])
    draft_probs = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    target_probs = torch.zeros(1, 3, VOCAB)
    target_probs[..., forced] = 1.0

    accepted, bonus = dflash2_rejection_sample(draft_tokens, target_probs, draft_probs, candidate_ids)

    assert accepted == 2
    assert int(bonus) == forced


# ---------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------


class _ConstantTarget(torch.nn.Module):
    """Minimal stand-in for the verifier: always samples ``forced_token_id``.

    Only the surface ``spec_generate`` touches is implemented (``model.embed_tokens``,
    ``lm_head``, and a forward returning logits + per-layer hidden states).
    """

    def __init__(self, cfg: Qwen3Config, forced_token_id: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.num_layers = cfg.num_target_layers
        self.vocab_size = cfg.vocab_size
        self.forced_token_id = forced_token_id
        self.device = torch.device("cpu")

    def forward(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=None,
        output_hidden_states=False,
    ):
        hidden = self.model.embed_tokens(input_ids)
        keep = input_ids.shape[1] if logits_to_keep is None else logits_to_keep
        # A very peaked distribution so the sampled path is deterministic too.
        logits = torch.zeros(input_ids.shape[0], keep, self.vocab_size)
        logits[..., self.forced_token_id] = 30.0
        return SimpleNamespace(logits=logits, hidden_states=[hidden] * (self.num_layers + 1))


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_spec_generate_emits_the_targets_tokens(temperature):
    """Greedy and rejection-sampled decoding both emit what the target would.

    The draft is free to propose anything; a verifier that always wants the same
    token must still produce exactly that token ``max_new_tokens`` times, which
    is what "the output is provably unchanged" means for both accept paths.
    """
    torch.manual_seed(0)
    cfg = _draft_cfg()
    draft = Qwen3DFlash2DraftModel(cfg)
    target = _ConstantTarget(cfg, forced_token_id=3)

    prompt = torch.tensor([[1, 2, 3]])
    max_new_tokens = 5
    out = draft.spec_generate(target, prompt, max_new_tokens, stop_token_ids=None, temperature=temperature)

    assert out.shape == (1, prompt.shape[1] + max_new_tokens)
    torch.testing.assert_close(out[:, : prompt.shape[1]], prompt)
    assert torch.all(out[0, prompt.shape[1] :] == 3)


def test_spec_generate_stops_at_a_stop_token():
    torch.manual_seed(0)
    cfg = _draft_cfg()
    draft = Qwen3DFlash2DraftModel(cfg)
    target = _ConstantTarget(cfg, forced_token_id=9)

    prompt = torch.tensor([[1, 2, 3]])
    out = draft.spec_generate(target, prompt, 16, stop_token_ids=[9], temperature=0.0)

    assert out[0, -1].item() == 9
    assert out.shape[1] == prompt.shape[1] + 1
