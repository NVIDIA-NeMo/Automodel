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

"""EAGLE-4 draft: sliding-window draft layer, low-rank candidate head and the candidate-head trainer."""

import math

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle import draft_llama
from nemo_automodel.components.speculative.eagle.draft_llama import (
    EAGLE4_ARCHITECTURE,
    LlamaEagle3DraftModel,
    _draft_sliding_window,
    attach_eagle3_cp_attention,
)
from nemo_automodel.components.speculative.eagle.eagle4 import (
    CANDIDATE_HEAD_KEYS,
    Eagle4CandidateHeadConfig,
    Eagle4CandidateHeadTrainerModule,
    Eagle4Config,
)

VOCAB, HIDDEN, SEQ = 64, 32, 8


def _config(
    *,
    sliding_window: int | None = 3,
    rank: int | None = None,
    num_candidates: int = 8,
    norm_output: bool = True,
    attn_implementation: str = "eager",
) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = VOCAB
    config.target_hidden_size = HIDDEN
    config.attn_implementation = attn_implementation
    config.fc_norm = True
    config.norm_output = norm_output
    candidate_head = None if rank is None else Eagle4CandidateHeadConfig(rank=rank, num_candidates=num_candidates)
    eagle4 = Eagle4Config(sliding_window=sliding_window, candidate_head=candidate_head)
    config.update(eagle4.draft_config_updates(aux_layer_ids=[1, 2, 3]))
    return config


def _draft(**kwargs) -> LlamaEagle3DraftModel:
    torch.manual_seed(0)
    return LlamaEagle3DraftModel(_config(**kwargs)).to(torch.float32)


def _unroll(draft, aux, input_ids, *, steps=2, attention_mask=None, position_ids=None, seq_lens=None):
    """Run ``steps`` TTT steps; return the per-step hidden states, each of shape [batch, sequence, hidden]."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    hidden = draft.project_hidden_states(aux)
    cache: list[list[torch.Tensor]] = [[], []]
    outputs = []
    for _ in range(steps):
        hidden = draft(
            input_ids=input_ids,
            projected_hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_hidden=cache,
            seq_lens=seq_lens,
        )
        outputs.append(hidden)
    return outputs


def _inputs(batch=2, seq=SEQ):
    torch.manual_seed(1)
    return torch.randn(batch, seq, 3 * HIDDEN), torch.randint(0, VOCAB, (batch, seq))


# --- sliding window ---------------------------------------------------------


def test_sliding_window_is_read_only_for_eagle4_configs():
    config = _config(sliding_window=3)
    assert _draft_sliding_window(config) == 3

    config.architectures = ["LlamaEagle3DraftModel"]  # a Qwen-derived EAGLE-3 config keeps full attention
    assert _draft_sliding_window(config) is None

    full = _config(sliding_window=None)
    assert full.layer_types == ["full_attention"]
    assert _draft_sliding_window(full) is None

    no_layer_types = _config(sliding_window=5)
    no_layer_types.layer_types = None
    assert _draft_sliding_window(no_layer_types) == 5
    no_layer_types.use_sliding_window = False
    assert _draft_sliding_window(no_layer_types) is None


@pytest.mark.parametrize("window", [0, None, True])
def test_sliding_window_rejects_an_invalid_size(window):
    config = _config(sliding_window=3)
    config.sliding_window = window
    with pytest.raises(ValueError, match="sliding_window"):
        _draft_sliding_window(config)


def test_window_hides_context_keys_outside_the_window():
    """Changing position 0 must not reach positions >= W at any TTT depth, but does without a window."""
    aux, input_ids = _inputs()
    perturbed = aux.clone()
    perturbed[:, 0] += 5.0
    for window, reaches_far_positions in ((3, False), (None, True)):
        draft = _draft(sliding_window=window)
        with torch.no_grad():
            base = _unroll(draft, aux, input_ids)
            changed = _unroll(draft, perturbed, input_ids)
        for step_base, step_changed in zip(base, changed):
            assert not torch.allclose(step_base[:, :3], step_changed[:, :3])
            far_moved = not torch.allclose(step_base[:, 3:], step_changed[:, 3:], atol=1e-6)
            assert far_moved == reaches_far_positions


def test_window_covering_the_sequence_matches_full_attention():
    aux, input_ids = _inputs()
    # Depth i leaves W - i context keys, so W = SEQ + 2 still covers the sequence at depth 2.
    windowed, full = _draft(sliding_window=SEQ + 2), _draft(sliding_window=None)
    full.load_state_dict(windowed.state_dict())
    with torch.no_grad():
        for a, b in zip(_unroll(windowed, aux, input_ids, steps=3), _unroll(full, aux, input_ids, steps=3)):
            torch.testing.assert_close(a, b)


def test_window_slides_with_the_query_like_the_serving_kernel():
    """Every query sees the last W keys up to itself, its chain included, as the EAGLE-4 serving kernels do.

    With W = 4, uniform scores and only V_0[4] = 1: at depth 0 the row-7 query (position 7)
    sees keys 4..7 and returns 1/4; at depth 1 the row-7 query stands at position 8 and sees
    the context keys 5..7 plus its chain key, so key 4 is out of its window and it returns 0.
    """
    attn = _draft(sliding_window=4).model.layers[0].self_attn
    zeros = torch.zeros(1, attn.num_heads, SEQ, attn.head_dim)
    v0 = zeros.clone()
    v0[:, :, 4] = 1.0
    mask = draft_llama._build_causal_mask(torch.ones(1, SEQ, dtype=torch.long), torch.float32)
    root = attn._eager_attention_forward(zeros, [zeros], [v0], mask, 0, 1, SEQ)
    depth1 = attn._eager_attention_forward(zeros, [zeros, zeros], [v0, zeros], mask, 1, 1, SEQ)
    torch.testing.assert_close(root[0, 7], torch.full_like(root[0, 7], 0.25))
    torch.testing.assert_close(depth1[0, 7], torch.zeros_like(depth1[0, 7]))
    # The depth-1 query of row 6 stands at position 7 and still sees key 4 (keys 4..6 plus its chain).
    torch.testing.assert_close(depth1[0, 6], torch.full_like(depth1[0, 6], 0.25))


def test_window_smaller_than_the_unroll_is_rejected():
    aux, input_ids = _inputs()
    with pytest.raises(ValueError, match="at least ttt_steps"):
        _unroll(_draft(sliding_window=2), aux, input_ids, steps=3)


def test_window_on_padded_rows_stays_finite():
    """A padding query whose whole window is padding must not produce NaNs."""
    aux, input_ids = _inputs()
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 2:] = 0
    with torch.no_grad():
        outputs = _unroll(_draft(sliding_window=2), aux, input_ids, attention_mask=attention_mask)
    assert all(torch.isfinite(h).all() for h in outputs)


def test_packed_window_matches_unpacked_documents():
    """Packing two documents must equal running each alone, with the window applied per document."""
    draft = _draft(sliding_window=3)
    aux, input_ids = _inputs(batch=1, seq=SEQ)
    lengths = (5, 3)
    seq_lens = torch.tensor([list(lengths)])
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    with torch.no_grad():
        packed = _unroll(draft, aux, input_ids, position_ids=position_ids, seq_lens=seq_lens)
        start = 0
        for n in lengths:
            alone = _unroll(draft, aux[:, start : start + n], input_ids[:, start : start + n])
            for step_packed, step_alone in zip(packed, alone):
                torch.testing.assert_close(step_packed[:, start : start + n], step_alone, atol=1e-5, rtol=1e-5)
            start += n


def _reference_flash(q, k, v, *, softmax_scale, causal, window_size, return_attn_probs, cu=None):
    """Plain-torch stand-in for flash_attn_func.

    Args:
        q: Tensor of shape [batch, sequence, heads, head_dim].
        k: Tensor of shape [batch, sequence, heads, head_dim].
        v: Tensor of shape [batch, sequence, heads, head_dim].

    Returns:
        (out of shape [batch, sequence, heads, head_dim], lse of shape [batch, heads, sequence], None).
    """
    assert causal and return_attn_probs
    seq = q.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * softmax_scale
    rows, cols = torch.arange(seq).view(-1, 1), torch.arange(seq).view(1, -1)
    keep = cols <= rows
    if window_size[0] >= 0:
        keep &= cols >= rows - window_size[0]
    scores = scores.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
    return out, lse, None


def _reference_flash_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    """Plain-torch stand-in for flash_attn_varlen_func over [tokens, heads, head_dim] inputs."""
    outs, lses = [], []
    bounds = cu_seqlens_q.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        out, lse, _ = _reference_flash(q[None, start:end], k[None, start:end], v[None, start:end], **kwargs)
        outs.append(out[0])
        lses.append(lse[0])
    return torch.cat(outs), torch.cat(lses, dim=-1), None


@pytest.mark.parametrize("packed", [False, True])
def test_flash_attention_path_applies_the_window(monkeypatch, packed):
    """The FA2 path (driven through a reference kernel) must match the eager windowed path."""
    monkeypatch.setattr(draft_llama, "_HAS_FA", True)
    monkeypatch.setattr(draft_llama, "_flash_attn_func", _reference_flash)
    monkeypatch.setattr(draft_llama, "_flash_attn_varlen_func", _reference_flash_varlen)
    eager = _draft(sliding_window=3)
    flash = LlamaEagle3DraftModel(_config(sliding_window=3, attn_implementation="flash_attention_2")).to(torch.float32)
    flash.load_state_dict(eager.state_dict())
    aux, input_ids = _inputs(batch=1)
    kwargs = {}
    if packed:
        kwargs = {"seq_lens": torch.tensor([[5, 3]]), "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])}
    with torch.no_grad():
        for a, b in zip(_unroll(eager, aux, input_ids, **kwargs), _unroll(flash, aux, input_ids, **kwargs)):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not draft_llama._HAS_FA,
    reason="FA2 path requires a CUDA device and the 'flash-attn' package",
)
@pytest.mark.parametrize("packed", [False, True])
def test_real_flash_attention_window_matches_eager(packed):
    device, dtype = torch.device("cuda"), torch.bfloat16
    eager = _draft(sliding_window=5).to(device=device, dtype=dtype)
    flash = LlamaEagle3DraftModel(_config(sliding_window=5, attn_implementation="flash_attention_2"))
    flash = flash.to(device=device, dtype=dtype)
    flash.load_state_dict(eager.state_dict())
    torch.manual_seed(2)
    seq = 32
    aux = torch.randn(1, seq, 3 * HIDDEN, device=device, dtype=dtype)
    input_ids = torch.randint(0, VOCAB, (1, seq), device=device)
    kwargs = {}
    if packed:
        kwargs = {
            "seq_lens": torch.tensor([[20, 12]], device=device),
            "position_ids": torch.cat([torch.arange(20), torch.arange(12)]).unsqueeze(0).to(device),
        }
    with torch.no_grad():
        for a, b in zip(
            _unroll(eager, aux, input_ids, steps=3, **kwargs), _unroll(flash, aux, input_ids, steps=3, **kwargs)
        ):
            torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)


def test_context_parallel_rejects_the_sliding_window():
    with pytest.raises(NotImplementedError, match="sliding-window"):
        attach_eagle3_cp_attention(_draft(sliding_window=3), cp_group=object())


def test_parallel_drafting_rejects_the_sliding_window():
    config = _config(sliding_window=3)
    config.parallel_drafting = True
    config.mask_token_id = 0
    with pytest.raises(ValueError, match="parallel_drafting"):
        LlamaEagle3DraftModel(config)


# --- candidate head ---------------------------------------------------------


def test_candidate_head_registers_top_level_keys_only_when_configured():
    assert not CANDIDATE_HEAD_KEYS & _draft().state_dict().keys()
    draft = _draft(rank=4)
    state = draft.state_dict()
    assert CANDIDATE_HEAD_KEYS <= state.keys()
    assert state["efficient_lm_head.weight"].shape == (VOCAB, 4)
    assert state["efficient_lm_head_down_proj.weight"].shape == (4, HIDDEN)
    assert "d2t" not in state and "t2d" not in state


@pytest.mark.parametrize("norm_output", [True, False])
def test_candidate_logits_read_the_same_normed_state_as_lm_head(norm_output):
    """With a rank-``hidden`` head equal to lm_head, the candidate logits equal the full logits."""
    draft = _draft(rank=HIDDEN, norm_output=norm_output)
    with torch.no_grad():
        draft.efficient_lm_head_down_proj.weight.copy_(torch.eye(HIDDEN))
        draft.efficient_lm_head.weight.copy_(draft.lm_head.weight)
    hidden = torch.randn(2, SEQ, HIDDEN)
    torch.testing.assert_close(draft.compute_candidate_logits(hidden), draft.compute_logits(hidden))


@pytest.mark.parametrize(
    "head_config",
    [{"type": "weighted_low_rank", "rank": 4, "num_candidates": 8}, {"type": "low_rank", "rank": 0}, "low_rank"],
)
def test_candidate_head_rejects_unsupported_configs(head_config):
    config = _config(rank=4)
    config.efficient_lm_head_config = head_config
    with pytest.raises(ValueError, match="efficient_lm_head_config"):
        LlamaEagle3DraftModel(config)


def test_candidate_head_rank_must_stay_below_the_hidden_size():
    config = _config(rank=4)
    config.efficient_lm_head_config = {"type": "low_rank", "rank": HIDDEN + 1, "num_candidates": 8}
    with pytest.raises(ValueError, match="exceeds min"):
        LlamaEagle3DraftModel(config)


def test_svd_init_zeroes_directions_beyond_the_numerical_rank():
    """A head of rank 2 initialized at rank 4 keeps finite factors and still reproduces lm_head."""
    draft = _draft(rank=4)
    with torch.no_grad():
        draft.lm_head.weight.copy_(torch.randn(VOCAB, 2) @ torch.randn(2, HIDDEN))
    draft.init_efficient_lm_head_from_lm_head()
    assert torch.isfinite(draft.efficient_lm_head.weight).all()
    product = draft.efficient_lm_head.weight @ draft.efficient_lm_head_down_proj.weight
    torch.testing.assert_close(product, draft.lm_head.weight, atol=1e-4, rtol=1e-4)


def test_candidate_head_requires_the_full_vocabulary():
    config = _config(rank=4)
    config.draft_vocab_size = VOCAB // 2
    with pytest.raises(ValueError, match="full target vocabulary"):
        LlamaEagle3DraftModel(config)


def test_svd_init_gives_the_best_rank_r_approximation(monkeypatch):
    monkeypatch.setattr(draft_llama, "_SVD_CHUNK_ROWS", 10)  # exercise the chunked passes
    draft = _draft(rank=4)
    with torch.no_grad():
        draft.lm_head.weight.copy_(torch.randn(VOCAB, HIDDEN))
    draft.init_efficient_lm_head_from_lm_head()
    product = draft.efficient_lm_head.weight @ draft.efficient_lm_head_down_proj.weight
    u, s, vh = torch.linalg.svd(draft.lm_head.weight.double(), full_matrices=False)
    best = (u[:, :4] * s[:4]) @ vh[:4]
    torch.testing.assert_close(product.double(), best, atol=1e-4, rtol=1e-4)
    # The factors split the singular values evenly.
    torch.testing.assert_close(draft.efficient_lm_head.weight.double().norm(dim=0), s[:4].sqrt(), atol=1e-4, rtol=1e-4)


def test_candidate_head_methods_require_the_head():
    draft = _draft()
    with pytest.raises(RuntimeError, match="efficient_lm_head_config"):
        draft.compute_candidate_logits(torch.zeros(1, 1, HIDDEN))
    with pytest.raises(RuntimeError, match="efficient_lm_head_config"):
        draft.init_efficient_lm_head_from_lm_head()


# --- config -----------------------------------------------------------------


def test_draft_config_updates_describe_the_serving_layout():
    updates = Eagle4Config(sliding_window=4096, candidate_head=Eagle4CandidateHeadConfig()).draft_config_updates(
        aux_layer_ids=[3, 31, 59]
    )
    assert updates == {
        "architectures": [EAGLE4_ARCHITECTURE],
        "hidden_act": "silu",
        "num_hidden_layers": 1,
        "layer_types": ["sliding_attention"],
        "use_sliding_window": True,
        "sliding_window": 4096,
        "num_aux_hidden_states": 3,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 31, 59], "use_aux_hidden_state": True},
        "efficient_lm_head_config": {"type": "low_rank", "rank": 128, "num_candidates": 128},
    }
    stage1 = Eagle4Config().draft_config_updates(aux_layer_ids=[1, 2, 3])
    assert stage1["layer_types"] == ["full_attention"] and stage1["use_sliding_window"] is False
    assert "efficient_lm_head_config" not in stage1


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_eagle4_config_rejects_invalid_windows(window):
    with pytest.raises(ValueError, match="sliding_window"):
        Eagle4Config(sliding_window=window)


@pytest.mark.parametrize("field", ["rank", "num_candidates"])
@pytest.mark.parametrize("value", [0, -3, 2.0, False])
def test_candidate_head_config_rejects_non_positive_sizes(field, value):
    with pytest.raises(ValueError, match=field):
        Eagle4CandidateHeadConfig(**{field: value})


# --- candidate-head trainer -------------------------------------------------


def _trainer(draft, ttt_steps=2, num_candidates=8):
    return Eagle4CandidateHeadConfig(rank=4, num_candidates=num_candidates).build(draft, ttt_steps=ttt_steps)


def _batch(loss_mask=None):
    aux, input_ids = _inputs()
    if loss_mask is None:
        loss_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": loss_mask,
        "aux_hidden_states": aux,
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ttt_steps": 0}, "ttt_steps"),
        ({"ttt_steps": True}, "ttt_steps"),
        ({"num_candidates": VOCAB + 1}, "num_candidates"),
    ],
)
def test_candidate_head_trainer_rejects_bad_arguments(kwargs, match):
    args = {"ttt_steps": 2, "num_candidates": 8, **kwargs}
    with pytest.raises(ValueError, match=match):
        Eagle4CandidateHeadTrainerModule(_draft(rank=4), **args)


def test_candidate_head_trainer_requires_a_candidate_head():
    with pytest.raises(ValueError, match="efficient_lm_head_config"):
        Eagle4CandidateHeadTrainerModule(_draft(), ttt_steps=2, num_candidates=8)


def test_exact_low_rank_head_has_full_recall_and_zero_kl():
    """When lm_head itself has rank r, the SVD init reproduces it: recall 1 and a zero KL."""
    draft = _draft(rank=4)
    with torch.no_grad():
        draft.lm_head.weight.copy_(torch.randn(VOCAB, 4) @ torch.randn(4, HIDDEN))
    draft.init_efficient_lm_head_from_lm_head()
    batch = _batch()
    metrics = _trainer(draft, ttt_steps=1, num_candidates=1)(**batch)
    assert metrics.accuracy.item() == pytest.approx(1.0)
    assert metrics.loss.item() == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("num_candidates, recall", [(1, 0.0), (VOCAB, 1.0)])
def test_recall_counts_the_teacher_top1_inside_the_top_k(num_candidates, recall):
    """A low-rank head that inverts the full head ranks the teacher's top-1 last."""
    draft = _draft(rank=HIDDEN)
    with torch.no_grad():
        draft.efficient_lm_head_down_proj.weight.copy_(torch.eye(HIDDEN))
        draft.efficient_lm_head.weight.copy_(-draft.lm_head.weight)
    trainer = Eagle4CandidateHeadTrainerModule(draft, ttt_steps=2, num_candidates=num_candidates)
    assert trainer(**_batch()).accuracy.item() == pytest.approx(recall)


def test_tied_scores_count_against_the_teacher_top1():
    """A head that scores every token the same retrieves nothing unless K covers the whole vocabulary."""
    draft = _draft(rank=4)
    with torch.no_grad():
        draft.efficient_lm_head.weight.zero_()
    for num_candidates, recall in ((VOCAB - 1, 0.0), (VOCAB, 1.0)):
        trainer = Eagle4CandidateHeadTrainerModule(draft, ttt_steps=1, num_candidates=num_candidates)
        assert trainer(**_batch()).accuracy.item() == pytest.approx(recall)


def test_training_reduces_the_loss_and_only_updates_the_factors():
    draft = _draft(rank=4)
    draft.init_efficient_lm_head_from_lm_head()
    draft.requires_grad_(False)
    draft.efficient_lm_head.requires_grad_(True)
    draft.efficient_lm_head_down_proj.requires_grad_(True)
    frozen_before = {n: p.detach().clone() for n, p in draft.named_parameters() if not p.requires_grad}
    trainer = _trainer(draft)
    optimizer = torch.optim.Adam([p for p in trainer.parameters() if p.requires_grad], lr=1e-2)
    batch = _batch()
    first = trainer(**batch).loss.item()
    for _ in range(30):
        optimizer.zero_grad()
        trainer(**batch).loss.backward()
        optimizer.step()
    assert trainer(**batch).loss.item() < first
    for name, param in draft.named_parameters():
        if name in frozen_before:
            assert param.grad is None
            torch.testing.assert_close(param, frozen_before[name])


def test_supervision_follows_the_shifted_loss_mask_and_document_bounds():
    draft = _draft(rank=4)
    loss_mask = torch.zeros(2, SEQ, dtype=torch.long)
    loss_mask[:, 4:] = 1
    batch = _batch(loss_mask)
    # Step 0 supervises slots 4..7; the shift moves the mask to slots 3..6 for step 1.
    assert _trainer(draft)(**batch).valid_tokens.item() == 2 * (4 + 4)
    # Packing: depth 1 is supervised only where at least 2 tokens remain in the
    # document, which drops slot 4 of slots 3..6.
    doc_remaining = torch.tensor([[8, 7, 6, 5, 1, 3, 2, 1]] * 2)
    position_ids = torch.arange(SEQ).expand(2, -1)
    seq_lens = torch.tensor([[SEQ]] * 2)
    metrics = _trainer(draft)(**batch, doc_remaining=doc_remaining, position_ids=position_ids, seq_lens=seq_lens)
    assert metrics.valid_tokens.item() == 2 * (4 + 3)
    assert 0.0 <= metrics.accuracy.item() <= 1.0
    assert math.isfinite(metrics.loss.item())


def test_target_supervision_is_ignored():
    draft = _draft(rank=4)
    batch = _batch()
    plain = _trainer(draft)(**batch)
    with_target = _trainer(draft)(
        **batch,
        target_logits=torch.randn(2, SEQ, VOCAB),
        target_probs=torch.rand(2, SEQ, VOCAB),
        position_mask=torch.zeros(2, SEQ, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(plain.loss, with_target.loss)


# --- Gemma4 targets ---------------------------------------------------------


def test_gemma4_target_builds_a_windowed_llama_style_draft():
    """A Gemma4 target's EAGLE-4 draft keeps its window and the global RoPE, as a SwiGLU LlamaConfig."""
    from transformers import Gemma4TextConfig

    from nemo_automodel.components.speculative.eagle.registry import EAGLE3_DRAFT_REGISTRY

    target = Gemma4TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        hidden_activation="gelu_pytorch_tanh",
    )
    target.rope_parameters = {
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1e6},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
    }
    to_draft = EAGLE3_DRAFT_REGISTRY["Gemma4ForConditionalGeneration"].eagle4_draft_config
    config = to_draft(target)
    assert target.rope_parameters["full_attention"]["rope_theta"] == 1e6  # the target config is untouched
    assert config["rope_parameters"]["rope_theta"] == 1e6 and config["rope_parameters"]["rope_type"] == "default"
    assert "per_layer_config" not in config
    config.update(draft_vocab_size=VOCAB, target_hidden_size=32, fc_norm=True, norm_output=True)
    config.update(
        Eagle4Config(sliding_window=3, candidate_head=Eagle4CandidateHeadConfig(rank=4)).draft_config_updates(
            aux_layer_ids=[0, 1, 2]
        )
    )
    draft = LlamaEagle3DraftModel(LlamaConfig.from_dict(config))
    assert draft.config.layer_types == ["sliding_attention"]  # not forced back to full attention
    assert draft.model.layers[0].self_attn.sliding_window == 3
    assert draft.config.hidden_act == "silu"
    attn = draft.model.layers[0].self_attn
    # The global base (1e6), not the sliding one (1e4), sets the rotary frequencies.
    assert attn.rotary_emb.inv_freq[1].item() == pytest.approx(1e6 ** (-2 / attn.head_dim), rel=1e-5)
