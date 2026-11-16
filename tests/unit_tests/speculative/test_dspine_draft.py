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

"""Tests for the DSpine draft model: adjacent causal injection, refinement, and checkpoint layout."""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.attention.dflash_mask import create_dflash_sdpa_mask
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.draft_qwen3_dspine import (
    DSpineCodeBridge,
    Qwen3DSpineDraftModel,
)

VOCAB = 64
HIDDEN = 32
NUM_LAYERS = 3
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
CODE = 16
RANK = 8
TOP_K = 4


def dspine_dflash_config(num_layers=NUM_LAYERS, teacher_layers=(1, 2), **overrides):
    """The ``dflash_config`` block the DSpine recipe stamps, at test sizes."""
    cfg = {
        "mask_token_id": MASK_ID,
        "target_layer_ids": TARGET_LAYER_IDS,
        "block_size": BLOCK_SIZE,
        "draft_block_causal": True,
        "selected_prefix_code_bridge": {"code_size": CODE, "rank": RANK, "eps": 1e-6},
        "selected_prefix_feature_layers": list(range(1, num_layers + 1)),
        "selected_prefix_feature_writes": True,
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_terminal": False,
        "selected_prefix_feature_teacher_layers": list(teacher_layers),
        "selected_prefix_last_write_refine": {"kind": "exact_sender", "version": 1, "head_topk": TOP_K, "weight": 1.0},
    }
    cfg.update(overrides)
    return cfg


def draft_cfg(num_layers=NUM_LAYERS, attention_backend="sdpa", **dflash_overrides):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_theta=1000000,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = dspine_dflash_config(num_layers=num_layers, **dflash_overrides)
    cfg._attn_implementation = attention_backend
    return cfg


def build_draft(seed=0, **kwargs):
    torch.manual_seed(seed)
    draft = Qwen3DSpineDraftModel(draft_cfg(**kwargs))
    draft.initialize_transfer_space(torch.randn(VOCAB, HIDDEN))
    return draft


def randomize_writes(draft, scale=0.5):
    """Move the bridge off its identity start so the injection does something."""
    with torch.no_grad():
        bridge = draft.selected_prefix_bridge
        bridge.feature_write_weights.normal_(std=scale)
        bridge.predecessor_gate.weight.normal_(std=scale)
        bridge.gate.bias.normal_(std=scale)


def draft_inputs(bsz=2, seq_len=12, anchors=((0, 4, 7), (2, 5, 8)), seed=1):
    """Draft forward inputs for blocks at the given anchors, with the causal block mask."""
    gen = torch.Generator().manual_seed(seed)
    anchor_positions = torch.tensor(anchors)
    n_blocks = anchor_positions.shape[1]
    keep = torch.ones_like(anchor_positions, dtype=torch.bool)
    noise = torch.randn(bsz, n_blocks * BLOCK_SIZE, HIDDEN, generator=gen)
    target_hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen)
    block_pos = (anchor_positions.unsqueeze(-1) + torch.arange(BLOCK_SIZE)).view(bsz, -1)
    position_ids = torch.cat([torch.arange(seq_len).expand(bsz, -1), block_pos], dim=1)
    mask = create_dflash_sdpa_mask(
        anchor_positions, keep, seq_len, BLOCK_SIZE, torch.device("cpu"), dtype=noise.dtype, causal=True
    )
    block_ids = torch.randint(0, MASK_ID, (bsz, n_blocks, BLOCK_SIZE), generator=gen)
    return dict(
        position_ids=position_ids,
        attention_mask=mask,
        noise_embedding=noise,
        target_hidden=target_hidden,
        block_token_ids=block_ids,
    )


# ---------------------------------------------------------------------------
# Config and checkpoint layout
# ---------------------------------------------------------------------------


def test_config_is_stamped_with_the_serving_architecture_and_causality():
    cfg = draft_cfg()
    Qwen3DSpineDraftModel(cfg)
    assert cfg.architectures == ["DSpineDraftModel"]
    assert cfg.is_causal is True


# Shapes of the released tencent/Qwen3-8B-DSpine-b16 bridge (C=1024, R=512, H=4096,
# L=5), scaled to the test sizes; the key names must match exactly so SGLang's
# strict DSPINE loader accepts the trained checkpoint.
def test_state_dict_matches_the_released_dspine_layout():
    state = Qwen3DSpineDraftModel(draft_cfg()).state_dict()
    bridge = {k: (tuple(v.shape), v.dtype) for k, v in state.items() if k.startswith("selected_prefix_bridge.")}
    assert bridge == {
        "selected_prefix_bridge.codebook": ((VOCAB, CODE), torch.float32),
        "selected_prefix_bridge.early_read_weights": ((NUM_LAYERS - 1, CODE, HIDDEN), torch.float32),
        "selected_prefix_bridge.read_weight": ((CODE, HIDDEN), torch.float32),
        "selected_prefix_bridge.feature_write_weights": ((NUM_LAYERS, HIDDEN, RANK), torch.float32),
        "selected_prefix_bridge.prev.weight": ((RANK, CODE), torch.float32),
        "selected_prefix_bridge.gate.weight": ((RANK, CODE), torch.float32),
        "selected_prefix_bridge.gate.bias": ((RANK,), torch.float32),
        "selected_prefix_bridge.predecessor_gate.weight": ((RANK, CODE), torch.float32),
        "selected_prefix_bridge.out.weight": ((CODE, RANK), torch.float32),
        "selected_prefix_bridge.independent_write_version": ((), torch.int64),
        "selected_prefix_bridge.terminal_disabled_version": ((), torch.int64),
    }
    for key in (
        "draft_block_causal_version",
        "last_write_refine_version",
        "selected_prefix_bridge.independent_write_version",
        "selected_prefix_bridge.terminal_disabled_version",
    ):
        assert state[key].item() == 1
    assert not state["selected_prefix_bridge.out.weight"].any()
    # Everything else is the plain DFlash layout (no embedding / head tensors).
    dflash_keys = set(Qwen3DFlashDraftModel(draft_cfg()).state_dict())
    other = {k for k in state if not k.startswith("selected_prefix_bridge.")}
    assert other == dflash_keys | {"draft_block_causal_version", "last_write_refine_version"}


def test_bf16_cast_keeps_version_scalars_int64_and_safetensors_round_trips(tmp_path):
    draft = build_draft().to(torch.bfloat16)
    state = {k: v.contiguous() for k, v in draft.state_dict().items()}
    save_file(state, str(tmp_path / "model.safetensors"))
    loaded = load_file(str(tmp_path / "model.safetensors"))
    assert loaded["selected_prefix_bridge.codebook"].dtype == torch.bfloat16
    assert loaded["selected_prefix_bridge.independent_write_version"].dtype == torch.int64
    assert loaded["last_write_refine_version"].item() == 1
    restored = Qwen3DSpineDraftModel(draft_cfg()).to(torch.bfloat16)
    restored.load_state_dict(loaded)
    for key, value in draft.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"draft_block_causal": False}, "draft_block_causal"),
        ({"selected_prefix_feature_writes": False}, "feature_writes"),
        ({"selected_prefix_feature_write_mode": "shared"}, "independent"),
        ({"selected_prefix_terminal": True}, "terminal"),
        ({"selected_prefix_feature_layers": [1, 2]}, "every draft layer"),
        ({"selected_prefix_feature_teacher_layers": [0]}, "teacher_layers"),
        ({"selected_prefix_last_write_refine": {"kind": "other", "head_topk": 4}}, "exact_sender"),
        ({"selected_prefix_last_write_refine": {"kind": "exact_sender", "head_topk": 6}}, "power of two"),
        ({"selected_prefix_last_write_refine": {"kind": "exact_sender", "head_topk": 128}}, "power of two"),
        ({"selected_prefix_code_bridge": {"code_size": HIDDEN + 1, "rank": RANK}}, "code_size"),
        ({"output_multiplier": 2.0}, "plain target head"),
        ({"projector_type": "domino", "emb_dim": 8, "gru_hidden_dim": 8}, "Domino"),
    ],
)
def test_rejects_configs_the_serving_path_cannot_load(overrides, match):
    with pytest.raises(ValueError, match=match):
        Qwen3DSpineDraftModel(draft_cfg(**overrides))


def test_rejects_an_explicitly_non_causal_config():
    cfg = draft_cfg()
    cfg.is_causal = False
    with pytest.raises(ValueError, match="is_causal"):
        Qwen3DSpineDraftModel(cfg)


def test_rejects_a_single_layer_draft():
    with pytest.raises(ValueError, match="at least 2 draft layers"):
        Qwen3DSpineDraftModel(draft_cfg(num_layers=1, teacher_layers=(1,)))


# ---------------------------------------------------------------------------
# Transfer space
# ---------------------------------------------------------------------------


def test_transfer_space_codes_are_unit_norm_and_readouts_orthonormal():
    torch.manual_seed(0)
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=NUM_LAYERS, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    head = torch.randn(VOCAB, HIDDEN)
    bridge.initialize_transfer_space(head, chunk_size=7)
    torch.testing.assert_close(bridge.codebook.norm(dim=-1), torch.ones(VOCAB))
    torch.testing.assert_close(bridge.read_weight @ bridge.read_weight.T, torch.eye(CODE), atol=1e-5, rtol=0)
    for early in bridge.early_read_weights:
        torch.testing.assert_close(early, bridge.read_weight)
    # The codes are the whitened principal components of the normalized, centred
    # head rows: before their final normalization they have identity covariance.
    rows = torch.nn.functional.normalize(head, dim=-1)
    rows = rows - rows.mean(dim=0)
    projected = rows @ bridge.read_weight.T
    whitened = projected / projected.pow(2).sum(dim=0).sqrt()
    torch.testing.assert_close(whitened.T @ whitened, torch.eye(CODE), atol=1e-4, rtol=0)
    expected = torch.nn.functional.normalize(whitened, dim=-1)
    # Eigenvector signs are arbitrary, so compare codes up to a per-column sign.
    signs = torch.sign((expected * bridge.codebook).sum(dim=0))
    torch.testing.assert_close(bridge.codebook, expected * signs, atol=1e-4, rtol=0)


def test_transfer_space_is_independent_of_the_chunk_size():
    head = torch.randn(VOCAB, HIDDEN)
    a = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    b = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    a.initialize_transfer_space(head, chunk_size=5)
    b.initialize_transfer_space(head, chunk_size=VOCAB)
    torch.testing.assert_close(a.codebook, b.codebook, atol=1e-5, rtol=0)


def test_transfer_space_readiness_tracks_initialization_and_checkpoint_loads():
    fresh = Qwen3DSpineDraftModel(draft_cfg())
    assert not fresh.transfer_space_ready
    trained = build_draft()
    assert trained.transfer_space_ready
    # A restore that carries the codebook (e.g. a resumed checkpoint) counts too.
    fresh.load_state_dict(trained.state_dict())
    assert fresh.transfer_space_ready
    torch.testing.assert_close(fresh.selected_prefix_bridge.codebook, trained.selected_prefix_bridge.codebook)
    partial = Qwen3DSpineDraftModel(draft_cfg())
    state = {k: v for k, v in trained.state_dict().items() if k != "selected_prefix_bridge.codebook"}
    partial.load_state_dict(state, strict=False)
    assert not partial.transfer_space_ready


def _broadcast_transfer_space_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        draft = build_draft()
        bridge = draft.selected_prefix_bridge
        if rank == 1:
            # Simulate a device whose eigendecomposition picked the opposite signs.
            with torch.no_grad():
                for tensor in (bridge.codebook, bridge.read_weight, bridge.early_read_weights):
                    tensor.neg_()
        draft.broadcast_transfer_space(src=0)
        torch.save(
            {
                name: t.detach().clone()
                for name, t in (
                    ("codebook", bridge.codebook),
                    ("read", bridge.read_weight),
                    ("early", bridge.early_read_weights),
                )
            },
            f"{out_dir}/rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_broadcast_transfer_space_aligns_every_rank_with_the_source(tmp_path):
    mp.spawn(
        _broadcast_transfer_space_worker,
        args=(2, str(tmp_path / "process-group"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0, rank1 = torch.load(tmp_path / "rank0.pt"), torch.load(tmp_path / "rank1.pt")
    for name in rank0:
        torch.testing.assert_close(rank1[name], rank0[name], rtol=0, atol=0)
    torch.testing.assert_close(rank0["codebook"], build_draft().selected_prefix_bridge.codebook, rtol=0, atol=0)


def test_transfer_space_rejects_a_mismatched_head():
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    with pytest.raises(ValueError, match="full \\[vocab, hidden\\]"):
        bridge.initialize_transfer_space(torch.randn(VOCAB - 1, HIDDEN))


def test_transfer_space_rejects_a_rank_deficient_head():
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    head = torch.randn(VOCAB, 4) @ torch.randn(4, HIDDEN)  # only 4 principal directions
    with pytest.raises(ValueError, match="positive principal directions"):
        bridge.initialize_transfer_space(head)


# ---------------------------------------------------------------------------
# Adjacent causal injection
# ---------------------------------------------------------------------------


def test_untrained_dspine_matches_a_causal_dflash_draft():
    """Zero writes make every injection the identity: a fresh DSpine is a causal DFlash."""
    dspine = build_draft()
    dflash = Qwen3DFlashDraftModel(draft_cfg())
    missing, unexpected = dflash.load_state_dict(dspine.state_dict(), strict=False)
    assert not missing
    assert all(k.startswith("selected_prefix_bridge.") or k.endswith("_version") for k in unexpected)
    inputs = draft_inputs()
    out = dspine(**inputs)
    expected = dflash(
        position_ids=inputs["position_ids"],
        attention_mask=inputs["attention_mask"],
        noise_embedding=inputs["noise_embedding"],
        target_hidden=inputs["target_hidden"],
    )
    torch.testing.assert_close(out.hidden_states, expected)


def test_bridge_writes_only_from_the_immediate_predecessor():
    torch.manual_seed(0)
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    bridge.initialize_transfer_space(torch.randn(VOCAB, HIDDEN))
    with torch.no_grad():
        bridge.feature_write_weights.normal_()
    hidden = torch.randn(1, 2, BLOCK_SIZE, HIDDEN)
    ids = torch.randint(0, VOCAB, (1, 2, BLOCK_SIZE))
    base = bridge(hidden, 0, ids)
    torch.testing.assert_close(base[..., 0, :], hidden[..., 0, :])  # the anchor row is never written
    perturbed = hidden.clone()
    perturbed[0, 0, 1] += torch.randn(HIDDEN)
    changed = (bridge(perturbed, 0, ids) - base).abs().amax(dim=-1)[0]
    # Row 1 changes (it is the perturbed row), row 2 receives row 1's message;
    # row 0, row 3, and the other block are untouched.
    assert changed[0, 1] > 0 and changed[0, 2] > 0
    assert changed[0, 0] == 0 and changed[0, 3] == 0 and (changed[1] == 0).all()


def test_first_position_receives_the_anchor_tokens_code():
    torch.manual_seed(0)
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    bridge.initialize_transfer_space(torch.randn(VOCAB, HIDDEN))
    with torch.no_grad():
        bridge.feature_write_weights.normal_()
    hidden = torch.randn(1, 1, BLOCK_SIZE, HIDDEN)
    ids = torch.tensor([[[3, 9, 9, 9]]])
    other_anchor = ids.clone()
    other_anchor[..., 0] = 4
    a, b = bridge(hidden, 1, ids), bridge(hidden, 1, other_anchor)
    assert (a[..., 1, :] != b[..., 1, :]).any()
    torch.testing.assert_close(a[..., 2:, :], b[..., 2:, :])
    # The anchor hidden itself does not feed position 1: its code is the token's.
    moved = hidden.clone()
    moved[..., 0, :] += 1.0
    torch.testing.assert_close(bridge(moved, 1, ids)[..., 1:, :], a[..., 1:, :])


def test_write_is_scaled_by_the_receivers_rms():
    torch.manual_seed(0)
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    with torch.no_grad():
        bridge.feature_write_weights.normal_()
    hidden = torch.randn(5, HIDDEN)
    receiver, predecessor = torch.randn(5, CODE), torch.randn(5, CODE)
    delta = bridge.write(hidden, receiver, predecessor, 0) - hidden
    delta_scaled = bridge.write(3.0 * hidden, receiver, predecessor, 0) - 3.0 * hidden
    torch.testing.assert_close(delta_scaled, 3.0 * delta, rtol=1e-5, atol=1e-5)


def test_feature_is_scale_invariant_with_norm_sqrt_code():
    bridge = DSpineCodeBridge(hidden_size=HIDDEN, num_layers=2, vocab_size=VOCAB, code_size=CODE, rank=RANK)
    hidden = torch.randn(3, HIDDEN)
    feature = bridge.feature(hidden, 1)
    torch.testing.assert_close(feature.norm(dim=-1), torch.full((3,), math.sqrt(CODE)))
    torch.testing.assert_close(bridge.feature(7.0 * hidden, 1), feature)


def test_in_block_attention_and_injection_are_causal():
    draft = build_draft()
    randomize_writes(draft)
    inputs = draft_inputs()
    base = draft(**inputs).hidden_states.view(2, 3, BLOCK_SIZE, HIDDEN)
    later = dict(inputs)
    later["noise_embedding"] = inputs["noise_embedding"].clone().view(2, 3, BLOCK_SIZE, HIDDEN)
    later["noise_embedding"][:, :, 2] += 1.0
    later["noise_embedding"] = later["noise_embedding"].view(2, 3 * BLOCK_SIZE, HIDDEN)
    changed = (draft(**later).hidden_states.view(2, 3, BLOCK_SIZE, HIDDEN) - base).abs().amax(dim=-1)
    assert (changed[:, :, :2] == 0).all()
    assert (changed[:, :, 2:] > 0).all()


def test_teacher_forcing_replaces_predecessors_only_in_its_layers_and_blocks():
    draft = build_draft()
    randomize_writes(draft)
    inputs = draft_inputs()
    base = draft(**inputs)
    replace = torch.tensor([[True, False, False], [False, False, False]])
    forced = draft(**inputs, replace_mask=replace)
    diff = (forced.hidden_states - base.hidden_states).abs().amax(dim=-1).view(2, 3, BLOCK_SIZE)
    assert (diff[0, 0, 2:] > 0).all()
    assert (diff[0, 1:] == 0).all() and (diff[1] == 0).all()
    # With no teacher-forced layers the mask is inert.
    plain = build_draft(teacher_layers=())
    randomize_writes(plain)
    torch.testing.assert_close(plain(**inputs, replace_mask=replace).hidden_states, plain(**inputs).hidden_states)


def test_refine_reproduces_the_forward_when_the_last_layer_is_teacher_forced():
    """``refine`` is the last write redone with exact codes; teacher-forcing every
    layer, including the last, makes the forward's last write exactly that."""
    draft = build_draft(teacher_layers=range(1, NUM_LAYERS + 1))
    randomize_writes(draft)
    inputs = draft_inputs()
    replace = torch.ones(2, 3, dtype=torch.bool)
    out = draft(**inputs, replace_mask=replace)
    pre = out.pre_injection.view(2, 3, BLOCK_SIZE, HIDDEN)[:, :, 1:]
    refined = draft.refine(pre, inputs["block_token_ids"][:, :, :-1])
    torch.testing.assert_close(refined, out.hidden_states.view(2, 3, BLOCK_SIZE, HIDDEN)[:, :, 1:])


def test_forward_output_shapes_and_finiteness():
    draft = build_draft()
    out = draft(**draft_inputs())
    assert out.hidden_states.shape == (2, 3 * BLOCK_SIZE, HIDDEN)
    assert out.pre_injection.shape == (2, 3 * BLOCK_SIZE, HIDDEN)
    assert out.layer_features.shape == (NUM_LAYERS, 2, 3 * BLOCK_SIZE, CODE)
    assert torch.isfinite(out.hidden_states).all()


def test_forward_rejects_missing_mask_cache_and_bad_blocks():
    draft = build_draft()
    inputs = draft_inputs()
    with pytest.raises(ValueError, match="explicit causal block attention mask"):
        draft(**{**inputs, "attention_mask": None})
    with pytest.raises(ValueError, match="no draft KV cache"):
        draft(**inputs, use_cache=True)
    with pytest.raises(ValueError, match="block_size"):
        draft(**{**inputs, "block_token_ids": inputs["block_token_ids"][..., :-1]})


def test_spec_generate_points_to_sglang():
    with pytest.raises(NotImplementedError, match="DSPINE"):
        build_draft().spec_generate(
            torch.nn.Identity(),
            torch.zeros(1, 4, dtype=torch.long),
            max_new_tokens=4,
            stop_token_ids=None,
            temperature=0.0,
        )
