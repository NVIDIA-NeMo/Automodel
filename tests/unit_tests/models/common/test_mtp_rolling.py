# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Pins MTPModule's cumulative left-rolling of input_ids and position_ids.

At depth ``k`` the MTP module embeds the token originally at position
``t + k + 1`` for slot ``t``. This is implemented by cumulatively rolling
``cur_input_ids`` (and ``cur_position_ids``) left by one at each depth in
``MTPModule.forward`` (see ``components/models/common/mtp/mtp.py``).

These tests intercept what reaches each sublayer and verify the rolled
inputs match a hand-rolled reference. The label-rolling on the loss side is
covered separately by ``tests/unit_tests/loss/test_mtp_cross_boundary.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.mtp.mtp import MTPConfig, MTPModule, roll_tensor


class _RecordingSublayer(nn.Module):
    """Sublayer that records its kwargs and returns hidden_states unchanged."""

    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # Detach + clone so a later in-place op can't mutate what we recorded.
        rec = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                rec[k] = v.detach().clone()
            else:
                rec[k] = v
        self.calls.append(rec)
        return hidden_states


def _build_module(num_depths: int = 3, pattern_length: int = 1) -> MTPModule:
    """Build an MTPModule with recording sublayers (no real attention/MoE)."""
    cfg = MTPConfig(
        num_layers=num_depths,
        layer_pattern="A" * pattern_length,
        loss_scaling_factor=1.0,
        use_repeated_layer=False,
    )
    block_types = ["recording"] * pattern_length

    def factory(global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        return _RecordingSublayer()

    return MTPModule(cfg, block_types, factory)


def _embed_fn_identity(input_ids: torch.LongTensor) -> torch.Tensor:
    """Embedding stub: cast int IDs to float, keep the seq dim intact.

    The MTP module passes the embedded result as ``embed_input`` to sublayer 0
    of each depth; we recover the un-embedded IDs by casting back to long.
    """
    return input_ids.to(torch.float32).unsqueeze(-1)  # shape [..., S, 1]


def _extract_embed_ids(rec_embed: torch.Tensor) -> list[int]:
    """Inverse of ``_embed_fn_identity``: pull the original IDs out."""
    return rec_embed.squeeze(-1).to(torch.long).tolist()


def test_cumulative_input_ids_rolling_1d():
    """Depth k sees input_ids rolled left by ``k+1`` (cumulative)."""
    mtp = _build_module(num_depths=3, pattern_length=1)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    hidden = torch.zeros(input_ids.shape[0], 4)

    mtp.forward(hidden, input_ids=input_ids, embed_fn=_embed_fn_identity)

    # 3 sublayers (1 per depth), each recorded one call.
    sublayer_calls = [layer.calls[0] for layer in mtp.layers]
    assert len(sublayer_calls) == 3

    for depth in range(3):
        expected_ids = roll_tensor(input_ids, shifts=-(depth + 1), dim=-1).tolist()
        got_ids = _extract_embed_ids(sublayer_calls[depth]["embed_input"])
        assert got_ids == expected_ids, (
            f"depth {depth}: expected ids rolled by -{depth + 1}, got {got_ids} (expected {expected_ids})"
        )


def test_cumulative_position_ids_rolling_1d():
    """Depth k sees position_ids rolled left by ``k+1`` (cumulative)."""
    mtp = _build_module(num_depths=3, pattern_length=1)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    position_ids = torch.arange(8, dtype=torch.long)
    hidden = torch.zeros(input_ids.shape[0], 4)

    mtp.forward(hidden, input_ids=input_ids, embed_fn=_embed_fn_identity, position_ids=position_ids)

    for depth in range(3):
        expected_pos = roll_tensor(position_ids, shifts=-(depth + 1), dim=-1).tolist()
        got_pos = mtp.layers[depth].calls[0]["position_ids"].tolist()
        assert got_pos == expected_pos, (
            f"depth {depth}: expected position_ids rolled by -{depth + 1}, got {got_pos} (expected {expected_pos})"
        )


def test_cumulative_rolling_2d_batch():
    """Per-row cumulative rolling in 2D ``[B, S]`` mode."""
    mtp = _build_module(num_depths=2, pattern_length=1)
    input_ids = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]],
        dtype=torch.long,
    )
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1).contiguous()
    hidden = torch.zeros(2, 8, 4)

    mtp.forward(hidden, input_ids=input_ids, embed_fn=_embed_fn_identity, position_ids=position_ids)

    for depth in range(2):
        expected_ids = roll_tensor(input_ids, shifts=-(depth + 1), dim=-1)
        got_ids = mtp.layers[depth].calls[0]["embed_input"].squeeze(-1).to(torch.long)
        assert torch.equal(got_ids, expected_ids), f"depth {depth}: input_ids rolling mismatch in 2D batch"

        expected_pos = roll_tensor(position_ids, shifts=-(depth + 1), dim=-1)
        got_pos = mtp.layers[depth].calls[0]["position_ids"]
        assert torch.equal(got_pos, expected_pos), f"depth {depth}: position_ids rolling mismatch in 2D batch"


def test_multi_sublayer_per_depth_sees_same_rolled_inputs():
    """When pattern_length > 1, all sublayers of a single depth see the
    rolled inputs from THAT depth (no further intra-depth rolling).
    """
    mtp = _build_module(num_depths=2, pattern_length=2)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    position_ids = torch.arange(8, dtype=torch.long)
    hidden = torch.zeros(input_ids.shape[0], 4)

    mtp.forward(hidden, input_ids=input_ids, embed_fn=_embed_fn_identity, position_ids=position_ids)

    # 4 sublayers (2 depths × 2 sublayers/depth).
    assert len(mtp.layers) == 4
    for depth in range(2):
        for sub_in_depth in range(2):
            flat = depth * 2 + sub_in_depth
            got_pos = mtp.layers[flat].calls[0]["position_ids"].tolist()
            expected_pos = roll_tensor(position_ids, shifts=-(depth + 1), dim=-1).tolist()
            assert got_pos == expected_pos, f"depth {depth} sublayer {sub_in_depth}: position_ids mismatch"
    # Only sublayer 0 of each depth gets embed_input.
    assert "embed_input" in mtp.layers[0].calls[0]
    assert "embed_input" not in mtp.layers[1].calls[0]
    assert "embed_input" in mtp.layers[2].calls[0]
    assert "embed_input" not in mtp.layers[3].calls[0]


def test_precomputed_embed_inputs_path_skips_token_rolling():
    """When ``embed_inputs`` is supplied, the per-depth token rolling is
    skipped — the caller has already prepared the rolled embeddings.
    Position_ids rolling still happens (it's independent of embed source).
    """
    mtp = _build_module(num_depths=2, pattern_length=1)
    # Two pre-computed embeddings (one per depth) marked with distinct values
    # so we can tell them apart in the captured embed_input.
    emb0 = torch.full((8, 4), 11.0)
    emb1 = torch.full((8, 4), 22.0)
    position_ids = torch.arange(8, dtype=torch.long)
    hidden = torch.zeros(8, 4)

    mtp.forward(hidden, embed_inputs=(emb0, emb1), position_ids=position_ids)

    # Depth 0 sees emb0 verbatim; depth 1 sees emb1 verbatim.
    assert torch.equal(mtp.layers[0].calls[0]["embed_input"], emb0)
    assert torch.equal(mtp.layers[1].calls[0]["embed_input"], emb1)

    # Position_ids still roll cumulatively even on the embed_inputs path.
    for depth in range(2):
        expected_pos = roll_tensor(position_ids, shifts=-(depth + 1), dim=-1).tolist()
        got_pos = mtp.layers[depth].calls[0]["position_ids"].tolist()
        assert got_pos == expected_pos, f"depth {depth}: position_ids rolling expected on embed_inputs path"


def test_trailing_positions_zero_after_roll():
    """The roll is left-shift with trailing zeros (not wrap-around). At depth
    k, the last ``k+1`` slots of the rolled tensor are zero — these are the
    positions for which there is no valid future token to predict.
    """
    mtp = _build_module(num_depths=3, pattern_length=1)
    input_ids = torch.arange(1, 9, dtype=torch.long)  # [1,2,3,4,5,6,7,8]
    hidden = torch.zeros(8, 4)

    mtp.forward(hidden, input_ids=input_ids, embed_fn=_embed_fn_identity)

    for depth in range(3):
        ids = _extract_embed_ids(mtp.layers[depth].calls[0]["embed_input"])
        n_trailing = depth + 1
        assert ids[-n_trailing:] == [0] * n_trailing, (
            f"depth {depth}: expected {n_trailing} trailing zeros, got tail {ids[-n_trailing:]}"
        )
        # The remaining prefix should match the original IDs shifted by n_trailing.
        assert ids[:-n_trailing] == input_ids[n_trailing:].tolist()
