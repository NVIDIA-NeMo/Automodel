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

"""Tests for MTP cross-sequence-boundary label masking.

When MTP is enabled on a packed sequence, the loss at depth k uses labels
shifted left by ``k+1``. If the source position (``t + k + 1``) falls into a
*different* sub-sequence than position ``t``, the prediction would cross a
packing boundary — which is nonsensical — and must be masked out with
``ignore_index`` so it does not contribute to the loss.

The cross-boundary logic lives at
``nemo_automodel/components/loss/mtp.py::calculate_mtp_loss`` lines 117-120
and is driven by either an explicit ``seq_idx`` or a derived one from
``cu_seqlens``.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from nemo_automodel.components.loss.mtp import calculate_mtp_loss

IGNORE = -100


def _hand_masked_labels(
    labels: torch.Tensor,
    seq_idx: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    """Compute the expected per-depth masked labels by hand.

    Mirrors the in-function logic so we can compare against it exactly:
      - left-shift labels by depth+1 (trailing positions filled with 0)
      - mask the trailing ``depth+1`` positions with ``ignore_index``
      - mask positions where ``seq_idx[t+depth+1] != seq_idx[t]``
    """
    shift = depth + 1
    if labels.dim() == 1:
        L = labels.shape[0]
        rolled = torch.cat([labels[shift:], torch.zeros(shift, dtype=labels.dtype)])
        out = rolled.clone()
        out[-shift:] = IGNORE
        rolled_seq = torch.cat([seq_idx[shift:], torch.zeros(shift, dtype=seq_idx.dtype)])
        out = torch.where(rolled_seq != seq_idx, torch.full_like(out, IGNORE), out)
        return out
    # 2D path
    B, S = labels.shape
    rolled = torch.cat([labels[:, shift:], torch.zeros(B, shift, dtype=labels.dtype)], dim=1)
    out = rolled.clone()
    out[:, -shift:] = IGNORE
    rolled_seq = torch.cat([seq_idx[:, shift:], torch.zeros(B, shift, dtype=seq_idx.dtype)], dim=1)
    out = torch.where(rolled_seq != seq_idx, torch.full_like(out, IGNORE), out)
    return out


class _CaptureLoss:
    """Mock loss callable that records the labels it received at each call."""

    def __init__(self):
        self.captured_labels: list[torch.Tensor] = []

    def __call__(self, **kwargs):
        self.captured_labels.append(kwargs["labels"].clone())
        return torch.zeros((), requires_grad=True)


def _run_capture(
    *,
    labels: torch.Tensor,
    seq_idx: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    depths: int = 2,
    hidden_dim: int = 4,
) -> list[torch.Tensor]:
    """Run calculate_mtp_loss with a capture loss; return masked labels per depth."""
    # mtp_per_depth_h: D x [B, S, H] (or [T, H]) — only shape matters because
    # calculate_loss is mocked.
    if labels.dim() == 1:
        h_shape = (labels.shape[0], hidden_dim)
    else:
        h_shape = (labels.shape[0], labels.shape[1], hidden_dim)
    mtp_per_depth_h = [torch.zeros(h_shape, dtype=torch.float32, requires_grad=True) for _ in range(depths)]
    cap = _CaptureLoss()
    # Patch calculate_loss in the mtp module's namespace.
    with mock.patch("nemo_automodel.components.loss.mtp.calculate_loss", side_effect=lambda loss_fn, **kw: cap(**kw)):
        calculate_mtp_loss(
            loss_fn=mock.MagicMock(),  # signature only matters because calculate_loss is patched
            mtp_per_depth_h=mtp_per_depth_h,
            labels=labels,
            model=mock.MagicMock(),
            scaling_factor=1.0,
            cu_seqlens=cu_seqlens,
            seq_idx=seq_idx,
            ignore_index=IGNORE,
        )
    return cap.captured_labels


def test_cross_boundary_masked_via_seq_idx_1d():
    """Two 4-token sub-seqs in an 8-token packed sample. With seq_idx supplied
    directly, depth-0 must mask position 3 (rolls into sub-seq 1); depth-1 must
    mask positions 2 and 3 (both roll into sub-seq 1).
    """
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    seq_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    captured = _run_capture(labels=labels, seq_idx=seq_idx, depths=2)
    assert len(captured) == 2

    # Depth 0 (shift=1): rolled labels = [2,3,4,5,6,7,8,0]; cross-seq at t=3
    # (rolled is from sub-seq 1 while t=3 is sub-seq 0). Trailing 1 position
    # is also masked. So positions {3, 7} are IGNORE; rest are real.
    d0 = captured[0]
    assert d0.tolist() == [2, 3, 4, IGNORE, 6, 7, 8, IGNORE]

    # Depth 1 (shift=2): rolled labels = [3,4,5,6,7,8,0,0]; cross-seq at t∈{2,3}
    # (sources at t=4,5 are sub-seq 1). Trailing 2 positions are also masked.
    # So positions {2, 3, 6, 7} are IGNORE.
    d1 = captured[1]
    assert d1.tolist() == [3, 4, IGNORE, IGNORE, 7, 8, IGNORE, IGNORE]


def test_cross_boundary_masked_via_cu_seqlens_1d():
    """Same scenario but seq_idx is derived from cu_seqlens via searchsorted."""
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

    captured = _run_capture(labels=labels, cu_seqlens=cu_seqlens, depths=2)
    assert len(captured) == 2

    # Derived seq_idx = searchsorted([4, 8], [0..7]) = [0,0,0,0,1,1,1,1].
    # Expected masks match the previous test.
    assert captured[0].tolist() == [2, 3, 4, IGNORE, 6, 7, 8, IGNORE]
    assert captured[1].tolist() == [3, 4, IGNORE, IGNORE, 7, 8, IGNORE, IGNORE]


def test_no_masking_when_seq_idx_is_constant():
    """If every token belongs to a single sub-sequence, cross-seq mask is a
    no-op and only the trailing-shift mask applies.
    """
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    seq_idx = torch.zeros(8, dtype=torch.long)  # all same sub-seq

    captured = _run_capture(labels=labels, seq_idx=seq_idx, depths=2)

    # Depth 0: only trailing 1 masked.
    assert captured[0].tolist() == [2, 3, 4, 5, 6, 7, 8, IGNORE]
    # Depth 1: only trailing 2 masked.
    assert captured[1].tolist() == [3, 4, 5, 6, 7, 8, IGNORE, IGNORE]


def test_three_subseqs_uneven_widths():
    """Three sub-seqs of widths 3, 2, 4 in a 9-token pack. Verifies that
    masking respects unequal boundaries at depth 0 and depth 2.
    """
    labels = torch.arange(1, 10, dtype=torch.long)  # [1..9]
    seq_idx = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)

    captured_seq = _run_capture(labels=labels, seq_idx=seq_idx, depths=3)
    captured_cu = _run_capture(labels=labels, cu_seqlens=cu_seqlens, depths=3)

    # Hand-mask reference (validates the in-function logic against an
    # independent re-implementation).
    expected = [_hand_masked_labels(labels, seq_idx, d).tolist() for d in range(3)]

    for d in range(3):
        assert captured_seq[d].tolist() == expected[d], f"seq_idx path mismatch at depth {d}"
        assert captured_cu[d].tolist() == expected[d], f"cu_seqlens path mismatch at depth {d}"


def test_cross_boundary_2d_batch():
    """2D ``[B, S]`` labels. seq_idx is broadcast across the batch by the
    function under test. Verifies the broadcasting + masking together.
    """
    labels = torch.tensor(
        [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]],
        dtype=torch.long,
    )
    seq_idx_1d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    captured = _run_capture(labels=labels, seq_idx=seq_idx_1d, depths=2)
    seq_idx_2d = seq_idx_1d.unsqueeze(0).expand(2, -1)
    expected = [_hand_masked_labels(labels, seq_idx_2d, d) for d in range(2)]
    for d in range(2):
        assert torch.equal(captured[d], expected[d]), f"depth {d} mismatch"


def test_seq_idx_shape_mismatch_raises_under_2d():
    """If a 2D seq_idx is supplied whose shape does NOT match labels, the
    function should raise — silent broadcasting would mask the wrong tokens
    under PP chunking.
    """
    labels = torch.zeros(2, 8, dtype=torch.long)
    bad_seq_idx = torch.zeros(3, 8, dtype=torch.long)  # batch dim doesn't match
    with pytest.raises(ValueError, match="seq_idx.shape"):
        _run_capture(labels=labels, seq_idx=bad_seq_idx, depths=1)


def test_depth_beyond_subseq_length_fully_masked():
    """A 2-token sub-seq at depth=2 has no in-bounds rolled label. The mask
    must be IGNORE everywhere for that sub-seq's positions.
    """
    labels = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    # Two 2-token sub-seqs.
    seq_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    captured = _run_capture(labels=labels, seq_idx=seq_idx, depths=3)

    # Depth 2 (shift=3): rolled = [4, 0, 0, 0]; trailing 3 → IGNORE.
    # Position 0 rolls to position 3 (sub-seq 1) ≠ sub-seq 0 → IGNORE.
    # So all 4 positions are IGNORE.
    assert captured[2].tolist() == [IGNORE, IGNORE, IGNORE, IGNORE]


def test_thd_flat_labels_squeeze_3d_hidden_states():
    """Regression: under THD packing the model unsqueezes ``mtp_per_depth_h``
    back to ``[1, T, H]`` (model.py post-MTP-forward), but the recipe pops 1D
    ``[T]`` labels from the THD-flattened batch. ``cut_cross_entropy`` asserts
    ``hidden_states.shape[:-1] == labels.shape``, so calculate_mtp_loss must
    squeeze the synthetic batch axis when labels are 1D.
    """
    T, H = 8, 4
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)  # (T,)
    # Mimic model.py:790: per-depth hidden states are unsqueezed to (1, T, H).
    mtp_per_depth_h = [torch.zeros((1, T, H), dtype=torch.float32, requires_grad=True) for _ in range(2)]

    captured_hidden: list[torch.Tensor] = []
    captured_labels: list[torch.Tensor] = []

    def _capture(loss_fn, **kw):
        captured_hidden.append(kw["hidden_states"])
        captured_labels.append(kw["labels"])
        return torch.zeros((), requires_grad=True)

    with mock.patch("nemo_automodel.components.loss.mtp.calculate_loss", side_effect=_capture):
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        # Pass a FusedLinearCrossEntropy instance so calculate_mtp_loss takes
        # the hidden_states branch (the bug only manifests there).
        calculate_mtp_loss(
            loss_fn=FusedLinearCrossEntropy(),
            mtp_per_depth_h=mtp_per_depth_h,
            labels=labels,
            model=mock.MagicMock(),
            scaling_factor=1.0,
            ignore_index=IGNORE,
        )

    assert len(captured_hidden) == 2
    for d, (h, lab) in enumerate(zip(captured_hidden, captured_labels)):
        # After the fix, the synthetic batch axis must be squeezed so
        # h.shape[:-1] == lab.shape (cce's invariant).
        assert h.shape == (T, H), f"depth {d}: expected hidden_states (T, H), got {tuple(h.shape)}"
        assert lab.shape == (T,), f"depth {d}: expected labels (T,), got {tuple(lab.shape)}"
        assert h.shape[:-1] == lab.shape


def test_2d_labels_and_3d_hidden_states_unchanged():
    """Sanity: when labels are already 2D ``[B, S]`` and hidden states are
    ``[B, S, H]`` (the BSHD path), the reconciliation must be a no-op."""
    B, S, H = 2, 8, 4
    labels = torch.arange(1, B * S + 1, dtype=torch.long).view(B, S)
    mtp_per_depth_h = [torch.zeros((B, S, H), dtype=torch.float32, requires_grad=True) for _ in range(2)]

    captured_hidden: list[torch.Tensor] = []

    def _capture(loss_fn, **kw):
        captured_hidden.append(kw["hidden_states"])
        return torch.zeros((), requires_grad=True)

    with mock.patch("nemo_automodel.components.loss.mtp.calculate_loss", side_effect=_capture):
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        calculate_mtp_loss(
            loss_fn=FusedLinearCrossEntropy(),
            mtp_per_depth_h=mtp_per_depth_h,
            labels=labels,
            model=mock.MagicMock(),
            scaling_factor=1.0,
            ignore_index=IGNORE,
        )

    for d, h in enumerate(captured_hidden):
        assert h.shape == (B, S, H), f"depth {d}: BSHD path should be unchanged, got {tuple(h.shape)}"
