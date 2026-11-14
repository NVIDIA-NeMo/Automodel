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

"""Tests that ``calculate_mtp_loss`` prepares the LM head once for every depth.

``calculate_mtp_loss`` dispatches each MTP depth through ``calculate_loss``.
On the linear CE path the head is prepared once with ``prepare_lm_weight`` and
threaded through the main loss and all depths via the ``lm_weight`` argument:
a dense head is converted once and shared, and a sharded DTensor head is
streamed in vocabulary chunks by each depth instead of being gathered (covered
across real ranks by ``test_linear_ce_streamed.py``). Before that threading,
each depth gathered its own copy, and ``1 + D`` copies retained for backward
OOMed large-vocab MoE (e.g. Nemotron-Ultra, 256k vocab; eos job 344476745).

These tests assert the threading is (a) numerically identical to resolving the
head per depth, (b) looks the head up exactly once, (c) skips the lookup when
the weight is supplied, and (d) preserves gradients.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import nemo_automodel.components.loss.linear_ce as _lce
import nemo_automodel.components.loss.mtp as _mtp
import nemo_automodel.components.loss.utils as _lutils
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.models.common.mtp import roll_tensor

H, V, B, S, D = 8, 32, 2, 6, 2
IGN, SF, NLT = -100, 0.1, 20


@pytest.fixture
def ref_cce(monkeypatch):
    """Run the FusedLinearCrossEntropy path on CPU with a reference torch CE.

    We are validating the LM-head threading, not the cut_cross_entropy
    kernel, so a dense reference matmul+CE (matching cut_cross_entropy's
    sum-reduction / shift=False contract) is the right oracle and keeps the
    test CPU-only.
    """

    def _ref(
        hidden,
        lm_weight,
        targets,
        ignore_index=-100,
        softcap=None,
        reduction="sum",
        shift=False,
        filter_eps=None,
        accum_e_fp32=False,
        impl="cce",
    ):
        logits = hidden.float() @ lm_weight.float().t()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
            ignore_index=ignore_index,
            reduction=reduction,
        )

    monkeypatch.setattr(_lce, "linear_cross_entropy", _ref, raising=False)
    monkeypatch.setattr(_lce, "HAVE_CUT_CROSS_ENTROPY", True, raising=False)


class _TinyModel(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.lm_head = nn.Linear(H, V, bias=False)
        if weight is not None:
            self.lm_head.weight = nn.Parameter(weight)
        self.training = True

    def get_output_embeddings(self):
        return self.lm_head


def _reference_mtp_loss(hs, labels, W):
    """Independent MTP loss: per-depth roll/mask + dense CE, summed and scaled."""
    cur = labels.clone()
    total = torch.zeros(())
    for k, h in enumerate(hs):
        cur = roll_tensor(cur, shifts=-1, dim=-1)
        masked = cur.clone()
        masked[..., -min(k + 1, labels.shape[-1]) :] = IGN
        logits = h.float() @ W.float().t()
        total = (
            total
            + F.cross_entropy(logits.reshape(-1, V), masked.reshape(-1).long(), ignore_index=IGN, reduction="sum") / NLT
        )
    return total * (SF / D)


def _inputs(requires_grad=False, seed=0):
    torch.manual_seed(seed)
    hs = [torch.randn(B, S, H, requires_grad=requires_grad) for _ in range(D)]
    labels = torch.randint(0, V, (B, S))
    return hs, labels


def _count_gathers():
    """Count weight retrievals at the shared preparation boundary."""
    n = {"c": 0}
    real = _lutils._get_lm_head_weight

    def spy(model):
        n["c"] += 1
        return real(model)

    return (
        n,
        mock.patch.object(_lutils, "_get_lm_head_weight", side_effect=spy),
    )


def test_mtp_loss_matches_reference(ref_cce):
    torch.manual_seed(1)
    m = _TinyModel()
    hs, labels = _inputs()
    got = calculate_mtp_loss(
        FusedLinearCrossEntropy(reduction="sum"),
        mtp_per_depth_h=[h.clone() for h in hs],
        labels=labels,
        model=m,
        scaling_factor=SF,
        num_label_tokens=NLT,
        ignore_index=IGN,
    )
    expected = _reference_mtp_loss(hs, labels, m.lm_head.weight.detach())
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_mtp_loss_optionally_returns_unscaled_per_depth_losses(ref_cce):
    torch.manual_seed(7)
    m = _TinyModel()
    hs, labels = _inputs(requires_grad=True)
    output = calculate_mtp_loss(
        FusedLinearCrossEntropy(reduction="sum"),
        mtp_per_depth_h=hs,
        labels=labels,
        model=m,
        scaling_factor=SF,
        num_label_tokens=NLT,
        ignore_index=IGN,
        return_per_depth=True,
    )
    assert isinstance(output, _mtp.MTPLossOutput)

    expected_per_depth = []
    cur_labels = labels
    for depth, hidden_states in enumerate(hs, start=1):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        masked = cur_labels.clone()
        masked[..., -depth:] = IGN
        logits = hidden_states.float() @ m.lm_head.weight.float().t()
        expected_per_depth.append(
            F.cross_entropy(logits.reshape(-1, V), masked.reshape(-1), ignore_index=IGN, reduction="sum") / NLT
        )

    assert len(output.per_depth_losses) == D
    for actual, expected in zip(output.per_depth_losses, expected_per_depth):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.loss, sum(expected_per_depth) * (SF / D), rtol=1e-5, atol=1e-6)

    output.loss.backward()
    assert all(loss.grad_fn is not None for loss in output.per_depth_losses)


def test_lm_head_gathered_once(ref_cce):
    torch.manual_seed(1)
    m = _TinyModel()
    hs, labels = _inputs()
    n, patch_weight = _count_gathers()
    with patch_weight:
        calculate_mtp_loss(
            FusedLinearCrossEntropy(reduction="sum"),
            mtp_per_depth_h=[h.clone() for h in hs],
            labels=labels,
            model=m,
            scaling_factor=SF,
            num_label_tokens=NLT,
            ignore_index=IGN,
        )
    assert n["c"] == 1, f"expected exactly 1 LM-head lookup, got {n['c']} (pre-fix == 1+D up the stack)"


def test_lm_weight_passthrough_skips_gather(ref_cce):
    torch.manual_seed(1)
    m = _TinyModel()
    hs, labels = _inputs()
    n, patch_weight = _count_gathers()
    with patch_weight:
        calculate_mtp_loss(
            FusedLinearCrossEntropy(reduction="sum"),
            mtp_per_depth_h=[h.clone() for h in hs],
            labels=labels,
            model=m,
            scaling_factor=SF,
            num_label_tokens=NLT,
            ignore_index=IGN,
            lm_weight=m.lm_head.weight.detach(),
        )
    assert n["c"] == 0, f"caller-supplied lm_weight must avoid any lookup, got {n['c']}"


def test_mtp_loss_grads_match_reference(ref_cce):
    hs1, labels = _inputs(requires_grad=True)
    m = _TinyModel()
    W2 = m.lm_head.weight.detach().clone().requires_grad_(True)
    hs2 = [h.detach().clone().requires_grad_(True) for h in hs1]
    calculate_mtp_loss(
        FusedLinearCrossEntropy(reduction="sum"),
        mtp_per_depth_h=hs1,
        labels=labels,
        model=m,
        scaling_factor=SF,
        num_label_tokens=NLT,
        ignore_index=IGN,
    ).backward()
    _reference_mtp_loss(hs2, labels, W2).backward()
    torch.testing.assert_close(m.lm_head.weight.grad, W2.grad, rtol=1e-4, atol=1e-6)
    for a, b in zip(hs1, hs2):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)


def test_pipeline_loss_fused_ce_routes_bare_tensor_as_hidden_states():
    """PP fused-CE contract: the last stage skips lm_head and emits a bare
    hidden-states tensor"""
    from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss

    m = _TinyModel()
    m.training = False  # a bare tensor carries no MTP tail regardless
    loss_mod = PipelineCausalLMLoss(FusedLinearCrossEntropy(), m)

    hidden = torch.randn(B, S, H)  # what the last PP stage now emits
    labels = torch.randint(0, V, (B, S))

    captured = {}

    def fake_calc(loss_fn, **kw):
        captured.update(kw)
        return torch.zeros((), requires_grad=True)

    with mock.patch.object(_mtp, "calculate_loss", side_effect=fake_calc):
        loss_mod(hidden, labels)

    assert captured["hidden_states"] is hidden
    assert captured["logits"] is None


def test_pipeline_loss_fused_ce_with_mtp_tuple_raises():
    """FusedLinearCrossEntropy cannot consume an MTP tuple output (it carries
    logits, not the hidden states the fused loss needs), so the last-stage loss
    must reject it explicitly instead of silently mishandling the tensors."""
    from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss

    loss_mod = PipelineCausalLMLoss(FusedLinearCrossEntropy(), _TinyModel())

    logits = torch.randn(B, S, V)
    mtp_h = torch.randn(B, S, H)  # per-depth hidden states appended by an MTP model
    labels = torch.randint(0, V, (B, S))

    with pytest.raises(ValueError, match="FusedLinearCrossEntropy is not supported with MTP"):
        loss_mod((logits, mtp_h), labels)


def test_non_fused_path_does_not_gather_lm_head():
    """Logit-based losses (non-Fused) must not look up the LM-head weight."""
    hs, labels = _inputs()

    def fake_calc(loss_fn, **kw):  # avoid invoking a real loss
        return torch.zeros((), requires_grad=True)

    class _LogitLoss(nn.Module):
        pass

    m = _TinyModel()
    with (
        mock.patch.object(_mtp, "calculate_loss", side_effect=fake_calc),
        mock.patch.object(
            _lutils, "_get_lm_head_weight", side_effect=AssertionError("non-fused path must not look up the LM head")
        ),
    ):
        calculate_mtp_loss(
            _LogitLoss(),
            mtp_per_depth_h=[h.clone() for h in hs],
            labels=labels,
            model=m,
            scaling_factor=SF,
            ignore_index=IGN,
        )
