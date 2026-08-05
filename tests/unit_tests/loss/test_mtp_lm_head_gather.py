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

"""Tests for the MTP-loss LM-head gather-once optimization.

``calculate_mtp_loss`` dispatches each MTP depth through ``calculate_loss``.
On the ``FusedLinearCrossEntropy`` path, ``calculate_loss`` materializes the LM
head via ``DTensor.full_tensor()`` (an all-gather) because cut_cross_entropy
needs the dense ``[vocab, hidden]`` weight. Re-gathering it for the main loss
*and* every MTP depth left ``1 + D`` gathered copies retained for backward,
which accumulate on-device and OOM large-vocab MoE (e.g. Nemotron-Ultra,
256k vocab; eos job 344476745). The fix gathers the head ONCE and threads it
through the main loss and all depths via the new ``lm_weight`` argument.

These tests assert the change is (a) numerically identical to gathering per
depth, (b) gathers the head exactly once, (c) is a no-op gather when the weight
is supplied, (d) preserves gradients, and (e) still works for the sharded
DTensor LM head where the OOM occurred.
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

    We are validating the LM-head gather threading, not the cut_cross_entropy
    kernel, so a dense reference matmul+CE (matching cut_cross_entropy's
    sum-reduction / shift=False contract) is the right oracle and keeps the
    test CPU-only.
    """

    def _ref(
        hidden, lm_weight, targets, ignore_index=-100, softcap=None, reduction="sum", shift=False, filter_eps=None
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
    """Patch _get_lm_head_weight in both namespaces with a counting wrapper."""
    n = {"c": 0}
    real = _lutils._get_lm_head_weight

    def spy(model):
        n["c"] += 1
        return real(model)

    return (
        n,
        mock.patch.object(_mtp, "_get_lm_head_weight", side_effect=spy),
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


def test_lm_head_gathered_once(ref_cce):
    torch.manual_seed(1)
    m = _TinyModel()
    hs, labels = _inputs()
    n, p1, p2 = _count_gathers()
    with p1, p2:
        calculate_mtp_loss(
            FusedLinearCrossEntropy(reduction="sum"),
            mtp_per_depth_h=[h.clone() for h in hs],
            labels=labels,
            model=m,
            scaling_factor=SF,
            num_label_tokens=NLT,
            ignore_index=IGN,
        )
    assert n["c"] == 1, f"expected exactly 1 LM-head gather, got {n['c']} (pre-fix == 1+D up the stack)"


def test_lm_weight_passthrough_skips_gather(ref_cce):
    torch.manual_seed(1)
    m = _TinyModel()
    hs, labels = _inputs()
    n, p1, p2 = _count_gathers()
    with p1, p2:
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
    assert n["c"] == 0, f"caller-supplied lm_weight must avoid any gather, got {n['c']}"


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
    """Logit-based losses (non-Fused) must not trigger the LM-head weight gather."""
    hs, labels = _inputs()

    def fake_calc(loss_fn, **kw):  # avoid invoking a real loss
        return torch.zeros((), requires_grad=True)

    class _LogitLoss(nn.Module):
        pass

    m = _TinyModel()
    with (
        mock.patch.object(_mtp, "calculate_loss", side_effect=fake_calc),
        mock.patch.object(
            _mtp, "_get_lm_head_weight", side_effect=AssertionError("non-fused path must not gather the LM head")
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


def test_sharded_lm_head_gathers_once_and_matches(ref_cce):
    """Sharded DTensor LM head (the OOM scenario): exactly one all-gather, and
    loss + grads match the unsharded reference."""
    from torch.distributed import Backend

    if "threaded" not in getattr(Backend, "backend_type_map", {}):
        pytest.skip("threaded c10d backend unavailable in this torch build")
    try:
        from tests.unit_tests.distributed.test_grad_utils import spawn_threads_and_init_comms
    except Exception as e:  # pragma: no cover
        pytest.skip(f"threaded dist harness unavailable: {e}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    @spawn_threads_and_init_comms(world_size=2)
    def _body(self=None):
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        torch.manual_seed(0)
        W_full = torch.randn(V, H)
        hs = [torch.randn(B, S, H, requires_grad=True) for _ in range(D)]
        labels = torch.randint(0, V, (B, S))
        m = _TinyModel(weight=distribute_tensor(W_full.clone(), mesh, [Shard(0)]))

        n, p1, p2 = _count_gathers()
        with p1, p2:
            loss = calculate_mtp_loss(
                FusedLinearCrossEntropy(reduction="sum"),
                mtp_per_depth_h=hs,
                labels=labels,
                model=m,
                scaling_factor=SF,
                num_label_tokens=NLT,
                ignore_index=IGN,
            )
        loss.backward()
        assert n["c"] == 1, f"sharded head must gather once, got {n['c']}"

        W2 = W_full.clone().requires_grad_(True)
        hs2 = [h.detach().clone().requires_grad_(True) for h in hs]
        _reference_mtp_loss(hs2, labels, W2).backward()
        torch.testing.assert_close(
            loss, _reference_mtp_loss([h.detach() for h in hs], labels, W_full), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(m.lm_head.weight.grad.full_tensor(), W2.grad, rtol=1e-4, atol=1e-6)

    _body()
