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

"""A batch with no supervised tokens must contribute 0.0, not NaN.

``num_label_tokens`` is the global DP-reduced count of non-ignored labels, so it
is 0 when every label in the batch is ``ignore_index`` -- e.g. a
gradient-accumulation window where truncation cut the answer off every sample.
The sum-reduced loss is then 0.0 and ``loss / num_label_tokens`` is ``0.0 / 0``,
which is NaN; NaN propagates through ``backward()`` into every parameter, so the
run continues with a destroyed model instead of failing.

All four losses that accept ``num_label_tokens`` must agree here. The optional
dependencies of ``linear_ce`` / ``te_parallel_ce`` are stubbed so the guard is
exercised without the real kernels, following the pattern already used in
``test_linear_ce.py``.
"""

import pytest
import torch

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

B, S, V = 2, 8, 32


def _fully_masked_batch():
    torch.manual_seed(0)
    logits = torch.randn(B, S, V)
    labels = torch.full((B, S), -100)
    num_label_tokens = int((labels != -100).sum())
    assert num_label_tokens == 0
    return logits, labels, num_label_tokens


def _assert_zero(loss):
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all(), f"expected a finite loss, got {loss}"
    assert float(loss) == 0.0


def test_masked_ce_zero_label_tokens_is_zero():
    """The reference behaviour the other three must match."""
    logits, labels, n = _fully_masked_batch()
    _assert_zero(MaskedCrossEntropy(reduction="sum")(logits, labels, num_label_tokens=n))


def test_chunked_ce_zero_label_tokens_is_zero():
    logits, labels, n = _fully_masked_batch()
    _assert_zero(ChunkedCrossEntropy(reduction="sum")(logits, labels, num_label_tokens=n))


def test_fused_linear_ce_zero_label_tokens_is_zero(monkeypatch):
    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", True)
    # Sum-reduced loss over zero supervised tokens is 0.0; the bug is 0.0 / 0.
    monkeypatch.setattr(
        linear_ce_mod,
        "linear_cross_entropy",
        lambda hidden, weight, targets=None, **kw: torch.tensor(0.0),
        raising=False,
    )

    loss_fn = linear_ce_mod.FusedLinearCrossEntropy(reduction="sum")
    out = loss_fn(torch.randn(B, S, 4), torch.full((B, S), -100), torch.randn(V, 4), num_label_tokens=0)
    _assert_zero(out)


def test_te_parallel_ce_zero_label_tokens_is_zero(monkeypatch):
    from nemo_automodel.components.loss import te_parallel_ce as te_mod

    monkeypatch.setattr(te_mod, "HAVE_TE_PARALLEL_CE", True)
    monkeypatch.setattr(
        te_mod,
        "parallel_cross_entropy",
        lambda logits, labels, eps, reduce_loss, tp_group, ignore_index: torch.zeros(B, S),
        raising=False,
    )

    loss_fn = te_mod.TEParallelCrossEntropy(reduction="sum")
    out = loss_fn(torch.randn(B, S, V), torch.full((B, S), -100), num_label_tokens=0)
    _assert_zero(out)


@pytest.mark.parametrize("reduction", ["sum"])
def test_nonzero_label_tokens_still_normalizes(reduction):
    """The guard must not disturb the normal path."""
    torch.manual_seed(0)
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    n = int((labels != -100).sum())
    assert n == B * S

    masked = MaskedCrossEntropy(reduction=reduction)(logits, labels.clone(), num_label_tokens=n)
    chunked = ChunkedCrossEntropy(reduction=reduction)(logits.clone(), labels.clone(), num_label_tokens=n)

    assert torch.isfinite(masked).all() and float(masked) > 0.0
    assert torch.isfinite(chunked).all() and float(chunked) > 0.0
    assert float(masked) == pytest.approx(float(chunked), rel=1e-5)
