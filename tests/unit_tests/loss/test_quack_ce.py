# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.quack_ce import QuackCrossEntropy


def test_quack_cross_entropy_uses_shared_masking_and_normalization():
    calls = {}

    def fake_cross_entropy(logits, labels, **kwargs):
        calls.update(kwargs)
        return F.cross_entropy(logits, labels, **kwargs)

    with patch(
        "nemo_automodel.components.loss.quack_ce.safe_import_from",
        return_value=(True, fake_cross_entropy),
    ):
        loss_fn = QuackCrossEntropy(ignore_index=-7, reduction="sum")

    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.randint(0, 5, (2, 3))
    labels_ref = labels.clone()
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    labels_ref[mask == 0] = -7

    actual = loss_fn(logits, labels, mask=mask, num_label_tokens=4)
    expected = F.cross_entropy(logits.view(-1, 5), labels_ref.view(-1), ignore_index=-7, reduction="sum") / 4

    torch.testing.assert_close(actual, expected)
    assert calls == {"ignore_index": -7, "reduction": "sum"}


def test_quack_cross_entropy_reports_missing_dependency():
    with (
        patch("nemo_automodel.components.loss.quack_ce.safe_import_from", return_value=(False, None)),
        pytest.raises(ImportError, match="quack-kernels"),
    ):
        QuackCrossEntropy()
