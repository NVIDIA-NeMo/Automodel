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

"""Chunked softcapped cross entropy of Inkling2 on CUDA, where its compiled functions and TF32 products run."""

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.inkling2.loss import Inkling2ChunkedLinearCrossEntropy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="The compiled loss functions run on CUDA")

USABLE, CAP = 37, 30.0


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hidden states [2, 9, 16], a head [40, 16], labels and token weights [2, 9] on CUDA.

    Hidden states and head hold BF16 values in either dtype: the loss computes with BF16 head rows,
    and its TF32 products are exact for BF16 operands only. Logits of about 16 sit where the cap of
    30 bends them, so its derivative is a visible factor of the gradients.
    """
    generator = torch.Generator().manual_seed(241)
    hidden = (4 * torch.randn(2, 9, 16, generator=generator)).bfloat16().to("cuda", dtype).requires_grad_(True)
    head = torch.randn(40, 16, generator=generator).bfloat16().float().cuda().requires_grad_(True)
    labels = torch.randint(0, USABLE, (2, 9), generator=generator)
    labels[0, 2] = labels[1, -1] = -100
    weights = torch.rand(2, 9, generator=generator)
    return hidden, head, labels.cuda(), weights.cuda()


def _fp64_token_losses(hidden: torch.Tensor, head: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy [batch, sequence] of the capped logits over the usable rows, zero where ignored.

    Args:
        hidden: FP64 hidden states [batch, sequence, hidden].
        head: FP64 head [vocab, hidden]; only the first USABLE rows enter the softmax.
        labels: Class indices [batch, sequence], -100 where ignored.
    """
    capped = CAP * torch.tanh(hidden @ head[:USABLE].T / CAP)
    return F.cross_entropy(capped.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").view_as(labels)


@pytest.mark.parametrize("weighted", [False, True], ids=["summed", "token_weights"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunked_loss_and_gradients_match_fp64_on_cuda(dtype: torch.dtype, weighted: bool) -> None:
    """Chunks that end inside a row: a summed loss takes its gradients from the forward, token weights recompute."""
    hidden, head, labels, weights = _inputs(dtype)
    loss_fn = Inkling2ChunkedLinearCrossEntropy(USABLE, logit_softcapping=CAP, chunk_tokens=5)
    loss = loss_fn(hidden, labels, head, loss_weights=weights if weighted else None)
    got_hidden, got_head = torch.autograd.grad(loss, [hidden, head])

    wide_hidden = hidden.detach().double().requires_grad_(True)
    wide_head = head.detach().double().requires_grad_(True)
    losses = _fp64_token_losses(wide_hidden, wide_head, labels)
    expected = (losses * weights if weighted else losses).sum()
    expected_hidden, expected_head = torch.autograd.grad(expected, [wide_hidden, wide_head])

    assert loss.dtype == torch.float32 and got_hidden.dtype == dtype and got_head.dtype == torch.float32
    torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=1e-5)
    # The gradient products round the FP32 logit gradients to TF32, which keeps ten mantissa bits,
    # and a BF16 hidden gradient keeps eight.
    for got, wanted in ((got_hidden, expected_hidden), (got_head, expected_head)):
        torch.testing.assert_close(got.double(), wanted, rtol=2e-2, atol=5e-3 * wanted.abs().max().item())
    assert torch.count_nonzero(got_head[USABLE:]) == 0
    assert torch.count_nonzero(got_hidden[labels == -100]) == 0


def test_chunked_loss_without_gradients_matches_fp64_on_cuda() -> None:
    """Without a gradient the forward computes the token losses alone."""
    hidden, head, labels, _ = _inputs(torch.bfloat16)
    loss_fn = Inkling2ChunkedLinearCrossEntropy(USABLE, logit_softcapping=CAP, reduction="none", chunk_tokens=5)
    with torch.no_grad():
        losses = loss_fn(hidden, labels, head)
    expected = _fp64_token_losses(hidden.detach().double(), head.detach().double(), labels)
    assert losses.shape == labels.shape and losses.dtype == torch.float32
    torch.testing.assert_close(losses.double(), expected, rtol=1e-5, atol=1e-5)
