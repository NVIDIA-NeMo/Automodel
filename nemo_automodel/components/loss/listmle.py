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

"""ListMLE ranking loss over a teacher's top-k tokens.

Distillation objectives such as :func:`~nemo_automodel.components.loss.soft_ce
.masked_soft_cross_entropy` match the teacher's probability *mass*. A drafter in
speculative decoding is instead accepted or rejected on whether its ranking of
the next few candidates agrees with the target's, so ViSpec (arXiv:2509.15235)
adds a Plackett-Luce ranking term next to the distribution term. It is shared
here rather than kept beside one algorithm because both the EAGLE-1/2 and the
ViSpec objectives use it.
"""

from __future__ import annotations

import torch


def listmle_loss(logits: torch.Tensor, target_probs: torch.Tensor, topk: int) -> torch.Tensor:
    """ListMLE ranking loss over the target's top-k tokens.

    Scores the student on reproducing the target's *ordering* of its ``topk``
    most likely tokens: the Plackett-Luce likelihood of drawing those tokens,
    under the student's logits, in the target's own descending-probability
    order.

    Args:
        logits: Tensor of shape [tokens, vocab] -- the student's logits at the
            supervised positions.
        target_probs: Tensor of shape [tokens, vocab] -- the target's
            probabilities at the same positions.
        topk: Number of top target tokens to rank.

    Returns:
        Scalar Tensor: the mean over ``tokens`` of the summed negative
        log-likelihood of the target's top-k ordering.
    """
    _, topk_indices = torch.topk(target_probs, k=topk, dim=-1)
    topk_logits = logits.gather(-1, topk_indices).float()
    # log sum_{j >= i} exp(logit_j) for every rank i, computed as a reversed
    # cumulative logsumexp so the tail sums come out in one pass.
    log_denominator = torch.flip(torch.logcumsumexp(torch.flip(topk_logits, dims=[-1]), dim=-1), dims=[-1])
    return -torch.mean((topk_logits - log_denominator).sum(dim=-1))
