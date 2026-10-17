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

"""Parity of Muown with an independent implementation of the paper's Algorithm 1.

The authors' repository (https://github.com/kcc-lion/muown, commit 3bd0c054) has no license file, so
it is not vendored. ``_algorithm1_step`` instead re-implements Algorithm 1 of arXiv:2605.10797 from
the paper, including Appendix C's Nesterov momentum, 0.2 * sqrt(max(m, n)) scaling and decoupled
weight decay. Like the authors' code, a ``[3d, d]`` matrix is orthogonalized as three ``[d, d]``
blocks with scale 0.2 * sqrt(d).
"""

import math

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("dion")

from nemo_automodel.components.optim.muown import Muown

LR, MU, BETAS, EPS, NS_EPS = 0.02, 0.95, (0.9, 0.95), 1e-8, 1e-7


@pytest.fixture(autouse=True)
def eager_optimizer():
    # Dion and Muown helpers are torch.compile'd; compare eager CPU arithmetic.
    with torch.compiler.set_stance("force_eager"):
        yield


def _newton_schulz(matrix: Tensor, steps: int = 5) -> Tensor:
    """Quintic BF16 Newton-Schulz orthogonalization, as in ``torch.optim.Muon``.

    Args:
        matrix: FP32 tensor of shape [m, n].
        steps: Polynomial iteration count.

    Returns:
        BF16 tensor of shape [m, n] approximating the orthogonal polar factor.
    """
    x = matrix.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + NS_EPS)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=-4.7750, alpha=2.0315), x, beta=3.4445)
    return x.T if tall else x


def _algorithm1_step(weight: Tensor, grad: Tensor, state: dict, *, nesterov: bool, weight_decay: float) -> Tensor:
    """Apply one Muown step from Algorithm 1 (arXiv:2605.10797) to an FP32 [m, n] matrix.

    Args:
        weight: Effective weight W of shape [m, n] (output, input).
        grad: Gradient of the loss with respect to W, shape [m, n].
        state: Holds g, r, m_g, v_g of shape [m, 1], M of shape [m, n] and step t; updated in place.
        nesterov: Use the Nesterov momentum of Algorithm 1; otherwise use M directly.
        weight_decay: Decoupled weight decay coefficient (Appendix C).

    Returns:
        New effective weight of shape [m, n].
    """
    g, r = state["g"], state["r"]
    unit = weight / g  # D = Diag(1/r) R, with R = Diag(r/g) W
    direction = unit * r
    grad_g = (grad * unit).sum(dim=1, keepdim=True)
    grad_direction = (g / r) * (grad - unit * grad_g)  # Diag(g/r) Proj_D(grad_W)
    momentum = state["M"].mul_(MU).add_(grad_direction)
    update = grad_direction.add(momentum, alpha=MU) if nesterov else momentum
    rows, cols = weight.shape
    if rows == 3 * cols:
        ortho = torch.cat([_newton_schulz(block) for block in update.split(cols)])
        scale = 0.2 * math.sqrt(cols)
    else:
        ortho = _newton_schulz(update)
        scale = 0.2 * math.sqrt(max(rows, cols))
    direction = direction.add(ortho, alpha=-LR * scale)
    state["t"] += 1
    beta1, beta2 = BETAS
    state["m_g"].mul_(beta1).add_(grad_g, alpha=1 - beta1)
    state["v_g"].mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
    denominator = (state["v_g"] / (1 - beta2 ** state["t"])).sqrt().add_(EPS)
    g.addcdiv_(state["m_g"] / (1 - beta1 ** state["t"]), denominator, value=-LR)
    state["r"] = direction.norm(dim=1, keepdim=True)
    new_weight = g * (direction / state["r"])
    if weight_decay:
        new_weight = new_weight.add(weight, alpha=-LR * weight_decay)
        g.copy_(new_weight.norm(dim=1, keepdim=True))
    return new_weight


@pytest.mark.parametrize("shape", [(16, 16), (32, 12), (12, 32), (48, 16)], ids=["square", "tall", "wide", "qkv"])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_muown_matches_paper_algorithm1(shape, nesterov, weight_decay):
    torch.manual_seed(0)
    initial = torch.randn(shape) / math.sqrt(shape[1])
    weight = nn.Parameter(initial.clone())
    optimizer = Muown(
        [weight],
        lr=LR,
        mu=MU,
        betas=BETAS,
        epsilon=EPS,
        ns_epsilon=NS_EPS,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    row_norm = initial.norm(dim=1, keepdim=True)
    expected = initial.clone()
    state = {
        "g": row_norm.clone(),
        "r": row_norm.clone(),
        "M": torch.zeros_like(initial),
        "m_g": torch.zeros_like(row_norm),
        "v_g": torch.zeros_like(row_norm),
        "t": 0,
    }
    for _ in range(5):
        grad = torch.randn(shape)
        weight.grad = grad.clone()
        optimizer.step()
        expected = _algorithm1_step(expected, grad, state, nesterov=nesterov, weight_decay=weight_decay)
        # Rows have unit-scale norms, so atol is relative to the weight scale. Both sides run the same
        # FP32 and BF16 operation order, so the comparison is expected to be exact up to this tolerance.
        torch.testing.assert_close(weight.detach(), expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(optimizer.state[weight]["g"], state["g"], rtol=0, atol=1e-6)
    # Guard against a vacuous comparison: five steps must move the weights well beyond the tolerance.
    assert (weight.detach() - initial).abs().max() > 1e-2
