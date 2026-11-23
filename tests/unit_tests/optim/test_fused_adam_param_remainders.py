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

"""FP32 master weights of TE FusedAdam: explicit FP32 copies against ``store_param_remainders``.

``store_param_remainders=True`` keeps a BF16 parameter plus the 16 low bits of its FP32 master instead of the
whole FP32 master (2 bytes per element instead of 4). The tests state what that changes:

* the FP32 master weights and both moments follow the same trajectory, bit for bit;
* the BF16 parameter the model computes with is the master rounded to nearest in both modes and differs only
  where the master lies exactly halfway between two BF16 values: the explicit copy rounds that tie to even,
  the remainder split rounds it away from zero.

The reconstruction of the master from parameter and remainder is written out here and does not use the optimizer.
"""

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common.utils import HAVE_TE
from nemo_automodel.shared.te_patches import apply_te_patches

pytestmark = pytest.mark.skipif(
    not (HAVE_TE and torch.cuda.is_available()),
    reason="requires Transformer Engine and CUDA",
)

STEPS = 8
SHAPES = ((2048, 2048), (1024, 96), (4096,), (7, 3))
LR = 3.0e-4
BETAS = (0.9, 0.95)
EPS = 1.0e-8
WEIGHT_DECAY = 0.1


@dataclass
class _Snapshot:
    """State of one parameter after an optimizer step."""

    param_bits: torch.Tensor
    master_bits: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


def _bf16_bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().view(torch.int16).to(torch.int32) & 0xFFFF


def _fp32_bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().view(torch.int32).clone()


def _master_bits_from_remainder(param: torch.Tensor, remainder: torch.Tensor) -> torch.Tensor:
    """Bits of the FP32 master that a BF16 parameter and its int16 remainder encode.

    The parameter holds the high half of the master, incremented by one when the low half, read as a signed
    16-bit integer, is negative (low half >= 0x8000); the remainder is the low half.
    """
    high = _bf16_bits(param) - (remainder < 0).to(torch.int32)
    low = remainder.to(torch.int32) & 0xFFFF
    return (high << 16) | low


def _initial_values(seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [(0.02 * torch.randn(shape, generator=generator, device="cuda")).to(torch.bfloat16) for shape in SHAPES]


def _gradients(step: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1000 + step)
    return [(1.0e-3 * torch.randn(shape, generator=generator, device="cuda")).to(torch.bfloat16) for shape in SHAPES]


def _trajectory(*, master_weights: bool, store_param_remainders: bool) -> list[list[_Snapshot]]:
    """Run ``STEPS`` AdamW steps on the same initial values and gradients; snapshot every parameter after each."""
    from transformer_engine.pytorch.optimizers import FusedAdam

    apply_te_patches()
    params = [torch.nn.Parameter(value.clone()) for value in _initial_values(seed=0)]
    optimizer = FusedAdam(
        params,
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        adam_w_mode=True,
        master_weights=master_weights,
        master_weight_dtype=torch.float32,
        store_param_remainders=store_param_remainders,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    trajectory = []
    for step in range(STEPS):
        for param, grad in zip(params, _gradients(step)):
            param.grad = grad
        optimizer.step()
        snapshots = []
        for param in params:
            state = optimizer.state[param]
            if not master_weights:
                master_bits = _fp32_bits(param.detach().float())
            elif store_param_remainders:
                assert state["master_param"].dtype == torch.int16
                master_bits = _master_bits_from_remainder(param, state["master_param"])
            else:
                assert state["master_param"].dtype == torch.float32
                master_bits = _fp32_bits(state["master_param"])
            snapshots.append(
                _Snapshot(
                    param_bits=_bf16_bits(param),
                    master_bits=master_bits,
                    exp_avg=state["exp_avg"].detach().clone(),
                    exp_avg_sq=state["exp_avg_sq"].detach().clone(),
                )
            )
        trajectory.append(snapshots)
    return trajectory


def _reference_masters() -> list[list[torch.Tensor]]:
    """AdamW in float64 on the same initial values and gradients, written out from the update rule."""
    masters = [value.double() for value in _initial_values(seed=0)]
    exp_avgs = [torch.zeros_like(master) for master in masters]
    exp_avg_sqs = [torch.zeros_like(master) for master in masters]
    beta1, beta2 = BETAS
    trajectory = []
    for step in range(STEPS):
        correction1 = 1.0 - beta1 ** (step + 1)
        correction2 = 1.0 - beta2 ** (step + 1)
        for index, grad in enumerate(_gradients(step)):
            grad = grad.double()
            exp_avgs[index] = beta1 * exp_avgs[index] + (1.0 - beta1) * grad
            exp_avg_sqs[index] = beta2 * exp_avg_sqs[index] + (1.0 - beta2) * grad * grad
            update = (exp_avgs[index] / correction1) / ((exp_avg_sqs[index] / correction2).sqrt() + EPS)
            masters[index] = masters[index] - LR * (update + WEIGHT_DECAY * masters[index])
        trajectory.append([master.clone() for master in masters])
    return trajectory


def test_remainders_keep_the_fp32_master_and_moment_trajectory_bit_for_bit():
    explicit = _trajectory(master_weights=True, store_param_remainders=False)
    remainders = _trajectory(master_weights=True, store_param_remainders=True)

    for step in range(STEPS):
        for full, split in zip(explicit[step], remainders[step]):
            assert torch.equal(full.master_bits, split.master_bits), f"FP32 master differs at step {step}"
            assert torch.equal(full.exp_avg, split.exp_avg), f"first moment differs at step {step}"
            assert torch.equal(full.exp_avg_sq, split.exp_avg_sq), f"second moment differs at step {step}"


def test_both_modes_follow_adamw_on_the_fp32_master():
    reference = _reference_masters()
    for store_param_remainders in (False, True):
        trajectory = _trajectory(master_weights=True, store_param_remainders=store_param_remainders)
        for step in range(STEPS):
            for snapshot, expected in zip(trajectory[step], reference[step]):
                master = snapshot.master_bits.view(torch.float32).double()
                torch.testing.assert_close(master, expected, rtol=2.0e-5, atol=1.0e-9)


def test_bf16_parameters_differ_only_at_exact_ties_by_one_unit():
    explicit = _trajectory(master_weights=True, store_param_remainders=False)
    remainders = _trajectory(master_weights=True, store_param_remainders=True)

    ties = 0
    differing = 0
    elements = 0
    for step in range(STEPS):
        for full, split in zip(explicit[step], remainders[step]):
            master = full.master_bits
            low = master & 0xFFFF
            high = (master >> 16) & 0xFFFF
            # Round to nearest, ties to even (the cast of the explicit copy) and ties away from zero (the split).
            nearest_even = high + ((low > 0x8000) | ((low == 0x8000) & ((high & 1) == 1))).to(torch.int32)
            nearest_away = high + (low >= 0x8000).to(torch.int32)
            assert torch.equal(full.param_bits, nearest_even & 0xFFFF)
            assert torch.equal(split.param_bits, nearest_away & 0xFFFF)

            tie_kept_by_even = (low == 0x8000) & ((high & 1) == 0)
            mismatch = full.param_bits != split.param_bits
            assert torch.equal(mismatch, tie_kept_by_even)
            assert torch.all((split.param_bits - full.param_bits)[mismatch] == 1)
            ties += int((low == 0x8000).sum())
            differing += int(mismatch.sum())
            elements += master.numel()

    # The claim is only exercised if exact ties occur: about elements / 65536 of them are expected.
    assert ties >= 100, f"only {ties} exact ties in {elements} element-steps"
    assert 0 < differing < ties
    assert differing / elements < 1.0e-4


def test_without_master_weights_the_trajectory_is_a_different_one():
    """Control: the comparison above is sensitive to losing the FP32 master."""
    explicit = _trajectory(master_weights=True, store_param_remainders=False)
    plain = _trajectory(master_weights=False, store_param_remainders=False)

    final_full = explicit[-1][0]
    final_plain = plain[-1][0]
    differing = int((final_full.param_bits != final_plain.param_bits).sum())
    assert differing > 0.01 * final_full.param_bits.numel()
