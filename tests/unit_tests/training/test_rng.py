# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import random
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from nemo_automodel.training.rng import StatefulRNG, init_all_rng


def _next_values():
    """Return a tuple with one sample from each RNG backend."""
    return (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),
    )


def test_init_all_rng_reproducibility():
    """
    Calling ``init_all_rng`` twice with the same seed must reproduce
    identical sequences from all three RNG libraries.
    """
    init_all_rng(123)

    ref_vals = _next_values()  # reference sequence
    init_all_rng(123)  # reset with the same seed
    new_vals = _next_values()

    assert ref_vals == new_vals


def test_init_all_rng_uniqueness():
    """
    Different seeds should *change* the produced random numbers.
    """
    init_all_rng(1)
    vals_1 = _next_values()
    init_all_rng(2)
    vals_2 = _next_values()

    assert vals_1 != vals_2


def test_init_all_rng_ranked(monkeypatch):
    """
    When ``ranked=True`` the effective seed must be ``seed + rank`` where
    *rank* is provided by ``torch.distributed``.
    """
    # Create / patch a minimal ``torch.distributed`` facade
    rank = 7

    # If torch.distributed already exists, patch required attrs;
    # otherwise register a stub module.
    if "torch.distributed" not in sys.modules:
        dist_stub = ModuleType("torch.distributed")
        sys.modules["torch.distributed"] = dist_stub
    dist = sys.modules["torch.distributed"]

    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: rank, raising=False)

    base_seed = 10
    init_all_rng(base_seed, ranked=True)
    expected = random.Random(base_seed + rank).random()  # local reference
    actual = random.random()

    assert pytest.approx(expected) == actual


def test_stateful_rng_restores_state():
    """
    After leaving the ``StatefulRNG`` context, the *global* RNG states must be
    bit-for-bit identical to what they were before entering.
    """
    # Establish a known starting point
    init_all_rng(42)
    pre_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    with StatefulRNG(None):
        # Advance RNGs arbitrarily
        _ = [_next_values() for _ in range(5)]

    post_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )

    assert pre_state[0] == post_state[0]
    # NumPy & torch states are numpy arrays / tensors – use dedicated checks
    assert all(np.array_equal(a, b) for a, b in zip(pre_state[1][1:], post_state[1][1:]))
    assert torch.equal(pre_state[2], post_state[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_stateful_rng_cuda_state_restored():
    """
    Same as the previous test but for *all* CUDA streams.
    """
    init_all_rng(99)
    pre_cuda = torch.cuda.get_rng_state_all()

    with StatefulRNG(777):
        _ = torch.cuda.FloatTensor(10).uniform_()  # advance CUDA RNG

    post_cuda = torch.cuda.get_rng_state_all()
    assert all(torch.equal(a, b) for a, b in zip(pre_cuda, post_cuda))
