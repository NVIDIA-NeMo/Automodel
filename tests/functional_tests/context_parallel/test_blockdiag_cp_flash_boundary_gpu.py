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

"""Scheduled 1-GPU stress test for the FlashAttention boundary-segment guard.

The left-straddling boundary document (``sk > sq``) is the layout that provoked
asynchronous illegal-address reports in some FlashAttention builds, which
:func:`_flash_varlen_with_long_prefix_guard` peels onto a fixed-shape kernel. This
sweep exercises a large set of production-sized shapes whose query/key segment
lengths bracket FlashAttention tile boundaries, plus maximally packed layouts with
many non-tile-aligned tails. Every case runs the guard through a non-reentrant
activation checkpoint (the wrapper that first surfaced the failure, replaying the
attention forward during backward) and asserts finite forward and gradient
outputs. Explicit synchronizations pin any delayed illegal-address report to the
exact case that launched the offending kernel.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.distributed.blockdiag_cp.kernels import (
    _flash_varlen_with_long_prefix_guard,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="flash boundary stress requires CUDA")

H_Q = 32
H_KV = 8
HEAD_DIM = 128
LOCAL_LEN = 8192


def _boundary_cases() -> list[tuple[int, int]]:
    # (local query tokens in the boundary document, tokens inherited from the
    # left CP rank). Values bracket FlashAttention tile boundaries and include
    # the very asymmetric layouts omitted by the original three regressions.
    queries = [
        1,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        95,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        8191,
        8192,
    ]
    backs = [1, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 2048, 4095, 4096, 8191, 8192]
    cases = [(q, backs[index % len(backs)]) for index, q in enumerate(queries)]
    cases.extend((q, back) for q in (1, 17, 65, 129, 513, 1025, 2049, 4097, 8191) for back in (8191, 8192))
    return cases


def _tail_lengths(total: int, segments: int) -> list[int]:
    assert total >= segments > 0
    base, remainder = divmod(total, segments)
    return [base + (index < remainder) for index in range(segments)]


def _all_cases() -> list[tuple[str, int, int, list[int]]]:
    cases = [(f"boundary_q{q}_back{back}", q, q + back, []) for q, back in _boundary_cases()]
    # Production permits at most 512 packed documents. Exercise the maximum
    # segment count as well as mixed non-tile-aligned tails after an asymmetric
    # first segment.
    for first_q, back, tail_segments in (
        (1, 8192, 511),
        (17, 8191, 511),
        (129, 4097, 127),
        (2049, 8192, 31),
        (4097, 4095, 7),
    ):
        tails = _tail_lengths(LOCAL_LEN - first_q, tail_segments)
        cases.append(
            (
                f"tail_q{first_q}_back{back}_segments{tail_segments + 1}",
                first_q,
                first_q + back,
                tails,
            )
        )
    return cases


def _run_case(name: str, first_q: int, first_k: int, tails: list[int], device: torch.device) -> None:
    q_lengths = [first_q, *tails]
    k_lengths = [first_k, *tails]
    q_total = sum(q_lengths)
    k_total = sum(k_lengths)
    assert 0 < q_total <= LOCAL_LEN
    assert first_k >= first_q

    generator = torch.Generator(device=device).manual_seed(1729 + first_q * 17 + first_k * 31 + len(tails))
    query = torch.randn(
        q_total,
        H_Q,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    key = torch.randn(
        k_total,
        H_KV,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    value = torch.randn_like(key, requires_grad=True)
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _flash_varlen_with_long_prefix_guard(
            q,
            k,
            v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max(q_lengths),
            max_k=max(k_lengths),
            local_query_len=LOCAL_LEN,
            scale=HEAD_DIM**-0.5,
            dropout_p=0.0,
            meta={
                "first_q": first_q,
                "first_k": first_k,
                "max_tail": max(tails) if tails else 0,
            },
        )

    # Match the production text-attention wrapper: the failure that motivated
    # this stress test was observed while a non-reentrant checkpoint was
    # replaying an attention forward during backward.
    output = checkpoint(
        attention,
        query,
        key,
        value,
        use_reentrant=False,
        preserve_rng_state=False,
        determinism_check="none",
    )
    torch.cuda.synchronize(device)
    assert output.shape == query.shape, name
    assert torch.isfinite(output).all(), name
    output.float().square().mean().backward()
    torch.cuda.synchronize(device)
    for tensor in (query, key, value):
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name


@requires_cuda
def test_flash_boundary_shape_stress():
    """Sweep the boundary-guard shapes on a single GPU with checkpointed replay."""
    pytest.importorskip("flash_attn")
    device = torch.device("cuda", torch.cuda.current_device())
    for case in _all_cases():
        _run_case(*case, device)
        gc.collect()
        torch.cuda.empty_cache()
