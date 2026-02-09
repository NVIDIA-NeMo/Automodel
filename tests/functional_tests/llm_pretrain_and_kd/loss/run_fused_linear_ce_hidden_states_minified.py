#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""Minified regression test for FusedLinearCrossEntropy hidden_states shape handling.

This test targets a failure mode observed in distributed pretraining:
`cut_cross_entropy` asserts that `hidden_states.shape[:-1] == labels.shape`.

Some models return `output.hidden_states` as a single tensor (final hidden state) rather than
the usual tuple of per-layer hidden states. Code that blindly does `out.hidden_states[-1]`
will then drop the batch dimension and can trigger:

    assert e.size()[0:-1] == targets.size()

We validate that:
- the *bad* extraction (`hidden_states[-1]`) reproduces the assertion
- the shared extractor (`get_final_hidden_states`) returns a `[B, T, H]` tensor and avoids it
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from nemo_automodel.components.loss.linear_ce import (
    FusedLinearCrossEntropy,
    HAVE_CUT_CROSS_ENTROPY,
    MISSING_CUT_CROSS_ENTROPY_MSG,
)
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _init_distributed() -> None:
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


class _Out:
    """Minimal HF-like output that supports `"hidden_states" in out`."""

    def __init__(self, *, hidden_states):
        self.hidden_states = hidden_states

    def __contains__(self, key: str) -> bool:  # pragma: no cover
        return hasattr(self, key)


def main() -> int:
    _init_distributed()
    rank = _get_rank()

    if not HAVE_CUT_CROSS_ENTROPY:
        if rank == 0:
            print(f"SKIP: cut_cross_entropy unavailable: {MISSING_CUT_CROSS_ENTROPY_MSG}", file=sys.stderr)
        return 0

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    batch_size = 1
    seq_len = 8
    hidden_size = 32
    vocab_size = 64
    ignore_index = -100

    # Create a tensor-like `hidden_states` (final hidden state), mimicking custom models that
    # put the final hidden state directly into `output.hidden_states`.
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels.view(-1)[::7] = ignore_index
    lm_weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)

    out_tensor = _Out(hidden_states=hidden_states)

    # Reproduce the failure mode: indexing a tensor `hidden_states[-1]` drops the batch dim.
    bad_hidden_states = out_tensor.hidden_states[-1]
    try:
        _ = FusedLinearCrossEntropy(ignore_index=ignore_index, reduction="sum")(
            hidden_states=bad_hidden_states, labels=labels, lm_weight=lm_weight
        )
        if rank == 0:
            print("FAIL: Expected shape assertion for bad hidden_states extraction", file=sys.stderr)
        return 1
    except AssertionError:
        pass

    # Correct extraction: should always be [B, T, H].
    good_hidden_states = get_final_hidden_states(out_tensor)
    assert good_hidden_states is not None
    assert tuple(good_hidden_states.shape) == (batch_size, seq_len, hidden_size)

    # Also validate tuple/list style outputs.
    out_tuple = _Out(hidden_states=(hidden_states * 0.0, hidden_states * 0.5, hidden_states))
    good_hidden_states_2 = get_final_hidden_states(out_tuple)
    assert good_hidden_states_2 is not None
    assert tuple(good_hidden_states_2.shape) == (batch_size, seq_len, hidden_size)

    # If CUDA is available, run the full fused loss forward/backward.
    if not torch.cuda.is_available():
        if rank == 0:
            print("SKIP: CUDA not available (shape regression covered on CPU)", file=sys.stderr)
        return 0

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    hidden_states_cuda = hidden_states.to(device=device).detach().requires_grad_(True)
    labels_cuda = labels.to(device=device)
    lm_weight_cuda = lm_weight.to(device=device).detach().requires_grad_(True)

    loss_fn = FusedLinearCrossEntropy(ignore_index=ignore_index, reduction="sum")
    loss = loss_fn(hidden_states=hidden_states_cuda, labels=labels_cuda, lm_weight=lm_weight_cuda)

    if not torch.isfinite(loss):
        if rank == 0:
            print(f"FAIL: Non-finite loss: {loss}", file=sys.stderr)
        return 1

    loss.backward()

    if rank == 0:
        print("PASS: FusedLinearCrossEntropy hidden_states extraction is shape-safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

