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

"""KL divergence and DTensor gather helpers for capability validation.

Adapted from
``tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py``
(``_kl_divergence_from_logits`` and ``_maybe_gather_dtensor_to_replicated_local``).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate


def kl_divergence_from_logits(
    *, reference_logits: torch.Tensor, candidate_logits: torch.Tensor
) -> torch.Tensor:
    """Return per-token KL(reference || candidate) for full ``[*, V]`` logits.

    Both inputs are expected to be full (non-sharded) logits with matching
    shapes. The leading dimensions are flattened so the result is
    ``[batch_tokens]``.

    Args:
        reference_logits: Reference logits tensor (any shape ending in vocab).
        candidate_logits: Candidate logits tensor with matching shape.

    Returns:
        1-D tensor of per-token KL divergence values.
    """
    assert reference_logits.shape == candidate_logits.shape, (
        f"Shape mismatch: ref={tuple(reference_logits.shape)} vs cand={tuple(candidate_logits.shape)}"
    )
    vocab_size = reference_logits.shape[-1]
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_log_probs = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    # F.kl_div expects input=log(q), target=log(p) when log_target=True -> KL(p || q).
    return F.kl_div(cand_log_probs, ref_log_probs, reduction="none", log_target=True).sum(-1)


def maybe_gather_dtensor_to_replicated_local(
    x: torch.Tensor | DTensor, *, mesh: DeviceMesh
) -> torch.Tensor:
    """If ``x`` is a DTensor sharded across ``mesh``, gather to a replicated local tensor.

    No-op when ``x`` is already a regular tensor.

    Args:
        x: A possibly-DTensor logits tensor.
        mesh: The device mesh ``x`` may be sharded over.

    Returns:
        A plain ``torch.Tensor`` with the full (replicated) data.
    """
    if isinstance(x, DTensor):
        x = x.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
    return cast(torch.Tensor, x)
