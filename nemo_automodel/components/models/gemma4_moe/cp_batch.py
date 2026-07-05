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

"""Gemma4's contiguous-shard context-parallel batch sharding.

Contiguously shards the sequence across CP ranks (each rank keeps one
``seq_start:seq_end`` slice) so Gemma4 can run its own p2p ring FlexAttention
over the shards. It performs no collective -- the transport lives in Gemma4's
attention (see ``cp_attention.py``).

The implementation is the shared contiguous sharder in
``components/distributed/cp_sharder.py``; this module binds Gemma4's
model-specific arguments (vision-group-id metadata, ``_packed_seq_ids``
synthesis). Gemma4's ``prepare_model_inputs_for_cp`` exposes it through the
``CPSharder`` it returns under the ``"cp_sharder"`` batch key, which
``cp_utils.make_cp_batch_and_ctx`` invokes in place of the default
load-balanced ``context_parallel`` path.
"""

from typing import Any

from nemo_automodel.components.distributed.cp_sharder import shard_batch_contiguous


def make_contiguous_shard_cp_batch_and_ctx(
    cp_mesh,
    tp_mesh,
    batch,
    *,
    loss_mask=None,
    padding_token_id: int = 0,
    extra_seq_keys: dict[str, int] | None = None,
    extra_pad_values: dict[str, Any] | None = None,
):
    """Prepare and contiguously shard a batch for Gemma4's ring CP.

    Exposed as ``CPSharder.shard_batch`` by Gemma4's ``_cp_shard_batch``;
    ``cp_utils.make_cp_batch_and_ctx`` invokes it. Runs the shared pre-shard
    prep, then keeps one contiguous sequence slice per CP rank (no collective;
    the transport lives in Gemma4's own ring attention).
    """
    return shard_batch_contiguous(
        cp_mesh,
        tp_mesh,
        batch,
        loss_mask=loss_mask,
        padding_token_id=padding_token_id,
        extra_seq_keys=extra_seq_keys,
        extra_pad_values=extra_pad_values,
        synthesize_packed_seq_ids=True,
    )
