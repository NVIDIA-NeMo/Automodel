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

"""PEFT serialization of the fixed embedding rows used by gated MTP."""

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from nemo_automodel.shared.tied_weights import get_input_embeddings_weight_and_name

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_TOKEN_ROWS_SUFFIX = ".token_adapter.trainable_tokens_delta"


def token_rows_key(model: nn.Module) -> tuple[torch.Tensor, str]:
    """Find the input embedding table and its PEFT row tensor key."""
    weight, name = get_input_embeddings_weight_and_name(model)
    if weight is None or name is None:
        raise ValueError("MTP adapter requires an input embedding table")
    return weight, f"base_model.model.{name.removesuffix('.weight')}{_TOKEN_ROWS_SUFFIX}"


@torch.no_grad()
def transfer_token_rows(
    weight: torch.Tensor, ids: list[int], *, saved_rows: torch.Tensor | None = None
) -> torch.Tensor:
    """Collect selected embedding rows, or install saved rows into their owning shards.

    Args:
        weight: Embedding matrix of shape [vocab, hidden]. DTensors may shard or replicate
            either axis. Call on every rank in the weight mesh.
        ids: Ordered vocabulary row IDs.
        saved_rows: If provided, install this [tokens, hidden] tensor in place.

    Returns:
        The CPU row tensor of shape [tokens, hidden].
    """
    vocab, hidden = weight.shape
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < vocab for i in ids):
        raise ValueError(f"Invalid MTP embedding IDs for vocabulary of size {vocab}: {ids}")
    if saved_rows is not None and (saved_rows.shape != (len(ids), hidden) or not saved_rows.is_floating_point()):
        raise ValueError(f"Expected floating-point MTP embedding rows of shape {(len(ids), hidden)}")

    local = weight
    start, column = 0, 0
    if isinstance(weight, DTensor):
        if any(not isinstance(p, (Replicate, Shard)) for p in weight.placements):
            raise ValueError("MTP embedding rows require sharded or replicated weights")
        _, (start, column) = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
        local = weight.to_local()

    indices = torch.tensor(ids, device=local.device) - start
    owned = (indices >= 0) & (indices < local.shape[0])
    columns = slice(column, column + local.shape[1])
    if saved_rows is not None:
        values = saved_rows[:, columns].to(local)
        local.index_copy_(0, indices[owned], values[owned])
        return saved_rows

    rows = local.new_zeros((len(ids), hidden))
    rows[owned, columns] = local.index_select(0, indices[owned])
    if isinstance(weight, DTensor):
        for axis, placement in enumerate(weight.placements):
            if isinstance(placement, Shard):
                torch.distributed.all_reduce(rows, group=weight.device_mesh.get_group(axis))
    return rows.cpu()


def tokenizer_with_mask_tokens(
    tokenizer: "PreTrainedTokenizerBase", vocab_size: int, mask_ids: list[int]
) -> "PreTrainedTokenizerBase":
    """Name the reserved slots on a copy so a PEFT resize retains the trained vocabulary."""
    if len(tokenizer) > vocab_size:
        raise ValueError("MTP tokenizer is larger than the input embedding table")
    if len(tokenizer) == vocab_size:
        return tokenizer
    exported = deepcopy(tokenizer)
    mask_depths = {token_id: depth for depth, token_id in enumerate(mask_ids, 1)}
    first = len(exported)
    tokens = [
        f"<|automodel_mtp_mask_{mask_depths[i]}|>" if i in mask_depths else f"<|automodel_reserved_{i}|>"
        for i in range(first, vocab_size)
    ]
    exported.add_tokens(tokens, special_tokens=True)
    if len(exported) != vocab_size or exported.convert_tokens_to_ids(tokens) != list(range(first, vocab_size)):
        raise ValueError("Could not preserve MTP vocabulary IDs in the exported tokenizer")
    return exported
