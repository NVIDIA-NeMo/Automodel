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

"""Shared wire protocol between the remote target server and client.

The control plane is HTTP: the client POSTs ``input_ids`` and receives, in the
NCCL data path, only tensor *metadata* (dtype + shape) as JSON so it knows what
to ``recv``; the actual tensors arrive over NCCL. In the fallback path the body
is the binary :mod:`wire` blob instead.
"""

from __future__ import annotations

import json
from typing import Optional

import torch

from nemo_automodel.components.speculative.eagle.remote.wire import _DTYPE_TABLE, _DTYPE_TO_CODE

# HTTP endpoints (paths are matched without the leading slash by the server).
EP_GENERATE = "generate"
EP_SET_VOCAB_MAPPING = "set_vocab_mapping"
EP_MODEL_INFO = "model_info"
EP_INPUT_EMBEDDINGS = "input_embeddings"
EP_INIT_NCCL = "init_nccl"
EP_HEARTBEAT = "heartbeat"
EP_DISCONNECT = "disconnect"
EP_HEALTH = "health"

# Header the client sets to advertise NCCL capability and the server echoes to
# confirm the response tensors were sent over NCCL (body = metadata) rather than
# inline as a wire blob.
NCCL_HEADER = "X-NeMo-NCCL"

# The fixed order tensors are sent/received in for one ``generate`` response.
# Server send order and client recv order MUST agree on this list.
SUPERVISION_KEYS = ["aux_hidden_states", "target_probs", "position_mask", "input_ids", "loss_mask"]


def encode_nccl_metadata(
    tensor_dict: dict[str, Optional[torch.Tensor]],
    keys_order: list[str],
) -> bytes:
    """Encode tensor metadata (dtype code + shape) as a JSON HTTP body.

    Only metadata is encoded -- no tensor data. The client uses it to allocate
    the receive buffers before the NCCL recv.
    """
    metadata: dict[str, Optional[dict]] = {}
    for key in keys_order:
        tensor = tensor_dict.get(key)
        if tensor is None:
            metadata[key] = None
        else:
            metadata[key] = {"dtype_code": _DTYPE_TO_CODE[tensor.dtype], "shape": list(tensor.shape)}
    return json.dumps({"keys_order": keys_order, "metadata": metadata}).encode("utf-8")


def decode_nccl_metadata(raw: bytes) -> tuple[list[str], dict[str, Optional[dict]]]:
    """Decode the JSON metadata body into ``(keys_order, metadata)``."""
    payload = json.loads(raw.decode("utf-8"))
    return payload["keys_order"], payload["metadata"]


def dtype_from_code(code: int) -> torch.dtype:
    """Map a wire dtype code back to a ``torch.dtype``."""
    return _DTYPE_TABLE[code]
