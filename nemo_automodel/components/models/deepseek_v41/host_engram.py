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

"""Frozen native MXFP8 Engram storage backed by a local safetensors checkpoint.

The checkpoint is an immutable external dependency, not a model parameter or
buffer. Training checkpoints contain the projections/gates; resuming requires
the same source checkpoint at the configured path on every rank.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HostEngramTableConfig:
    """Immutable local source and logical dimensions of one frozen table."""

    checkpoint: str
    layer_idx: int
    num_embeddings: int
    embedding_dim: int

    def build(self, *, dtype: torch.dtype) -> FrozenHostEngramTable:
        return FrozenHostEngramTable(self, dtype=dtype)


class FrozenHostEngramTable(nn.Module):
    """CPU mmap lookup; only selected rows are decoded and transferred.

    No table parameters/buffers are registered: FSDP, optimizers, state_dict,
    and Module.to/to_empty must never materialize or refit this external table.
    """

    def __init__(self, config: HostEngramTableConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        if config.num_embeddings <= 0 or config.embedding_dim <= 0 or config.embedding_dim % 32:
            raise ValueError("Host Engram needs positive rows and head_dim divisible by 32")
        self.config = config
        self.dtype = dtype
        checkpoint = Path(config.checkpoint)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"layers.{config.layer_idx}.engram.embed"
        self._values = self._map(
            checkpoint / index[prefix + ".weight"],
            prefix + ".weight",
            (config.num_embeddings, config.embedding_dim),
            "F8_E4M3",
        )
        self._scales = self._map(
            checkpoint / index[prefix + ".scale"],
            prefix + ".scale",
            (config.num_embeddings, config.embedding_dim // 32),
            "F8_E8M0",
        )

    @staticmethod
    def _map(path: Path, name: str, shape: tuple[int, int], dtype: str) -> np.memmap:
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 100_000_000 or length + 8 > path.stat().st_size:
                raise ValueError(f"Invalid safetensors header length: {path}")
            metadata = json.loads(stream.read(length))[name]
        start, end = metadata["data_offsets"]
        if (
            metadata["dtype"] != dtype
            or tuple(metadata["shape"]) != shape
            or start < 0
            or end - start != math.prod(shape)
            or 8 + length + end > path.stat().st_size
        ):
            raise ValueError(f"Invalid native MXFP8 tensor {name} in {path}: {metadata}")
        # Read-only mapping. Advanced indexing below copies only selected bytes.
        return np.memmap(path, mode="r", dtype=np.uint8, offset=8 + length + start, shape=shape)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Map integer IDs [...] to frozen values [..., head_dim] on ids.device."""
        if ids.device.type not in ("cpu", "cuda"):
            raise ValueError("Host Engram supports CPU and CUDA lookup only")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Host Engram IDs must be int32 or int64")
        cpu_ids = ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        unique, inverse = torch.unique(cpu_ids, sorted=True, return_inverse=True)
        if unique.numel() and (unique[0] < 0 or unique[-1] >= self.config.num_embeddings):
            raise ValueError("Host Engram row ID out of range")
        indices = unique.numpy()
        values = torch.from_numpy(self._values[indices]).view(torch.float8_e4m3fn).float()
        raw_scales = torch.from_numpy(self._scales[indices]).to(torch.int32)
        scales = torch.exp2(raw_scales.float() - 127)
        # E8M0 byte 255 denotes NaN, not an additional finite exponent.
        scales.masked_fill_(raw_scales == 255, float("nan"))
        rows = (values.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)).flatten(-2).to(self.dtype)
        if ids.device.type == "cuda":
            rows = rows.pin_memory().to(ids.device, non_blocking=True)
            inverse = inverse.pin_memory().to(ids.device, non_blocking=True)
        return rows.index_select(0, inverse).reshape(*ids.shape, self.config.embedding_dim)
