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

from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)

def _safe_flatten(mesh: DeviceMesh, dim_names: Sequence[str], flat_name: str) -> None:
    """Flatten `mesh[dim_names]` into one new dimension named `flat_name`.

    Uses public `flatten` if available; falls back to private `_flatten`.
    """
    sub = mesh[tuple(dim_names)]
    if hasattr(sub, "flatten"):
        sub.flatten(mesh_dim_name=flat_name)
    else:  # PyTorch <2.2 uses a private method
        sub._flatten(mesh_dim_name=flat_name)  # type: ignore[attr-defined]

def get_world_size():
    """ returns world size """
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
    else:
        # Try to read from env (torchrun sets it), else assume 1
        ws = int(os.environ.get("WORLD_SIZE", "1"))
    return ws

@dataclass(slots=True)
class ParallelDims:
    """Logical parallelism description.

    +----------------------------------------------------------+
    | dp_replicate | dp_shard | cp | tp | pp | ep | world_size |
    +----------------------------------------------------------+
    | 1            | 1        | 1  | 1  | 1  | 1  | 1          | single-device
    +----------------------------------------------------------+
    | >1           | 1        | 1  | 1  | 1  | 1  | >1         | DDP
    +----------------------------------------------------------+
    | 1            | >1       | 1  | 1  | 1  | 1  | >1         | FSDP
    +----------------------------------------------------------+
    | >1           | >1       | 1  | 1  | 1  | 1  | >1         | HSDP
    +----------------------------------------------------------+

    Each attribute is the degree of parallelism for that axis.
      dp_replicate : data-parallel replication (DDP semantics)
      dp_shard     : data-parallel sharding (ZeRO style)
      cp           : chunk / spatial tensor parallel
      tp           : tensor parallel (column/row)
      pp           : pipeline parallel
      ep           : expert parallel
      world_size   : total processes
      enable_loss_parallel : optional loss-parallel flag
    """

    dp_replicate: int = 1
    dp_shard: int = -1           # ‑1 means “infer from world_size”
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    world_size: int = 1
    enable_loss_parallel: bool = False

    def __post_init__(self) -> None:
        self._validate_and_infer()

    def _validate_and_infer(self) -> None:
        """Validate config and, when dp_shard == -1, infer its value."""
        # Check type
        for attr_name in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"):
            val = getattr(self, attr_name)
            if not isinstance(val, int):
                raise TypeError(f"{attr_name} must be int (got {type(val).__name__})")

        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")

        # single-GPU guard
        if self.world_size == 1:

            multi_dims = [(k, getattr(self, k))
                for k in
                ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep")
                if getattr(self, k) > 1
            ]
            if multi_dims:
                raise ValueError(
                    f"world_size==1 but parallel dims {multi_dims} > 1. "
                    "Either launch more ranks or set them to 1."
                )

        # non-negative checks
        for n in ("dp_replicate", "cp", "tp", "pp", "ep"):
            val = getattr(self, n)
            if val < 1:
                raise ValueError(f"{n} must be >= 1 (got {val})")

        if self.dp_shard not in (-1, 1) and self.dp_shard < 1:
            raise ValueError("dp_shard must be -1 or >= 1")

        # infer dp_shard if requested
        if self.dp_shard == -1:
            denom = self.dp_replicate * self.cp * self.tp * self.pp
            if self.world_size % denom:
                raise ValueError(
                    f"world_size={self.world_size} not divisible by "
                    f"dp_replicate*cp*tp*pp={denom} (cannot infer dp_shard)"
                )
            self.dp_shard = self.world_size // denom

        # per-dim divisibility / range
        for n in ("dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"):
            v = getattr(self, n)
            if v > self.world_size:
                raise ValueError(f"{n} ({v}) cannot exceed world_size ({self.world_size})")
            if self.world_size % v:
                raise ValueError(f"world_size ({self.world_size}) must be divisible by {n} ({v})")

        # final product check
        total = (
            self.dp_replicate * self.dp_shard *
            self.cp * self.tp * self.pp * self.ep
        )
        if total != self.world_size:
            raise ValueError(
                f"Product of dims ({total}) != world_size ({self.world_size})"
            )

    @property
    def dp_enabled(self) -> bool:
        """ bool indicating whether data parallelism is enabled """
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        """ bool indicating whether data replication is enabled """
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        """ bool indicating whether dp_shard is enabled """
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        """ bool indicating whether context parallelism is enabled """
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        """ bool indicating whether tensor parallelism is enabled """
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        """ bool indicating whether pipeline parallelism is enabled """
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        """ bool indicating whether expert parallelism is enabled """
        return self.ep > 1

    @property
    def loss_parallel_enabled(self) -> bool:
        """ bool indicating whether loss parallelism is enabled """
        return self.tp > 1 and self.enable_loss_parallel

    @property
    def non_data_parallel_size(self) -> int:
        """ non-data parallel size """
        return self.cp * self.tp * self.pp * self.ep

    # TODO(akoumparouli): switch to enum instead of is_ddp/is_single_device
    @property
    def is_ddp(self):
        """ bool flag indicating whether it's DDP """
        return self.non_data_parallel_size == 1 and self.dp_shard == 1 and self.dp_replicate > 1

    @property
    def is_single_device(self):
        """ bool flag indicating whether it's single gpu """
        return self.non_data_parallel_size == 1 and self.dp_shard == 1 and self.dp_replicate == 1

    # ---------------- Mesh builder ---------------- #
    def build_mesh(self, device_type: str = "cuda") -> DeviceMesh:
        """Construct a DeviceMesh containing only dims with size>1."""
        dims, names = [], []
        for size, n in (
            (self.dp_replicate, "dp_replicate"),
            (self.dp_shard, "dp_shard"),
            (self.pp, "pp"),
            (self.cp, "cp"),
            (self.tp, "tp"),
            (self.ep, "ep"),
        ):
            if size > 1:
                dims.append(size)
                names.append(n)

        if not dims:  # all ones → 0-D mesh
            logger.info("All parallel dims ==1 → degenerate 0-D mesh")
            return init_device_mesh(device_type, [])  # empty mesh

        logger.info("Building %d-D device mesh: names=%s, sizes=%s", len(dims), names, dims)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Flatten commonly-used composite meshes so that downstream code
        # can obtain a dedicated process group quickly.
        dp_names, dp_shard_cp_names, dp_cp_names = [], [], []

        if self.dp_replicate_enabled:
            dp_names.append("dp_replicate")
            dp_cp_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_names.append("dp_shard")
            dp_shard_cp_names.append("dp_shard")
            dp_cp_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_names.append("cp")
            dp_cp_names.append("cp")

        if dp_names:
            _safe_flatten(mesh, dp_names, "dp")
        if dp_shard_cp_names:
            _safe_flatten(mesh, dp_shard_cp_names, "dp_shard_cp")
        if dp_cp_names:
            _safe_flatten(mesh, dp_cp_names, "dp_cp")

        return mesh
