# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# taken from https://raw.githubusercontent.com/nvidia-cosmos/cosmos-rl/a093202bb2221bf1751996dd830ea5f47e786544/cosmos_rl/utils/parallelism.py

import logging
import os
from dataclasses import dataclass
from enum import Enum

import numpy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class DimNames(str, Enum):
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP = "dp"
    CP = "cp"
    TP = "tp"
    EP = "ep"
    PP = "pp"
    DP_CP = "dp_cp"
    DP_CP_TP = "dp_cp_tp"


@dataclass(frozen=True)
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    pp_dynamic_shape: bool
    ep: int = 1
    # When ep is enabled, we can have different dp shard for the MoE module.
    # For example, suppose we have 64 GPUs, then we can have dp_shard equal
    # to 64 for the attention module, and have ep = 4, dp_shard_with_ep = 16
    # for the MoE module.
    dp_shard_with_ep: int = -1

    def __post_init__(self):
        if self.pp > 1:
            raise ValueError("Pipeline parallelism is not supported yet.")
        if self.ep > 1:
            raise ValueError("Exponential parallelism is not supported yet.")
        if self.dp_shard_with_ep > 1:
            raise ValueError("Exponential parallelism is not supported yet.")
        if self.pp_dynamic_shape:
            raise ValueError("Pipeline parallelism dynamic shape is not supported yet.")

        self._validate()
        self.build_mesh_info()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp, ep, dp_shard_with_ep = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.dp_shard_with_ep,
        )
        for d in (dp_replicate, cp, tp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must be -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1, f"dp_shard of size {dp_shard} is not valid, should be equal or greater than 1"

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        # Checks for MoE weights. Note that dp_shard is only used for the non-MoE weights.
        if ep > 1:
            assert dp_shard_with_ep == -1 or dp_shard_with_ep >= 1, " dp_shard_with_ep must -1 or >=1."
            if dp_shard_with_ep < 0:
                logging.info(
                    "dp_shard_with_ep is set to -1, will be automatically determined based "
                    f"on self.world_size {self.world_size} // {pp * ep * tp}."
                )
                self.dp_shard_with_ep = dp_shard_with_ep = self.world_size // (pp * ep * tp)
                logging.info(f"dp_shard_with_ep is set to {dp_shard_with_ep}.")
            assert dp_shard_with_ep >= 1, (
                f"WORLD_SIZE({self.world_size}) is not a multiple of pp({pp}) * ep({ep}) * tp({tp})"
            )

            if tp > 1:
                raise ValueError("tp must be 1 when ep > 1, since we only support MoE with pipeline parallelism.")

            if pp > 1 and dp_shard_with_ep > 1:
                raise ValueError(
                    "dp_shard_with_ep must be 1 when pp > 1, since we only support EP with pipeline parallelism."
                )

            if (pp * dp_shard_with_ep * ep * tp) != self.world_size:
                raise ValueError(
                    f"Invalid parallel dims: pp({pp}) * dp_shard_with_ep({dp_shard_with_ep}) * "
                    f"ep({ep}) * tp({tp}) != WORLD_SIZE({self.world_size})"
                )

    def build_mesh(self, device_or_backend_type: str) -> DeviceMesh:
        if device_or_backend_type in ("cuda", "cpu"):
            device_type = device_or_backend_type
        elif device_or_backend_type == "nccl":
            device_type = "cuda"
        elif device_or_backend_type == "gloo":
            device_type = "cpu"
        else:
            raise ValueError(f"Invalid device or backend type: {device_or_backend_type}")

        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            [
                DimNames.PP,
                DimNames.DP_REPLICATE,
                DimNames.DP_SHARD,
                DimNames.CP,
                DimNames.TP,
            ],  # reverse order to apply N-dim prallel.
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        return self._build_mesh(device_type, dims, names)

    def _build_mesh(
        self,
        device_type: str,
        dims: list[int],
        names: list[str],
    ) -> DeviceMesh:
        logging.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_cp_mesh_dim_names = []
        # Mesh useful for TP-merged FSDP
        dp_cp_tp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append(DimNames.DP_REPLICATE)
            dp_cp_mesh_dim_names.append(DimNames.DP_REPLICATE)
            dp_cp_tp_mesh_dim_names.append(DimNames.DP_REPLICATE)
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append(DimNames.DP_SHARD)
            dp_shard_cp_mesh_dim_names.append(DimNames.DP_SHARD)
            dp_cp_tp_mesh_dim_names.append(DimNames.DP_SHARD)
            dp_cp_mesh_dim_names.append(DimNames.DP_SHARD)
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append(DimNames.CP)
            dp_cp_tp_mesh_dim_names.append(DimNames.CP)
            dp_cp_mesh_dim_names.append(DimNames.CP)
        if self.tp_enabled:
            dp_cp_tp_mesh_dim_names.append(DimNames.TP)

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name=DimNames.DP)
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name=DimNames.DP_SHARD_CP)
        if dp_cp_tp_mesh_dim_names != []:
            mesh[tuple(dp_cp_tp_mesh_dim_names)]._flatten(mesh_dim_name=DimNames.DP_CP_TP)
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name=DimNames.DP_CP)

        self.mesh = mesh
        return mesh

    def get_rank_in_dim(self, mesh_dim_name: str, global_rank: int) -> int:
        if hasattr(self, "full_rank_info"):
            if mesh_dim_name in self.full_rank_info[global_rank]:
                return self.full_rank_info[global_rank][mesh_dim_name]
            else:
                raise ValueError(f"Mesh dim {mesh_dim_name} not found in rank info.")
        else:
            raise ValueError("full_rank_info is not set. Please call build_mesh() first.")

    def get_size_in_dim(self, mesh_dim_name: str) -> int:
        if hasattr(self, "full_world_size_info"):
            if mesh_dim_name in self.full_world_size_info:
                return self.full_world_size_info[mesh_dim_name]
            else:
                raise ValueError(f"Mesh dim {mesh_dim_name} not found in world size info.")
        else:
            try:
                return self.mesh.get_group(mesh_dim_name).size()
            except Exception:
                pass
            return 1

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def dp_shard_with_ep_enabled(self) -> bool:
        return self.dp_shard_with_ep > 1

    @property
    def pp_dynamic_shape_enabled(self) -> bool:
        return self.pp > 1 and self.pp_dynamic_shape

    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def dp_replicate_coord(self):
        if not self.dp_replicate_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="dp_replicate"), self.dp_replicate)

    @property
    def tp_coord(self):
        if not self.tp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="tp"), self.tp)

    @property
    def pp_coord(self):
        if not self.pp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="pp"), self.pp)

    @property
    def dp_shard_coord(self):
        if not self.dp_shard_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="dp_shard"), self.dp_shard)

    @property
    def cp_coord(self):
        if not self.cp_enabled:
            return 0, 1
        return (self.mesh.get_local_rank(mesh_dim="cp"), self.cp)

    @property
    def dp_shard_cp_coord(self):
        if not self.dp_shard_enabled and not self.cp_enabled:
            return 0, 1
        else:
            return self.mesh[tuple(("dp_shard_cp",))].get_local_rank(), self.mesh[tuple(("dp_shard_cp",))].size()

    def build_mesh_info(self):
        dims = [DimNames.PP, DimNames.DP_REPLICATE, DimNames.DP_SHARD, DimNames.CP, DimNames.TP]
        dim_paras = {
            DimNames.PP: self.pp,
            DimNames.DP_REPLICATE: self.dp_replicate,
            DimNames.DP_SHARD: self.dp_shard,
            DimNames.CP: self.cp,
            DimNames.TP: self.tp,
        }
        info = [{} for i in range(self.world_size)]
        meshes = [range(self.world_size)]
        for dim in dims:
            new_meshes = []
            for m in meshes:
                for r, arr in enumerate(numpy.array_split(m, dim_paras[dim])):
                    for d in list(arr):
                        if d in m:
                            info[d][dim] = r
                    new_meshes.append(list(arr))
            meshes = new_meshes
        # Note: full_rank_info will record the rank in each dimension for a global rank/device.
        # e.g: [{'pp': 0, 'dp_replicate': 0, 'dp_shard': 0, 'cp': 0, 'tp': 0, 'dp_shard_cp': 0, 'dp': 0},
        # {'pp': 0, 'dp_replicate': 0, 'dp_shard': 0, 'cp': 0, 'tp': 1, 'dp_shard_cp': 0, 'dp': 0}]
        self.full_rank_info = info
        self.full_world_size_info = dim_paras
        self.full_world_size_info[DimNames.DP_SHARD_CP] = self.dp_shard * self.cp
        self.full_world_size_info[DimNames.DP] = self.dp_replicate * self.dp_shard
        self.full_world_size_info[DimNames.DP_CP_TP] = self.dp_replicate * self.dp_shard * self.cp * self.tp

        for i in range(self.world_size):
            self.full_rank_info[i][DimNames.DP_CP_TP] = (
                self.full_rank_info[i][DimNames.DP_REPLICATE] * self.dp_shard * self.cp * self.tp
                + self.full_rank_info[i][DimNames.DP_SHARD] * self.cp * self.tp
                + self.full_rank_info[i][DimNames.CP] * self.tp
                + self.full_rank_info[i][DimNames.TP]
            )
            self.full_rank_info[i][DimNames.DP_SHARD_CP] = (
                self.full_rank_info[i][DimNames.DP_SHARD] * self.cp + self.full_rank_info[i][DimNames.CP]
            )
            self.full_rank_info[i][DimNames.DP] = (
                self.full_rank_info[i][DimNames.DP_REPLICATE] * self.dp_shard
                + self.full_rank_info[i][DimNames.DP_SHARD]
            )

        self.global_rank = int(os.environ.get("RANK", 0))
        logging.info(f"Full rank info: {self.full_rank_info}")
