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

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

from nemo_automodel.components.distributed.init_utils import (
    get_world_size_safe,
    initialize_distributed,
)
from nemo_automodel.components.distributed.mesh import MeshContext

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.init_utils import DistInfo


def build_distributed(cfg_dist: dict[str, Any]) -> "DistInfo":
    """Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def _to_dict(cfg: Any) -> dict:
    """Coerce a config object (ConfigNode / dataclass / dict) to a plain dict."""
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    raise TypeError(f"Cannot coerce {type(cfg).__name__} to dict. Pass a dict, ConfigNode, or dataclass.")


def init_distributed_and_build_mesh(
    distributed_cfg: Any,
    *,
    dist_env_cfg: Mapping[str, Any] | None = None,
) -> tuple["DistInfo", MeshContext]:
    """Initialize torch.distributed and build a :class:`MeshContext`.

    Combines :func:`build_distributed` and
    ``nemo_automodel.recipes._dist_setup.setup_distributed`` into a single
    call so external callers (e.g. the :class:`Engine`) don't have to wire
    the two stages by hand.

    Args:
        distributed_cfg: Flat distributed config — dict, ConfigNode, or dataclass.
            Keys consumed by ``parse_distributed_section``:
            ``strategy`` (e.g. ``"fsdp2"``), ``tp_size``, ``pp_size``, ``cp_size``,
            ``ep_size``, ``dp_size``, ``dp_replicate_size``, ``pipeline`` (dict),
            ``moe`` (dict), ``activation_checkpointing``, plus strategy-specific
            kwargs forwarded to ``FSDP2Config`` / ``MegatronFSDPConfig`` / ``DDPConfig``.
        dist_env_cfg: Optional environment config — ``backend`` (default ``"nccl"``),
            ``timeout_minutes`` (default ``1``). If torch.distributed is already
            initialized, this is ignored.

    Returns:
        ``(DistInfo, MeshContext)``.
    """
    # Lazy import to avoid pulling _dist_setup at module load.
    from nemo_automodel.recipes._dist_setup import setup_distributed

    dist_env = dict(dist_env_cfg or {})
    dist_info = build_distributed(
        {"backend": dist_env.get("backend", "nccl"), "timeout_minutes": dist_env.get("timeout_minutes", 1)}
    )

    cfg_dict = _to_dict(distributed_cfg)
    mesh = setup_distributed(cfg_dict, world_size=get_world_size_safe())

    return dist_info, mesh


__all__ = ["build_distributed", "init_distributed_and_build_mesh"]
