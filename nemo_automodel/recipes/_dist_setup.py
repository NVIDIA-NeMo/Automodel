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

"""Recipe-level helpers for parsing YAML distributed configs.

This module bridges the gap between raw YAML / :class:`ConfigNode` dicts
and the typed :class:`MeshContext` used by the component layer.
All dict handling lives here; the component layer (``mesh``) stays purely typed.
"""

import dataclasses
from typing import Any, Dict, Optional

from nemo_automodel.components.distributed.mesh import (
    STRATEGY_MAP,
    MeshContext,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig
from nemo_automodel.shared.utils import dtype_from_str

_PARALLELISM_DEFAULTS: Dict[str, Any] = {
    "tp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ep_size": 1,
    "dp_size": None,
    "dp_replicate_size": None,
}


def _validate_strategy_kwargs(
    strategy_name: str,
    strategy_cls: type,
    strategy_kwargs: Dict[str, Any],
) -> None:
    """Check that *strategy_kwargs* only contains fields recognised by *strategy_cls*."""
    valid_fields = {f.name for f in dataclasses.fields(strategy_cls)}
    unknown = set(strategy_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"Unknown options for strategy '{strategy_name}': {sorted(unknown)}")


def parse_distributed_section(cfg_dict: dict) -> dict:
    """Parse a flat distributed config dict into components for mesh creation.

    Returns a plain ``dict`` with:

    - ``strategy_config`` – instantiated strategy dataclass
    - ``pipeline_config`` – :class:`PipelineConfig` or ``None``
    - ``moe_config`` – :class:`MoEParallelizerConfig` or ``None``
    - ``activation_checkpointing`` – bool
    - ``tp_size``, ``pp_size``, ``cp_size``, ``ep_size``, ``dp_size``,
      ``dp_replicate_size`` – parallelism sizes
    - ``pp_enabled`` – ``True`` when ``pp_size > 1``

    Device meshes are **not** created here; that is done by
    :func:`setup_distributed`.
    """
    cfg = cfg_dict.copy()  # shallow copy — never mutate the caller's dict

    # -- strategy -----------------------------------------------------------
    strategy_name: str = cfg.pop("strategy", "fsdp2")
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid strategies: {list(STRATEGY_MAP.keys())}")
    strategy_cls = STRATEGY_MAP[strategy_name]

    # -- parallelism sizes --------------------------------------------------
    parallelism = {k: cfg.pop(k, default) for k, default in _PARALLELISM_DEFAULTS.items()}

    # -- sub-configs --------------------------------------------------------
    pipeline_dict: Optional[dict] = cfg.pop("pipeline", None)
    moe_dict: Optional[dict] = cfg.pop("moe", None)
    activation_checkpointing: bool = cfg.pop("activation_checkpointing", False)

    # Strip Hydra / OmegaConf meta keys (e.g. ``_target_``, ``_recursive_``,
    # ``_convert_``) that may leak from YAML configs.  They have no meaning
    # for the strategy constructor and should not trigger validation errors.
    _HYDRA_META_KEYS = {"_target_", "_recursive_", "_convert_"}
    for key in _HYDRA_META_KEYS:
        cfg.pop(key, None)

    # Everything still in *cfg* is forwarded to the strategy constructor.
    strategy_kwargs: Dict[str, Any] = cfg

    _validate_strategy_kwargs(strategy_name, strategy_cls, strategy_kwargs)

    # Route activation_checkpointing: for non-EP configs it goes on the
    # strategy config; for EP configs it stays only on MeshContext
    # (the MoE infra reads it from there).
    ep_size: int = parallelism.get("ep_size", 1)

    # YAML-level sanity: silently discard sub-configs that don't apply to the
    # current parallelism sizes (e.g. pipeline section present but pp_size=1,
    # which is common when a YAML template is overridden via CLI).
    pp_size: int = parallelism.get("pp_size", 1)
    if pipeline_dict is not None and pp_size <= 1:
        pipeline_dict = None
    if moe_dict is not None and ep_size <= 1:
        moe_dict = None
    if ep_size <= 1:
        strategy_kwargs["activation_checkpointing"] = activation_checkpointing

    strategy_config = strategy_cls(**strategy_kwargs)

    pipeline_config = PipelineConfig(**pipeline_dict) if pipeline_dict is not None else None

    # Instantiate nested _target_ configs (e.g. mp_policy) before constructing MoEParallelizerConfig
    if moe_dict is not None and "mp_policy" in moe_dict:
        mp_raw = moe_dict["mp_policy"]
        if isinstance(mp_raw, dict) and callable(mp_raw.get("_target_")):
            mp_raw = mp_raw.copy()
            target = mp_raw.pop("_target_")
            for key in ("param_dtype", "reduce_dtype", "output_dtype"):
                if key in mp_raw and isinstance(mp_raw[key], str):
                    mp_raw[key] = dtype_from_str(mp_raw[key])
            moe_dict["mp_policy"] = target(**mp_raw)

    moe_config = MoEParallelizerConfig(**(moe_dict or {})) if ep_size > 1 else None

    # Full cross-field validation is deferred to MeshContext.__post_init__
    # (called automatically when setup_distributed constructs the context).

    return {
        "strategy_config": strategy_config,
        "pipeline_config": pipeline_config,
        "moe_config": moe_config,
        "activation_checkpointing": activation_checkpointing,
        "pp_enabled": parallelism["pp_size"] > 1,
        **parallelism,
    }


def setup_distributed(cfg: Any, world_size: int) -> MeshContext:
    """Parse ``cfg.distributed`` and create device meshes.

    This is the main entry-point called by recipes.  It converts the
    config section into a fully-initialised :class:`MeshContext`
    (including ``device_mesh`` and ``moe_mesh``).

    Args:
        cfg: Top-level config (must have a ``distributed`` key).
        world_size: Total number of processes in the job.

    Returns:
        A :class:`MeshContext` with device meshes attached.
    """
    from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

    cfg_dict = cfg.distributed.to_dict() if not isinstance(cfg, dict) else cfg
    parsed = parse_distributed_section(cfg_dict)

    device_mesh, moe_mesh = create_device_mesh(
        parsed["strategy_config"],
        dp_size=parsed["dp_size"],
        dp_replicate_size=parsed["dp_replicate_size"],
        tp_size=parsed["tp_size"],
        pp_size=parsed["pp_size"],
        cp_size=parsed["cp_size"],
        ep_size=parsed["ep_size"],
        world_size=world_size,
    )

    return MeshContext(
        strategy_config=parsed["strategy_config"],
        pipeline_config=parsed["pipeline_config"],
        moe_config=parsed["moe_config"],
        activation_checkpointing=parsed["activation_checkpointing"],
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
    )
