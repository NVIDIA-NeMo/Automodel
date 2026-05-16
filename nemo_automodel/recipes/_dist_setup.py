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
    # Use `val if val is not None` so that explicit YAML nulls (``ep_size:``
    # or ``ep_size: null``) fall back to the default instead of propagating
    # None — dict.pop only returns the default when the key is *absent*.
    parallelism = {
        k: (v if v is not None else default)
        for k, default in _PARALLELISM_DEFAULTS.items()
        for v in [cfg.pop(k, default)]
    }

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

    # Instantiate mp_policy from YAML dict for the strategy config.
    # Follows the same ``_target_`` pattern used for MoE mp_policy below.
    if "mp_policy" in strategy_kwargs:
        mp_raw = strategy_kwargs["mp_policy"]
        if isinstance(mp_raw, dict):
            mp_raw = mp_raw.copy()
            target = mp_raw.pop("_target_", None)
            for key in ("param_dtype", "reduce_dtype", "output_dtype"):
                if key in mp_raw and isinstance(mp_raw[key], str):
                    mp_raw[key] = dtype_from_str(mp_raw[key])
            if target is not None and callable(target):
                strategy_kwargs["mp_policy"] = target(**mp_raw)
            else:
                from torch.distributed.fsdp import MixedPrecisionPolicy

                strategy_kwargs["mp_policy"] = MixedPrecisionPolicy(**mp_raw)

    # Instantiate offload_policy from YAML dict (same ``_target_`` pattern).
    if "offload_policy" in strategy_kwargs:
        op_raw = strategy_kwargs["offload_policy"]
        if isinstance(op_raw, dict):
            op_raw = op_raw.copy()
            target = op_raw.pop("_target_", None)
            if target is not None:
                if isinstance(target, str):
                    # Resolve dotted path to class
                    import importlib

                    mod_path, cls_name = target.rsplit(".", 1)
                    target = getattr(importlib.import_module(mod_path), cls_name)
                strategy_kwargs["offload_policy"] = target(**op_raw)
            else:
                from torch.distributed.fsdp import CPUOffloadPolicy

                strategy_kwargs["offload_policy"] = CPUOffloadPolicy(**op_raw)

    # Convert autocast_dtype string to torch.dtype if present.
    if "autocast_dtype" in strategy_kwargs:
        val = strategy_kwargs["autocast_dtype"]
        if isinstance(val, str):
            strategy_kwargs["autocast_dtype"] = dtype_from_str(val)

    _validate_strategy_kwargs(strategy_name, strategy_cls, strategy_kwargs)

    # Route activation_checkpointing: for non-EP configs it goes on the
    # strategy config; for EP configs it stays only on MeshContext
    # (the MoE infra reads it from there).
    ep_size: int = parallelism.get("ep_size") or 1

    # YAML-level sanity: silently discard sub-configs that don't apply to the
    # current parallelism sizes (e.g. pipeline section present but pp_size=1,
    # which is common when a YAML template is overridden via CLI).
    pp_size: int = parallelism.get("pp_size") or 1
    if pipeline_dict is not None and pp_size <= 1:
        pipeline_dict = None
    if moe_dict is not None and ep_size <= 1:
        moe_dict = None
    if ep_size <= 1:
        strategy_kwargs["activation_checkpointing"] = activation_checkpointing

    strategy_config = strategy_cls(**strategy_kwargs)

    if pipeline_dict is not None:
        pipeline_config = PipelineConfig(**pipeline_dict)
    elif pp_size > 1:
        pipeline_config = PipelineConfig()
    else:
        pipeline_config = None

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


def _distributed_cfg_to_dict(cfg: Any | None) -> dict:
    """Return a distributed config dict from ``cfg`` or an empty fallback."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg.copy()
    distributed_cfg = cfg.distributed
    return distributed_cfg.to_dict() if hasattr(distributed_cfg, "to_dict") else dict(distributed_cfg)


def setup_distributed(
    cfg: Any | None = None,
    world_size: Optional[int] = None,
    *,
    strategy: str | None = None,
    dp_size: int | None = None,
    dp_replicate_size: int | None = None,
    tp_size: int | None = None,
    pp_size: int | None = None,
    cp_size: int | None = None,
    ep_size: int | None = None,
    activation_checkpointing: bool | None = None,
    pipeline: dict | None = None,
    moe: dict | None = None,
    **strategy_kwargs: Any,
) -> MeshContext:
    """Parse distributed settings and create device meshes.

    This is the main entry-point called by recipes.  It converts the
    config section or programmatic keyword arguments into a fully-initialised
    :class:`MeshContext` (including ``device_mesh`` and ``moe_mesh``).

    Args:
        cfg: Optional distributed config dict or top-level config with a
            ``distributed`` key. Used as fallback when explicit keyword
            arguments are omitted.
        world_size: Total number of processes in the job. If ``None`` (default),
            the value is auto-detected from ``torch.distributed`` if initialized,
            or from the ``WORLD_SIZE`` environment variable, falling back to ``1``.
        strategy: Distributed strategy name (``fsdp2``, ``megatron_fsdp``, or ``ddp``).
        dp_size: Data-parallel size. If ``None``, inferred by mesh creation.
        dp_replicate_size: HSDP replicate size for FSDP2.
        tp_size: Tensor-parallel size.
        pp_size: Pipeline-parallel size.
        cp_size: Context-parallel size.
        ep_size: Expert-parallel size.
        activation_checkpointing: Enable activation checkpointing.
        pipeline: Optional pipeline sub-config.
        moe: Optional MoE parallelizer sub-config.
        **strategy_kwargs: Additional strategy-specific options.

    Returns:
        A :class:`MeshContext` with device meshes attached.
    """
    from nemo_automodel.components.distributed import mesh_utils
    from nemo_automodel.components.distributed.init_utils import get_world_size_safe

    if world_size is None:
        world_size = get_world_size_safe()

    cfg_dict = _distributed_cfg_to_dict(cfg)

    explicit_overrides = {
        "strategy": strategy,
        "dp_size": dp_size,
        "dp_replicate_size": dp_replicate_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "cp_size": cp_size,
        "ep_size": ep_size,
        "activation_checkpointing": activation_checkpointing,
        "pipeline": pipeline,
        "moe": moe,
    }
    for key, value in explicit_overrides.items():
        if value is not None:
            cfg_dict[key] = value
    for key, value in strategy_kwargs.items():
        if value is not None:
            cfg_dict[key] = value

    parsed = parse_distributed_section(cfg_dict)

    device_mesh, moe_mesh = mesh_utils.create_device_mesh(
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
