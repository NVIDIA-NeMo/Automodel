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
and the typed :class:`DistributedSetup` used by the component layer.
All dict handling lives here; the component layer stays typed. This module does
not initialize ``torch.distributed``. Recipes call ``initialize_distributed``
first, then pass the resulting world size here.
"""

import logging
from typing import Any, Dict, Optional

from nemo_automodel.components.distributed.config import (
    DistributedSetup,
    MoEParallelizerConfig,
    _resolve_strategy_config,
)
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

_PARALLELISM_DEFAULTS: Dict[str, Any] = {
    "tp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ep_size": 1,
    "dp_size": None,
    "dp_replicate_size": None,
}


def _normalize_activation_checkpointing(value: Any) -> bool | str:
    """Normalize YAML activation checkpointing values.

    ``True`` keeps the existing full checkpointing behavior. ``"selective"``
    enables PyTorch selective activation checkpointing for supported FSDP2
    paths.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower().replace("-", "_")
        if normalized in {"false", "off", "none", "disabled", "no"}:
            return False
        if normalized in {"true", "on", "full", "enabled", "yes"}:
            return True
        if normalized == "selective":
            return "selective"
    raise ValueError("distributed.activation_checkpointing must be a boolean or one of 'full', 'selective', 'false'.")


def parse_distributed_section(cfg_dict: dict) -> dict:
    """Parse a flat distributed config dict into components for mesh creation.

    Returns a plain ``dict`` with:

    - ``strategy_config`` – instantiated strategy dataclass
    - ``pipeline_config`` – :class:`PipelineConfig` or ``None``
    - ``moe_parallel_config`` – :class:`MoEParallelizerConfig` or ``None``
    - ``activation_checkpointing`` – bool
    - ``parallelism_sizes`` – :class:`ParallelismSizes`
    - ``tp_size``, ``pp_size``, ``cp_size``, ``ep_size``, ``dp_size``,
      ``dp_replicate_size`` – parallelism sizes
    - ``pp_enabled`` – ``True`` when ``pp_size > 1``

    Device meshes are **not** created here; that is done by
    :meth:`DistributedSetup.build`.
    """
    cfg = cfg_dict.copy()  # shallow copy — never mutate the caller's dict

    # -- strategy -----------------------------------------------------------
    strategy_name: str = cfg.pop("strategy", "fsdp2").lower()

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
    activation_checkpointing = _normalize_activation_checkpointing(cfg.pop("activation_checkpointing", False))

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

    ep_size: int = parallelism.get("ep_size") or 1
    if activation_checkpointing == "selective" and strategy_name != "fsdp2":
        raise ValueError("selective activation checkpointing is supported only for FSDP2 configs.")

    # `distributed.pipeline` and `pp_size` are validated asymmetrically:
    #   * pipeline block with pp_size<=1 -> WARN (inert; block ignored). This is
    #     often intentional -- a recipe keeps the pipeline section as a reference
    #     and toggles pp_size via CLI (e.g. ep8/pp1 vs ep4/pp2 parity debugging),
    #     and tests load such configs with `--distributed.pp_size 1`. A hard error
    #     can't be fixed by a static YAML edit when pp_size is overridden at launch.
    #   * pp_size>1 with no pipeline block -> ERROR. This is an unambiguous
    #     misconfiguration: PP is requested but the schedule/microbatch size are
    #     unspecified. (An explicit empty `pipeline: {}` opts into defaults.)
    pp_size: int = parallelism.get("pp_size") or 1
    if pipeline_dict is not None and pp_size <= 1:
        logger.warning(
            "`distributed.pipeline` is set but pp_size=%d (<= 1): pipeline parallelism "
            "is disabled and the pipeline settings are ignored. Set pp_size > 1 to enable it.",
            pp_size,
        )
        pipeline_dict = None
    if moe_dict is not None and ep_size <= 1:
        moe_dict = None

    strategy_config = _resolve_strategy_config(strategy_name, **strategy_kwargs)

    if pipeline_dict is not None:
        pipeline_dict = pipeline_dict.copy()
        if isinstance(pipeline_dict.get("dtype"), str):
            pipeline_dict["dtype"] = dtype_from_str(pipeline_dict["dtype"])
        pipeline_config = PipelineConfig(**pipeline_dict)
    elif pp_size > 1:
        raise ValueError(
            f"`pp_size={pp_size}` (> 1) enables pipeline parallelism but no "
            "`distributed.pipeline` block was provided. Add a `distributed.pipeline` "
            "section (e.g. `pp_schedule`, `pp_microbatch_size`); an empty "
            "`pipeline: {}` selects defaults."
        )
    else:
        pipeline_config = None

    # Default the pipeline communication dtype to the FSDP mixed-precision activation
    # dtype (the dtype of tensors crossing pipeline stage boundaries) so PP stage
    # shape inference matches the real activation dtype (e.g. bf16 compute under fp32
    # master weights). Deriving it from mp_policy.output_dtype is silent and correct;
    # an explicit mismatch is honored but warned, since it can corrupt inter-stage
    # recv buffers.
    if pipeline_config is not None and pp_size > 1:
        mp_policy = getattr(strategy_config, "mp_policy", None)
        activation_dtype = None
        if mp_policy is not None:
            activation_dtype = getattr(mp_policy, "output_dtype", None) or getattr(mp_policy, "param_dtype", None)
        if activation_dtype is not None:
            if pipeline_config.dtype is None:
                pipeline_config.dtype = activation_dtype
            elif pipeline_config.dtype != activation_dtype:
                logger.warning(
                    "pipeline.dtype=%s does not match the FSDP activation dtype "
                    "(mp_policy.output_dtype=%s) used for inter-stage communication; "
                    "this can corrupt pipeline stage shape inference. Leave pipeline.dtype "
                    "unset to derive it automatically.",
                    pipeline_config.dtype,
                    activation_dtype,
                )

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

    moe_parallel_config = MoEParallelizerConfig(**(moe_dict or {})) if ep_size > 1 else None

    return {
        "strategy_config": strategy_config,
        "pipeline_config": pipeline_config,
        "moe_parallel_config": moe_parallel_config,
        "activation_checkpointing": activation_checkpointing,
        "parallelism_sizes": ParallelismSizes(**parallelism),
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


def create_distributed_setup_from_config(
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
    pipeline: dict | None = None,
    moe: dict | None = None,
    **strategy_kwargs: Any,
) -> DistributedSetup:
    """Parse recipe distributed settings and create a distributed setup.

    This is the recipe-level adapter around :meth:`DistributedSetup.build`.
    It converts a YAML/config section or programmatic keyword arguments into a
    fully initialized :class:`DistributedSetup` (including ``device_mesh`` and
    ``moe_mesh`` through ``setup.mesh_context``). It does not initialize the
    process group; call ``initialize_distributed`` before this in distributed
    recipes.

    Args:
        cfg: Optional distributed config dict or top-level config with a
            ``distributed`` key. Used as fallback when explicit keyword
            arguments are omitted.
        world_size: Total number of processes in the job. If ``None`` (default),
            the value is auto-detected from ``torch.distributed`` if initialized,
            or from the ``WORLD_SIZE`` environment variable, falling back to ``1``.
        strategy: Distributed strategy name (``fsdp2``, ``megatron_fsdp``,
            ``megatron-fsdp``, ``mfsdp``, or ``ddp``).
        dp_size: Data-parallel size. If ``None``, inferred by mesh creation.
        dp_replicate_size: HSDP replicate size for FSDP2.
        tp_size: Tensor-parallel size.
        pp_size: Pipeline-parallel size.
        cp_size: Context-parallel size.
        ep_size: Expert-parallel size.
        pipeline: Optional pipeline sub-config.
        moe: Optional MoE parallelizer sub-config.
        **strategy_kwargs: Additional strategy-specific options.

    Returns:
        A :class:`DistributedSetup` with device meshes and policy configs attached.
    """
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
    return DistributedSetup.build(
        strategy=parsed["strategy_config"],
        parallelism_sizes=parsed["parallelism_sizes"],
        pipeline_config=parsed["pipeline_config"],
        moe_parallel_config=parsed["moe_parallel_config"],
        activation_checkpointing=parsed["activation_checkpointing"],
        world_size=world_size,
    )


def shard_optimizers_for_megatron_fsdp(model, optimizers, distributed_config, *, allow=True):
    """Apply Megatron-FSDP optimizer sharding per model part at the recipe layer.

    Kept here (not in components/optim) so the optim component does not import the
    distributed component. ``allow`` comes from the optimizer config's
    ``supports_megatron_fsdp_sharding`` flag: ``allow=False`` (e.g. Dion) makes
    ``maybe_shard_optimizer`` assert rather than silently skip under Megatron-FSDP.
    No-op unless ``distributed_config`` is a MegatronFSDPConfig running distributed.
    """
    from nemo_automodel.components.distributed.megatron_fsdp import maybe_shard_optimizer

    parts = list(getattr(model, "parts", [model]))
    return [maybe_shard_optimizer(part, opt, distributed_config, allow=allow) for part, opt in zip(parts, optimizers)]
