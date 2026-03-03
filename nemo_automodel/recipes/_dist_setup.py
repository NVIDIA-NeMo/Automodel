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


def validate_num_gpus(
    *,
    world_size: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    dp_size: Optional[int],
    dp_replicate_size: Optional[int],
) -> None:
    """Validate that parallelism dimensions are compatible with the number of GPUs.

    This runs **before** device-mesh creation so that users see a single,
    actionable error instead of a cryptic ``init_device_mesh`` crash.

    Raises:
        ValueError: With a message that explains the mismatch and suggests
            concrete fixes.
    """
    tp = tp_size if tp_size and tp_size > 0 else 1
    pp = pp_size if pp_size and pp_size > 0 else 1
    cp = cp_size if cp_size and cp_size > 0 else 1
    ep = ep_size if ep_size and ep_size > 0 else 1

    if world_size <= 0:
        raise ValueError(
            f"num_gpus (world_size) must be a positive integer, got {world_size}.\n"
            f"  Set the WORLD_SIZE environment variable or use "
            f"torchrun --nproc_per_node to configure the number of GPUs."
        )

    for name, val in [("tp_size", tp_size), ("pp_size", pp_size), ("cp_size", cp_size), ("ep_size", ep_size)]:
        if val is not None and val < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {val}.")

    # tp * pp * cp is the minimum number of GPUs required (dp >= 1 is implicit)
    explicit_product = tp * pp * cp
    if explicit_product > world_size:
        raise ValueError(
            f"Not enough GPUs: tp_size * pp_size * cp_size = "
            f"{tp} * {pp} * {cp} = {explicit_product}, "
            f"but only {world_size} GPU(s) available.\n"
            f"  The minimum number of GPUs required is "
            f"tp_size * pp_size * cp_size = {explicit_product}.\n"
            f"  Either reduce your parallelism sizes or increase the number of GPUs."
        )

    if world_size % explicit_product != 0:
        # Suggest the nearest valid GPU counts
        lower = (world_size // explicit_product) * explicit_product
        upper = lower + explicit_product
        suggestions = [v for v in (lower, upper) if v > 0]
        raise ValueError(
            f"num_gpus ({world_size}) is not divisible by "
            f"tp_size * pp_size * cp_size = {tp} * {pp} * {cp} = {explicit_product}.\n"
            f"  data-parallel degree (dp_size) is computed as: "
            f"num_gpus / (tp_size * pp_size * cp_size), which must be a whole number.\n"
            f"  To fix, either:\n"
            f"    - change num_gpus to a multiple of {explicit_product} "
            f"(nearest valid: {', '.join(map(str, suggestions))}), or\n"
            f"    - adjust tp_size, pp_size, or cp_size so their product divides {world_size}."
        )

    inferred_dp = world_size // explicit_product

    # When dp_size is explicitly set, it must be consistent with world_size
    if dp_size is not None and dp_size > 0:
        expected_world = tp * pp * cp * dp_size
        if expected_world != world_size:
            raise ValueError(
                f"Parallelism dimensions do not match the number of GPUs.\n"
                f"  tp_size * pp_size * cp_size * dp_size = "
                f"{tp} * {pp} * {cp} * {dp_size} = {expected_world}, "
                f"but num_gpus = {world_size}.\n"
                f"  To fix, either:\n"
                f"    - remove dp_size (set to null) to auto-infer it as {inferred_dp}, or\n"
                f"    - change num_gpus to {expected_world}, or\n"
                f"    - adjust dp_size to {inferred_dp}."
            )
        inferred_dp = dp_size

    # HSDP: dp_replicate_size must evenly divide dp_size
    if dp_replicate_size is not None and dp_replicate_size > 1:
        if inferred_dp % dp_replicate_size != 0:
            valid_values = [i for i in range(2, inferred_dp + 1) if inferred_dp % i == 0 and i < inferred_dp]
            hint = f"valid dp_replicate_size values: {valid_values}" if valid_values else "increase dp_size first"
            raise ValueError(
                f"dp_replicate_size ({dp_replicate_size}) does not evenly divide "
                f"dp_size ({inferred_dp}).\n"
                f"  For HSDP, dp_size must be a multiple of dp_replicate_size.\n"
                f"  To fix: {hint}."
            )
        if dp_replicate_size >= inferred_dp:
            raise ValueError(
                f"dp_replicate_size ({dp_replicate_size}) must be strictly less than "
                f"dp_size ({inferred_dp}).\n"
                f"  Pure DDP replication is not supported with FSDP2; there must be "
                f"at least 2 sharding groups.\n"
                f"  To fix: reduce dp_replicate_size or increase the number of GPUs."
            )

    # EP: (dp_size * cp_size) must be divisible by ep_size
    if ep > 1:
        dp_cp = inferred_dp * cp
        if dp_cp < ep:
            raise ValueError(
                f"ep_size ({ep}) exceeds dp_size * cp_size = "
                f"{inferred_dp} * {cp} = {dp_cp}.\n"
                f"  Expert-parallel degree cannot exceed the data-parallel * "
                f"context-parallel degree.\n"
                f"  To fix: reduce ep_size to at most {dp_cp}, "
                f"or increase the number of GPUs."
            )
        if dp_cp % ep != 0:
            valid_ep = [i for i in range(2, dp_cp + 1) if dp_cp % i == 0]
            raise ValueError(
                f"(dp_size * cp_size) = {inferred_dp} * {cp} = {dp_cp} "
                f"is not divisible by ep_size ({ep}).\n"
                f"  ep_size must evenly divide (dp_size * cp_size).\n"
                f"  Valid ep_size values for this configuration: {valid_ep}."
            )


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

    validate_num_gpus(
        world_size=world_size,
        tp_size=parsed["tp_size"],
        pp_size=parsed["pp_size"],
        cp_size=parsed["cp_size"],
        ep_size=parsed["ep_size"],
        dp_size=parsed["dp_size"],
        dp_replicate_size=parsed["dp_replicate_size"],
    )

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
