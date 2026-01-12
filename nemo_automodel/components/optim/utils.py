from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Optional

import torch.nn as nn

_import_error: Exception | None = None
try:
    from dion import Dion, Dion2, Muon, NorMuon
except Exception as e:  # pragma: no cover - handled at runtime
    Dion = Dion2 = Muon = NorMuon = None
    _import_error = e

logger = logging.getLogger(__name__)


def is_dion_optimizer(cfg_opt) -> bool:
    target = getattr(cfg_opt, "_target_", None)
    name = getattr(target, "__name__", "")
    module = getattr(target, "__module__", "")
    return module.startswith("dion") or name in {"Dion", "Dion2", "Muon", "NorMuon"}


def _separate_param_groups(
    model: nn.Module,
    base_lr: float,
    scalar_opt: str,
    weight_decay: float,
    scalar_betas: tuple[float, float] | None = None,
    scalar_eps: float | None = None,
    scalar_lr: float | None = None,
    embed_lr: float | None = None,
    lm_head_lr: float | None = None,
):
    """
    Separate model parameters into groups for Dion/Muon optimizers.

    Args:
        model: The model to optimize.
        base_lr: Base learning rate for matrix params (Muon algorithm).
        scalar_opt: Optimizer algorithm for scalar params ("adamw" or "lion").
        weight_decay: Weight decay for vector params.
        scalar_betas: (beta1, beta2) for scalar optimizer.
        scalar_eps: Epsilon for scalar optimizer.
        scalar_lr: Learning rate for scalar (vector/bias) params. Defaults to base_lr.
        embed_lr: Learning rate for embedding params. Defaults to scalar_lr or base_lr.
        lm_head_lr: Learning rate for lm_head. Defaults to base_lr / sqrt(d_in).
    """
    matrix_params = []
    vector_params = []
    embed_params = []
    lm_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = None
        try:
            module_name = name.rsplit(".", 1)[0]
            module = dict(model.named_modules()).get(module_name, None)
        except Exception:
            module = None

        if isinstance(module, nn.Embedding):
            embed_params.append(param)
            continue

        if "lm_head" in name:
            lm_head_params.append(param)
            continue

        if param.ndim == 2:
            matrix_params.append(param)
        else:
            vector_params.append(param)

    scalar_kwargs = {}
    if scalar_betas is not None:
        scalar_kwargs["beta1"] = scalar_betas[0]
        scalar_kwargs["beta2"] = scalar_betas[1]
    if scalar_eps is not None:
        scalar_kwargs["epsilon"] = scalar_eps

    effective_scalar_lr = scalar_lr if scalar_lr is not None else base_lr
    effective_embed_lr = embed_lr if embed_lr is not None else effective_scalar_lr

    param_groups = [
        dict(params=matrix_params),
        dict(
            params=vector_params,
            algorithm=scalar_opt,
            lr=effective_scalar_lr,
            weight_decay=weight_decay,
            **scalar_kwargs,
        ),
        dict(params=embed_params, algorithm=scalar_opt, lr=effective_embed_lr, weight_decay=0.0, **scalar_kwargs),
    ]

    if lm_head_params:
        # Use explicit lm_head_lr or scale by sqrt(d_in) as recommended in Dion docs
        if lm_head_lr is not None:
            effective_lm_head_lr = lm_head_lr
        else:
            first = lm_head_params[0]
            d_in = first.shape[-1] if first.ndim >= 2 else max(1, first.numel())
            effective_lm_head_lr = base_lr / math.sqrt(float(d_in))
        param_groups.append(
            dict(
                params=lm_head_params, algorithm=scalar_opt, lr=effective_lm_head_lr, weight_decay=0.0, **scalar_kwargs
            )
        )

    return param_groups


def _get_dion_mesh(distributed_mesh: Any) -> Any:
    if distributed_mesh is None:
        return None
    if not hasattr(distributed_mesh, "ndim") or distributed_mesh.ndim == 1:
        return distributed_mesh
    try:
        logger.info(f"[Dion] Extracting dp_shard_cp 1D submesh from distributed_mesh: {distributed_mesh}")
        dp_mesh_2d = distributed_mesh[("dp_replicate", "dp_shard_cp")]
        submesh = dp_mesh_2d["dp_shard_cp"]
        if hasattr(submesh, "ndim") and submesh.ndim == 1:
            logger.info(f"[Dion] Extracted dp_shard_cp 1D submesh via 2D mesh: {submesh}")
            return submesh
    except (KeyError, RuntimeError, TypeError) as e:
        logger.debug(f"[Dion] Could not access via (dp_replicate, dp_shard_cp): {e}")
    return distributed_mesh


def build_dion_optimizer(
    cfg_opt,
    model: nn.Module,
    distributed_mesh: Optional[Any] = None,
) -> Any:
    """
    Build a Dion-family optimizer with parameter grouping.

    Args:
        cfg_opt: ConfigNode for the optimizer.
        model: Model whose parameters are to be optimized.
        distributed_mesh: Optional DeviceMesh for FSDP/TP.
        process_group: Optional ProcessGroup for DDP.
    """
    if _import_error:
        raise RuntimeError("Failed to import Dion. Please install Dion.") from _import_error

    target = cfg_opt._target_

    cfg_dict = cfg_opt.to_dict()
    scalar_opt = cfg_dict.pop("scalar_opt", "adamw")
    scalar_betas = tuple(cfg_dict.pop("scalar_betas", [])) or None
    scalar_eps = cfg_dict.pop("scalar_eps", None)
    scalar_lr = cfg_dict.pop("scalar_lr", None)
    embed_lr = cfg_dict.pop("embed_lr", None)
    lm_head_lr = cfg_dict.pop("lm_head_lr", None)

    base_lr = float(cfg_dict.get("lr", 1e-4))
    weight_decay = float(cfg_dict.get("weight_decay", 0.0))

    signature = inspect.signature(target)
    valid_keys = set(signature.parameters.keys())
    cleaned_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_keys}

    param_groups = _separate_param_groups(
        model,
        base_lr,
        scalar_opt,
        weight_decay,
        scalar_betas=scalar_betas,
        scalar_eps=scalar_eps,
        scalar_lr=scalar_lr,
        embed_lr=embed_lr,
        lm_head_lr=lm_head_lr,
    )

    dion_mesh = _get_dion_mesh(distributed_mesh)

    if "distributed_mesh" in valid_keys:
        cleaned_kwargs["distributed_mesh"] = dion_mesh
    elif "replicate_mesh" in valid_keys:
        cleaned_kwargs["replicate_mesh"] = dion_mesh
    if "adjust_lr" in cfg_dict:
        cleaned_kwargs["adjust_lr"] = cfg_dict["adjust_lr"]

    logger.info(f"[Dion] Building optimizer with {len(param_groups)} param groups:")
    for i, pg in enumerate(param_groups):
        algo = pg.get("algorithm", "dion2 (default)")
        n_params = len(pg["params"])
        n_elements = sum(p.numel() for p in pg["params"])
        lr_override = pg.get("lr", "default")
        logger.info(f"  Group {i}: algo={algo}, params={n_params}, elements={n_elements:,}, lr={lr_override}")
    logger.info(f"[Dion] Optimizer kwargs: {cleaned_kwargs}")

    return target(param_groups, **cleaned_kwargs)
