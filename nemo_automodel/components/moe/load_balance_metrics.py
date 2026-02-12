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

"""MoE load balance metrics utilities.

Provides functions to enable load balance tracking on Gate modules,
collect per-layer expert load data, and compute brief/detailed metrics
suitable for wandb logging.

Expert utilization is a ratio of ``current_load / ideal_load`` where
``ideal_load = total_tokens / n_experts``.  A value of 1.0 means the
expert receives exactly its fair share; >1 = overloaded, <1 = underloaded,
0 = dead expert.

Modes:
- **brief**: Aggregated scalars (mean/median/min/max of cv and expert
  utilization) plus top-K/bottom-K individual expert utilization ratios.
- **detailed**: Everything in brief, plus per-layer breakdowns
  (``moe/layer_{i}/cv``, ``moe/layer_{i}/utilization_mean``, etc.).
"""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn


def enable_load_balance_tracking(model: nn.Module) -> None:
    """Enable load balance tracking on all Gate modules in the model.

    Sets ``_track_load_balance = True`` on every Gate instance found via
    ``model.modules()``.  This causes each Gate to store its most recent
    ``expert_load`` tensor after every forward pass with negligible overhead
    (one ``.detach()`` copy per layer).

    Args:
        model: The model (or model part) to enable tracking on.
    """
    from nemo_automodel.components.moe.layers import Gate

    for module in model.modules():
        if isinstance(module, Gate):
            module._track_load_balance = True


def collect_expert_loads(
    model: nn.Module,
    dp_group: torch.distributed.ProcessGroup | None = None,
) -> dict[str, dict]:
    """Collect the most recent expert load data from all Gate modules.

    When ``dp_group`` is provided, expert loads are all-reduced across the
    data-parallel group so the metrics reflect global token routing rather
    than a single rank's view.  This is important when DP > 1 or EP > 1
    because each rank only routes its local shard of tokens through the
    (replicated) gate.

    Args:
        model: The model (or model part) to collect from.
        dp_group: Optional DP (or DP+CP) process group for all-reducing
            expert loads.  Pass ``None`` to skip reduction (rank-local view).

    Returns:
        Dictionary mapping layer names to dicts with keys:
        - ``"expert_load"``: ``Tensor[n_experts]`` with token counts per expert.
        - ``"aux_loss"``: ``Optional[Tensor]`` scalar aux loss (if computed).
        - ``"n_experts"``: ``int`` number of routed experts.
    """
    from nemo_automodel.components.moe.layers import Gate

    loads: dict[str, dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, Gate) and module._last_expert_load is not None:
            expert_load = module._last_expert_load
            if dp_group is not None:
                expert_load = expert_load.clone()
                torch.distributed.all_reduce(expert_load, group=dp_group)
            loads[name] = {
                "expert_load": expert_load,
                "aux_loss": module._last_aux_loss,
                "n_experts": module.n_experts,
            }
    return loads


def _compute_per_layer_stats(layer_loads: dict[str, dict]):
    """Compute per-layer CV, aux_loss and per-layer utilization ratios.

    Returns:
        (per_layer_metrics, per_layer_utilizations) where:
        - per_layer_metrics: list of dicts with keys cv, aux_loss
        - per_layer_utilizations: list of Tensor[n_experts] with utilization ratio per layer
          (1.0 = ideal, >1 = overloaded, <1 = underloaded)
    """
    per_layer_metrics = []
    per_layer_utilizations = []

    for _name, data in sorted(layer_loads.items()):
        load = data["expert_load"].float()
        mean = load.mean()
        std = load.std()
        total = load.sum()

        cv = (std / mean).item() if mean > 0 else 0.0
        aux_loss = data["aux_loss"].item() if data["aux_loss"] is not None else None

        per_layer_metrics.append(
            {
                "cv": cv,
                "aux_loss": aux_loss,
            }
        )

        # Utilization ratio: current / ideal (1.0 = perfect balance)
        n_experts = load.shape[0]
        ideal = total / n_experts if n_experts > 0 else total
        utilization_ratio = (load / ideal) if ideal > 0 else torch.zeros_like(load)
        per_layer_utilizations.append(utilization_ratio)

    return per_layer_metrics, per_layer_utilizations


def _aggregate_stats(values: list[float], prefix: str) -> dict[str, float]:
    """Compute mean, median, min, max for a list of per-layer values.

    Args:
        values: List of per-layer scalar values.
        prefix: Key prefix, e.g. ``"moe/cv"``.

    Returns:
        Dict like ``{"moe/cv_mean": .., "moe/cv_median": .., "moe/cv_min": .., "moe/cv_max": ..}``.
    """
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _compute_expert_utilization(
    per_layer_utilizations: list[torch.Tensor],
    top_k: int = 5,
) -> dict[str, float]:
    """Compute top-K and bottom-K expert utilization ratios globally.

    Flattens utilization across all layers and experts, then emits only the
    ``top_k`` highest and ``top_k`` lowest entries.  This keeps the total
    number of wandb keys to at most ``2 * top_k`` regardless of model size.

    Values are ratios relative to ideal load: 1.0 = perfect balance,
    >1 = overloaded, <1 = underloaded, 0 = dead expert.

    All keys share the ``moe_expert_utilization/`` prefix so wandb
    renders them on a single chart.

    Args:
        per_layer_utilizations: List of Tensor[n_experts] utilization ratio per layer.
        top_k: Number of top (highest) and bottom (lowest) experts to emit.

    Returns:
        Dict like ``{"moe_expert_utilization/layer_0_expert_5": 1.23, ...}``.
    """
    if not per_layer_utilizations:
        return {}

    # Build flat list of (utilization, layer_idx, expert_idx)
    all_entries: list[tuple[float, int, int]] = []
    for layer_idx, util in enumerate(per_layer_utilizations):
        for expert_idx in range(util.shape[0]):
            all_entries.append((util[expert_idx].item(), layer_idx, expert_idx))

    all_entries.sort(key=lambda x: x[0])
    k = min(top_k, len(all_entries))

    metrics: dict[str, float] = {}
    # Bottom-K (lowest utilization)
    for val, layer_idx, expert_idx in all_entries[:k]:
        metrics[f"moe_expert_utilization/layer_{layer_idx}_expert_{expert_idx}"] = val
    # Top-K (highest utilization)
    for val, layer_idx, expert_idx in all_entries[-k:]:
        metrics[f"moe_expert_utilization/layer_{layer_idx}_expert_{expert_idx}"] = val

    return metrics


def _compute_utilization_aggregates(
    per_layer_utilizations: list[torch.Tensor],
    per_layer: bool = False,
) -> dict[str, float]:
    """Compute aggregate utilization stats across all experts globally.

    Args:
        per_layer_utilizations: List of Tensor[n_experts] utilization ratio per layer.
        per_layer: If True, also emit per-layer utilization means.

    Returns:
        Dict with ``moe/expert_utilization_{p25,median,p75,min,max}`` and optionally
        ``moe/layer_{i}/utilization_mean`` when ``per_layer=True``.
    """
    metrics: dict[str, float] = {}
    if not per_layer_utilizations:
        return metrics

    all_ratios: list[float] = []
    for layer_idx, util in enumerate(per_layer_utilizations):
        ratios = util.tolist()
        all_ratios.extend(ratios)
        if per_layer:
            metrics[f"moe/layer_{layer_idx}/utilization_mean"] = util.mean().item()

    # Global aggregates (skip mean — it's always 1.0 by construction)
    sorted_ratios = sorted(all_ratios)
    n = len(sorted_ratios)
    metrics["moe/expert_utilization_p25"] = sorted_ratios[n // 4] if n >= 4 else sorted_ratios[0]
    metrics["moe/expert_utilization_median"] = statistics.median(all_ratios)
    metrics["moe/expert_utilization_p75"] = sorted_ratios[3 * n // 4] if n >= 4 else sorted_ratios[-1]
    metrics["moe/expert_utilization_min"] = sorted_ratios[0]
    metrics["moe/expert_utilization_max"] = sorted_ratios[-1]
    return metrics


def compute_brief_metrics(
    layer_loads: dict[str, dict],
    top_k: int = 5,
) -> dict[str, float]:
    """Compute brief load-balance metrics: aggregated scalars + top-K/bottom-K utilization.

    Metrics produced:
    - ``moe/cv_{mean,median,min,max}`` — CV aggregated across all MoE layers.
    - ``moe/expert_utilization_{p25,median,p75,min,max}`` — utilization ratio stats
      across all experts globally (1.0 = ideal).
    - ``moe/aux_loss_mean`` — aux loss averaged across layers (when available).
    - ``moe_expert_utilization/layer_{i}_expert_{j}`` — top-K highest and bottom-K
      lowest utilization experts globally.

    Args:
        layer_loads: Output of :func:`collect_expert_loads`.
        top_k: Number of top/bottom experts to emit globally.

    Returns:
        Flat dictionary suitable for ``wandb.log()``.
    """
    per_layer, per_layer_utils = _compute_per_layer_stats(layer_loads)
    if not per_layer:
        return {}

    metrics: dict[str, float] = {}

    cvs = [m["cv"] for m in per_layer]
    aux_losses = [m["aux_loss"] for m in per_layer if m["aux_loss"] is not None]

    metrics.update(_aggregate_stats(cvs, "moe/cv"))
    if aux_losses:
        metrics["moe/aux_loss_mean"] = sum(aux_losses) / len(aux_losses)

    metrics.update(_compute_expert_utilization(per_layer_utils, top_k=top_k))
    metrics.update(_compute_utilization_aggregates(per_layer_utils, per_layer=False))
    return metrics


def compute_detailed_metrics(
    layer_loads: dict[str, dict],
    top_k: int = 5,
) -> dict[str, float]:
    """Compute detailed load-balance metrics: per-layer scalars + aggregates + utilization.

    Includes everything from :func:`compute_brief_metrics` plus per-layer
    breakdowns:
    - ``moe/layer_{i}/cv``, ``moe/layer_{i}/aux_loss``
    - ``moe/layer_{i}/utilization_mean`` — per-layer mean utilization.

    Args:
        layer_loads: Output of :func:`collect_expert_loads`.
        top_k: Number of top/bottom experts to emit globally.

    Returns:
        Flat dictionary suitable for ``wandb.log()``.
    """
    per_layer, per_layer_utils = _compute_per_layer_stats(layer_loads)
    if not per_layer:
        return {}

    metrics: dict[str, float] = {}

    cvs, aux_losses = [], []
    for i, m in enumerate(per_layer):
        metrics[f"moe/layer_{i}/cv"] = m["cv"]
        cvs.append(m["cv"])
        if m["aux_loss"] is not None:
            metrics[f"moe/layer_{i}/aux_loss"] = m["aux_loss"]
            aux_losses.append(m["aux_loss"])

    metrics.update(_aggregate_stats(cvs, "moe/cv"))
    if aux_losses:
        metrics["moe/aux_loss_mean"] = sum(aux_losses) / len(aux_losses)

    metrics.update(_compute_expert_utilization(per_layer_utils, top_k=top_k))
    metrics.update(_compute_utilization_aggregates(per_layer_utils, per_layer=True))
    return metrics
