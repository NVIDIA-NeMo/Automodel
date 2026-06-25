# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Opt-in CUDA memory profiling helpers."""

from __future__ import annotations

import gc
import logging
import os
from collections import defaultdict
from typing import Any

import torch
from torch.distributed.tensor import DTensor

_TRUE_VALUES = {"1", "true", "yes", "on"}


def memory_profile_enabled() -> bool:
    """Return whether CUDA memory phase logging is enabled."""
    return os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE", "").lower() in _TRUE_VALUES


def grad_profile_enabled() -> bool:
    """Return whether gradient norm profiling is enabled."""
    return os.environ.get("NEMO_AUTOMODEL_GRAD_PROFILE", "").lower() in _TRUE_VALUES


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_VALUES


def _current_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _rank_is_enabled(rank: int) -> bool:
    ranks = os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE_RANKS", "0").strip().lower()
    if ranks == "all":
        return True
    try:
        return rank in {int(item.strip()) for item in ranks.split(",") if item.strip()}
    except ValueError:
        return rank == 0


def _gib(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024**3


def _grad_category(name: str) -> str:
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_lm_head"
    if ".self_attn.q_proj" in name:
        return "attn_q_proj"
    if ".self_attn.k_proj" in name:
        return "attn_k_proj"
    if ".self_attn.v_proj" in name:
        return "attn_v_proj"
    if ".self_attn.o_proj" in name:
        return "attn_o_proj"
    if ".self_attn." in name:
        return "attn_other"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "norm"
    return "other"


def _tensor_census(limit: int) -> list[str]:
    seen_storages: set[tuple[str, int]] = set()
    groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(lambda: {"objects": 0, "storages": 0, "bytes": 0})
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                continue
            storage = obj.untyped_storage()
            storage_key = (str(obj.device), storage.data_ptr())
            storage_bytes = storage.nbytes()
            group_key = (
                tuple(obj.shape),
                str(obj.dtype).replace("torch.", ""),
                bool(obj.requires_grad),
                bool(obj.is_leaf),
                type(obj.grad_fn).__name__ if obj.grad_fn is not None else "None",
            )
            groups[group_key]["objects"] += 1
            if storage_key not in seen_storages:
                seen_storages.add(storage_key)
                groups[group_key]["storages"] += 1
                groups[group_key]["bytes"] += storage_bytes
        except Exception:
            continue

    rows = []
    for (shape, dtype, requires_grad, is_leaf, grad_fn), stats in groups.items():
        rows.append(
            (
                stats["bytes"],
                "shape=%s dtype=%s req_grad=%s leaf=%s grad_fn=%s objects=%d storages=%d bytes=%.3fGiB"
                % (
                    shape,
                    dtype,
                    requires_grad,
                    is_leaf,
                    grad_fn,
                    stats["objects"],
                    stats["storages"],
                    _gib(stats["bytes"]),
                ),
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows[:limit]]


def log_cuda_memory_profile(
    tag: str,
    *,
    logger: logging.Logger | None = None,
    include_tensors: bool = False,
    reset_peak: bool = False,
) -> None:
    """Log CUDA allocator state when memory profiling is enabled.

    Environment controls:
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE=1`` enables logging.
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE_RANKS=0,1`` selects ranks; ``all`` logs every rank.
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE_SYNC=1`` synchronizes before reading allocator state.
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE_TENSORS=1`` logs largest live CUDA tensor groups.
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE_TENSOR_LIMIT=12`` controls tensor census length.
    """
    if not memory_profile_enabled() or not torch.cuda.is_available():
        return

    rank = _current_rank()
    if not _rank_is_enabled(rank):
        return

    if _env_flag("NEMO_AUTOMODEL_MEMORY_PROFILE_SYNC"):
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    active = stats.get("active_bytes.all.current", 0)
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(
        "[cuda-memory] rank=%s device=%s tag=%s allocated=%.3fGiB reserved=%.3fGiB "
        "max_allocated=%.3fGiB max_reserved=%.3fGiB active=%.3fGiB inactive_split=%.3fGiB "
        "free=%.3fGiB total=%.3fGiB",
        rank,
        device,
        tag,
        _gib(allocated),
        _gib(reserved),
        _gib(max_allocated),
        _gib(max_reserved),
        _gib(active),
        _gib(inactive_split),
        _gib(free_bytes),
        _gib(total_bytes),
    )

    if include_tensors or _env_flag("NEMO_AUTOMODEL_MEMORY_PROFILE_TENSORS"):
        limit = int(os.environ.get("NEMO_AUTOMODEL_MEMORY_PROFILE_TENSOR_LIMIT", "12"))
        for row in _tensor_census(limit):
            logger.info("[cuda-memory-tensor] rank=%s tag=%s %s", rank, tag, row)

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def log_cuda_grad_profile(
    model_parts: list[torch.nn.Module],
    tag: str,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log local gradient norm hot spots when gradient profiling is enabled.

    Environment controls:
      - ``NEMO_AUTOMODEL_GRAD_PROFILE=1`` enables logging.
      - ``NEMO_AUTOMODEL_MEMORY_PROFILE_RANKS=0,1`` selects ranks; ``all`` logs every rank.
      - ``NEMO_AUTOMODEL_GRAD_PROFILE_LIMIT=20`` controls top-parameter rows.
    """
    if not grad_profile_enabled() or not torch.cuda.is_available():
        return

    rank = _current_rank()
    if not _rank_is_enabled(rank):
        return

    if logger is None:
        logger = logging.getLogger(__name__)

    category_sums: dict[str, torch.Tensor] = {}
    rows = []
    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    for part_idx, model_part in enumerate(model_parts):
        for name, param in model_part.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            placements = "regular"
            if isinstance(grad, DTensor):
                placements = ",".join(str(placement) for placement in grad.placements)
                grad_local = grad.to_local()
            else:
                grad_local = grad
            grad_local = grad_local.detach().float()
            local_sq = grad_local.square().sum().to(target_device)
            category = _grad_category(name)
            category_sums[category] = category_sums.get(category, torch.zeros((), device=target_device)) + local_sq
            rows.append(
                (
                    float(local_sq.sqrt().item()),
                    part_idx,
                    name,
                    tuple(grad_local.shape),
                    placements,
                )
            )

    total_sq = sum(category_sums.values(), torch.zeros((), device=target_device))
    total_norm = float(total_sq.sqrt().item())
    logger.info("[cuda-grad] rank=%s tag=%s local_total_norm=%.6f", rank, tag, float(total_sq.sqrt().item()))
    if _env_flag("NEMO_AUTOMODEL_GRAD_PROFILE_PRINT"):
        print(f"[cuda-grad] rank={rank} tag={tag} local_total_norm={total_norm:.6f}", flush=True)
    for category, local_sq in sorted(category_sums.items(), key=lambda item: float(item[1].item()), reverse=True):
        category_norm = float(local_sq.sqrt().item())
        logger.info(
            "[cuda-grad-category] rank=%s tag=%s category=%s local_norm=%.6f",
            rank,
            tag,
            category,
            category_norm,
        )
        if _env_flag("NEMO_AUTOMODEL_GRAD_PROFILE_PRINT"):
            print(
                f"[cuda-grad-category] rank={rank} tag={tag} category={category} local_norm={category_norm:.6f}",
                flush=True,
            )

    limit = int(os.environ.get("NEMO_AUTOMODEL_GRAD_PROFILE_LIMIT", "20"))
    for norm, part_idx, name, shape, placements in sorted(rows, reverse=True)[:limit]:
        logger.info(
            "[cuda-grad-param] rank=%s tag=%s local_norm=%.6f part=%s placements=%s shape=%s name=%s",
            rank,
            tag,
            norm,
            part_idx,
            placements,
            shape,
            name,
        )
