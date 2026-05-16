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

"""Thin function wrappers around ``Checkpointer`` + ``state_dict_adapter``.

The wrappers exist so the Engine — and any caller that just wants
"save / load / export" without thinking about ranks, mesh dimensions, and
state-dict adapters — can call three functions instead of constructing a
``Checkpointer`` and computing ranks by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
)
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh


# ── Rank derivation from MeshContext ──────────────────────────────────


def _dp_rank(mesh: MeshContext | None, *, include_cp: bool = True) -> int:
    """DP rank, optionally flattened against CP. Matches BaseRecipe._get_dp_rank()."""
    if mesh is None or mesh.device_mesh is None:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    dm = mesh.device_mesh
    if include_cp and MeshAxisName.CP in dm.mesh_dim_names and dm[MeshAxisName.CP].size() > 1:
        return get_flat_mesh(dm, "dp_cp").get_local_rank()
    return get_flat_mesh(dm, "dp").get_local_rank()


def _tp_rank(mesh: MeshContext | None) -> int:
    if mesh is None or mesh.device_mesh is None:
        return 0
    dm = mesh.device_mesh
    if MeshAxisName.TP not in dm.mesh_dim_names or dm[MeshAxisName.TP].size() == 1:
        return 0
    return dm.get_local_rank(MeshAxisName.TP)


def _pp_rank(mesh: MeshContext | None) -> int:
    if mesh is None or mesh.device_mesh is None:
        return 0
    dm = mesh.device_mesh
    if MeshAxisName.PP not in dm.mesh_dim_names or dm[MeshAxisName.PP].size() == 1:
        return 0
    return dm.get_local_rank(MeshAxisName.PP)


# ── Checkpointer construction ─────────────────────────────────────────


def make_checkpointer(
    *,
    config: CheckpointingConfig,
    mesh: MeshContext | None,
) -> Checkpointer:
    """Build a :class:`Checkpointer` with ranks derived from ``mesh``."""
    return Checkpointer(
        config=config,
        dp_rank=_dp_rank(mesh, include_cp=True),
        tp_rank=_tp_rank(mesh),
        pp_rank=_pp_rank(mesh),
        moe_mesh=mesh.moe_mesh if mesh is not None else None,
    )


# ── Public API ────────────────────────────────────────────────────────


def save_checkpoint(
    model: nn.Module | Any,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any | None,
    *,
    mesh: MeshContext | None,
    path: str | Path,
    config: CheckpointingConfig,
    peft_config: Any | None = None,
    tokenizer: Any | None = None,
) -> None:
    """Save model (+ optional optimizer/scheduler) to ``path``.

    ``path`` is the base checkpoint directory. The model is written to
    ``{path}/model`` and the optimizer to ``{path}/optim``.
    """
    ckpt = make_checkpointer(config=config, mesh=mesh)
    ckpt.save_model(model, str(path), peft_config=peft_config, tokenizer=tokenizer)
    if optimizer is not None:
        scheduler_list = [lr_scheduler] if lr_scheduler is not None else None
        ckpt.save_optimizer(optimizer, model, str(path), scheduler=scheduler_list)


def load_checkpoint(
    model: nn.Module | Any,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: Any | None,
    *,
    mesh: MeshContext | None,
    path: str | Path,
    config: CheckpointingConfig,
) -> None:
    """Load model (+ optional optimizer/scheduler) from ``path``."""
    ckpt = make_checkpointer(config=config, mesh=mesh)
    ckpt.load_model(model, str(path))
    if optimizer is not None:
        scheduler_list = [lr_scheduler] if lr_scheduler is not None else None
        ckpt.load_optimizer(optimizer, model, str(path), scheduler=scheduler_list)


def export_weights(
    model: nn.Module | Any,
    *,
    to_hf: bool = True,
    mesh: MeshContext | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate ``(name, tensor)`` pairs of the model's parameters.

    Materializes any :class:`DTensor` to a full tensor via ``.full_tensor()`` so
    consumers (RL refit, eval) see plain tensors. If the model exposes a
    ``state_dict_adapter`` and ``to_hf=True``, parameter names are converted
    to the HuggingFace format.

    Args:
        model: ``nn.Module`` or ``AutoPipeline``.
        to_hf: when ``True`` and a ``state_dict_adapter`` is present, apply
            its ``to_hf`` conversion to the keys.
        mesh: optional :class:`MeshContext` (passed to the adapter when present).
    """
    parts = list(getattr(model, "parts", [model]))

    for part in parts:
        sd = part.state_dict()

        if to_hf:
            adapter = getattr(part, "state_dict_adapter", None)
            if adapter is not None:
                sd = adapter.to_hf(
                    sd,
                    exclude_key_regex=r".*_extra_state.*",
                    quantization=False,
                    device_mesh=mesh.moe_mesh if mesh is not None else None,
                )

        for name, tensor in sd.items():
            if isinstance(tensor, DTensor):
                yield name, tensor.full_tensor()
            else:
                yield name, tensor


__all__ = ["save_checkpoint", "load_checkpoint", "export_weights", "make_checkpointer"]
