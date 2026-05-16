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

"""Model + optimizer offload/onload between CPU and a CUDA device.

Used by ``Engine.to(device, ...)`` to free GPU memory during rollout phases
in RL pipelines, and to re-stage state back onto the device before training
resumes. Works for plain ``nn.Module`` and FSDP/FSDP2-wrapped modules; uses
the existing ``move_to_device`` helper that also moves buffers (which FSDP
modules don't move automatically).
"""

from __future__ import annotations

import gc
from typing import Any, Iterable

import torch
import torch.nn as nn

from nemo_automodel.components.training.utils import move_to_device


def _iter_parts(model: Any) -> Iterable[nn.Module]:
    """Yield model.parts if AutoPipeline, else (model,)."""
    if hasattr(model, "parts"):
        for part in model.parts:
            yield part
    else:
        yield model


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    """Move all optimizer state tensors (e.g. Adam's exp_avg, exp_avg_sq) to ``device``.

    Optimizer parameter tensors stay with the model; only the state tensors
    (running averages, step counters that are tensors, etc.) need moving.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def _zero_grads(model: Any, *, set_to_none: bool = True) -> None:
    """Drop gradients for every parameter so the grad buffer can be released."""
    for part in _iter_parts(model):
        for p in part.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()


def offload(
    model: Any,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    model_to_cpu: bool = True,
    optimizer_to_cpu: bool = True,
    drop_grad: bool = True,
) -> None:
    """Move model / optimizer state to CPU and (optionally) drop gradients.

    Args:
        model: ``nn.Module`` or ``AutoPipeline``.
        optimizer: optional optimizer whose state tensors should be moved.
        model_to_cpu: move model parameters and buffers to CPU.
        optimizer_to_cpu: move optimizer state tensors to CPU.
        drop_grad: set every parameter's ``.grad`` to None to release grad memory.
            Note: gradient buffers live alongside parameters, so dropping them
            requires ``model_to_cpu=True`` to actually free GPU memory.
    """
    if drop_grad:
        _zero_grads(model)

    if model_to_cpu:
        for part in _iter_parts(model):
            move_to_device(part, "cpu")

    if optimizer_to_cpu and optimizer is not None:
        _move_optimizer_state(optimizer, "cpu")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def onload(
    model: Any,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cuda",
    *,
    model_to_device: bool = True,
    optimizer_to_device: bool = True,
) -> None:
    """Move model / optimizer state back to ``device`` (default: ``"cuda"``).

    Args:
        model: ``nn.Module`` or ``AutoPipeline``.
        optimizer: optional optimizer whose state tensors should be moved.
        device: target device (``"cuda"``, ``"cuda:0"``, ...). Default ``"cuda"``.
        model_to_device: move model parameters and buffers to ``device``.
        optimizer_to_device: move optimizer state tensors to ``device``.
    """
    if model_to_device:
        for part in _iter_parts(model):
            move_to_device(part, device)

    if optimizer_to_device and optimizer is not None:
        _move_optimizer_state(optimizer, device)

    gc.collect()


__all__ = ["offload", "onload"]
