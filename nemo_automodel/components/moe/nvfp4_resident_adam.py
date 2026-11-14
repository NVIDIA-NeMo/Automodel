# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Adam step of resident NVFP4 expert weights in two kernels.

Transformer Engine ``FusedAdam`` steps an NVFP4 weight in three passes over its FP32 master weight: it casts the
BF16 gradient to FP32, runs Adam, and quantizes the master back into the weight. ``enable_nvfp4_fused_adam_step``
replaces these for the weights of ``GroupedExpertsTENVFP4`` with one kernel that runs Adam on the BF16 gradient
and records the largest magnitude of each updated master, and one kernel that writes the row-wise and column-wise
NVFP4 operands of every weight from its master. Master weights, moments, codes, block scales and outer scales
come out bit-identical to the ``FusedAdam`` step; the optimizer state stays where ``FusedAdam`` keeps it, so
checkpoints and schedulers are unaffected.

The kernels are built once per build directory with ``torch.utils.cpp_extension.load_inline`` (CUDA toolkit
required) under ``$NEMO_NVFP4_ADAM_BUILD_DIR`` or ``~/.cache/nemo_automodel/nvfp4_adam``.
"""

import contextlib
import hashlib
import importlib.util
import logging
import os
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe import nvfp4_adam_cuda as sources
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsTENVFP4

if TYPE_CHECKING:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

logger = logging.getLogger(__name__)

# An argument of ``Optimizer.step`` as step hooks receive it: the optimizer itself first, then the loss closure
# and the gradient scaler.
_StepArgument: TypeAlias = torch.optim.Optimizer | Callable[[], torch.Tensor] | torch.amp.GradScaler | None

# Building both extensions takes well under a minute; an older build lock cannot belong to a live build.
_STALE_LOCK_SECONDS = 600.0
_STATE_NAMES = ("master_param", "exp_avg", "exp_avg_sq")

_extension: ModuleType | None = None
_extension_lock = threading.Lock()
_EXTENSION_NAME = "nemo_nvfp4_adam"


def _build_directory() -> str:
    """Build directory of this source text, torch version and device architecture."""
    root = os.environ.get("NEMO_NVFP4_ADAM_BUILD_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "nemo_automodel", "nvfp4_adam"
    )
    digest = hashlib.sha1()
    for source in (sources.HEADER, sources.KERNEL, sources.BINDING):
        digest.update(source.encode())
    digest.update(torch.__version__.encode())
    major, minor = torch.cuda.get_device_capability()
    digest.update(f"sm_{major}{minor}".encode())
    directory = os.path.join(root, digest.hexdigest()[:12])
    os.makedirs(directory, exist_ok=True)
    return directory


def _load_extension() -> ModuleType:
    """Import the extension from its built library, building it on first use.

    ``load_inline`` takes a file lock in the build directory on every call, and a process killed while holding
    it leaves a lock that blocks every later caller. A finished library is therefore imported directly, and a
    lock older than any build is removed.
    """
    global _extension
    with _extension_lock:
        if _extension is not None:
            return _extension
        directory = _build_directory()
        library = os.path.join(directory, f"{_EXTENSION_NAME}.so")
        lock = os.path.join(directory, "lock")
        with contextlib.suppress(FileNotFoundError):
            if time.time() - os.path.getmtime(lock) > _STALE_LOCK_SECONDS:
                logger.warning("Removing the stale build lock %s of an interrupted build.", lock)
                os.remove(lock)
        if os.path.isfile(library) and not os.path.isfile(lock):
            spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, library)
            if spec is not None and spec.loader is not None:
                _extension = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_extension)
                return _extension
        from torch.utils.cpp_extension import load_inline

        major, minor = torch.cuda.get_device_capability()
        _extension = load_inline(
            name=_EXTENSION_NAME,
            # The binding defines its own PYBIND11_MODULE, so no ``functions`` list.
            cpp_sources=[sources.HEADER, sources.BINDING],
            cuda_sources=[sources.HEADER, sources.KERNEL],
            # One architecture, that of the current device; no fused multiply-adds beyond the ones written out.
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "-gencode",
                f"arch=compute_{major}{minor},code=sm_{major}{minor}",
            ],
            build_directory=directory,
            verbose=False,
        )
        return _extension


def _kernels_cover(weight: "NVFP4Tensor") -> bool:
    """Whether ``weight`` is quantized the way the requantization kernel does: 2D blocks, nearest, both operands."""
    quantizer = weight._quantizer
    return (
        quantizer.with_2d_quantization
        and quantizer.rowwise_usage
        and quantizer.columnwise_usage
        and not quantizer.with_rht
        and not quantizer.stochastic_rounding
        and not quantizer.with_amax_reduction
        and not quantizer.row_scaled_nvfp4
        and not quantizer.nvfp4_use_4over6
        and not weight._with_gemm_swizzled_scales
        and weight.amax_reduction_group is None
    )


class _FusedAdamStep:
    """Step hooks of one ``FusedAdam`` that run the Adam step of some of its NVFP4 weights in the two kernels.

    The pre-step hook takes the gradients of those weights away, so ``FusedAdam.step`` advances its step counter
    and updates every other parameter but skips them; the post-step hook puts the gradients back and steps the
    weights with the group's hyperparameters. A weight keeps the ``FusedAdam`` step while it has no optimizer
    state yet (its first step), when its state is not FP32 tensors, or when its gradient is not BF16; so does
    every weight of a step that is given a closure or a gradient scaler.
    """

    def __init__(self, weights: list["NVFP4Tensor"], extension: ModuleType) -> None:
        """Create the hooks.

        Args:
            weights: NVFP4 weights to step in the kernels.
            extension: Built extension with ``adam_requantize``.
        """
        self._weight_ids = {id(weight) for weight in weights}
        self._extension = extension
        # (parameter group index, weight, gradient) of the weights the running step leaves to the kernels.
        self._held: list[tuple[int, "NVFP4Tensor", torch.Tensor]] = []

    def hold_gradients(
        self, optimizer: torch.optim.Optimizer, args: tuple[_StepArgument, ...], kwargs: dict[str, _StepArgument]
    ) -> None:
        """Pre-step hook: take the gradients of the weights the kernels will step.

        Args:
            optimizer: The optimizer about to step.
            args: Positional arguments of ``step``, the optimizer first.
            kwargs: Keyword arguments of ``step``.
        """
        if any(argument is not None for argument in (*args[1:], *kwargs.values())):
            return
        for group_index, group in enumerate(optimizer.param_groups):
            for weight in group["params"]:
                if id(weight) not in self._weight_ids or weight.grad is None or weight.grad.dtype != torch.bfloat16:
                    continue
                state = optimizer.state.get(weight)
                if state is None or not all(
                    name in state
                    and not isinstance(state[name], DTensor)
                    and state[name].dtype == torch.float32
                    and state[name].is_contiguous()
                    for name in _STATE_NAMES
                ):
                    continue
                self._held.append((group_index, weight, weight.grad))
                weight.grad = None

    @torch.no_grad()
    def step_held_weights(
        self, optimizer: torch.optim.Optimizer, args: tuple[_StepArgument, ...], kwargs: dict[str, _StepArgument]
    ) -> None:
        """Post-step hook: return the gradients and step their weights.

        Args:
            optimizer: The optimizer that just stepped.
            args: Positional arguments of ``step``, the optimizer first.
            kwargs: Keyword arguments of ``step``.
        """
        held, self._held = self._held, []
        for group_index, group in enumerate(optimizer.param_groups):
            weights = [weight for index, weight, _ in held if index == group_index]
            if not weights:
                continue
            gradients = [gradient for index, _, gradient in held if index == group_index]
            for weight, gradient in zip(weights, gradients, strict=True):
                weight.grad = gradient
            states = [optimizer.state[weight] for weight in weights]
            masters = [state["master_param"] for state in states]
            beta1, beta2 = group["betas"]
            self._extension.adam_requantize(
                [gradient.contiguous() for gradient in gradients],
                masters,
                [state["exp_avg"] for state in states],
                [state["exp_avg_sq"] for state in states],
                [weight._rowwise_data for weight in weights],
                [weight._rowwise_scale_inv for weight in weights],
                [weight._columnwise_data for weight in weights],
                [weight._columnwise_scale_inv for weight in weights],
                [weight._amax_rowwise for weight in weights],
                [weight._amax_columnwise for weight in weights],
                group["lr"],
                beta1,
                beta2,
                group["eps"],
                group["step"],
                optimizer.adam_w_mode,
                1 if group["bias_correction"] else 0,
                group["weight_decay"],
            )


def enable_nvfp4_fused_adam_step(model_parts: list[nn.Module], optimizers: list[torch.optim.Optimizer]) -> None:
    """Step the resident NVFP4 expert weights of every model part in the fused kernels.

    Applies to the weights of ``GroupedExpertsTENVFP4`` that are 2D block-scaled, rounded to nearest and hold
    both operands, and to Transformer Engine ``FusedAdam`` optimizers with FP32 master weights that are not
    capturable. Every other weight and optimizer keeps the ``FusedAdam`` step, as do all weights when the kernels
    cannot be built (logged as a warning).

    Args:
        model_parts: Pipeline model parts paired one-to-one with ``optimizers``.
        optimizers: Optimizers that hold the parameters of the matching model part.

    Raises:
        ValueError: If there is not one optimizer per model part.
    """
    expert_modules = [
        [module for module in model_part.modules() if isinstance(module, GroupedExpertsTENVFP4)]
        for model_part in model_parts
    ]
    if not any(expert_modules):
        return
    if len(model_parts) != len(optimizers):
        raise ValueError(
            f"The NVFP4 Adam step expected one optimizer per model part, got {len(optimizers)} optimizers "
            f"for {len(model_parts)} parts"
        )
    from transformer_engine.pytorch.optimizers import FusedAdam
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

    steps: list[tuple[FusedAdam, list[NVFP4Tensor]]] = []
    for modules, optimizer in zip(expert_modules, optimizers, strict=True):
        if (
            not isinstance(optimizer, FusedAdam)
            or not optimizer.master_weights
            or optimizer.capturable
            or optimizer.use_decoupled_grad
        ):
            continue
        weights = [
            parameter
            for module in modules
            for parameter in module.parameters()
            if isinstance(parameter, NVFP4Tensor) and _kernels_cover(parameter)
        ]
        if weights:
            steps.append((optimizer, weights))
    if not steps:
        return
    try:
        extension = _load_extension()
    except (RuntimeError, OSError) as error:
        logger.warning("NVFP4 expert weights keep the FusedAdam step: the Adam kernels could not be built (%s).", error)
        return
    for optimizer, weights in steps:
        step = _FusedAdamStep(weights, extension)
        optimizer.register_step_pre_hook(step.hold_gradients)
        optimizer.register_step_post_hook(step.step_held_weights)
        logger.info("The Adam step of %d NVFP4 expert weights runs in fused kernels.", len(weights))
