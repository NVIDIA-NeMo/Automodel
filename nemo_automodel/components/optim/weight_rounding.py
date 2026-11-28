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

# The BF16 bit conversion below is adapted from:
# https://github.com/andrewgcodes/faircast
# Revision: 7c213b2f80834e00bfa7201b11421a3c1e462e51
# Original file: gsm8k_reference.py (MIT).
# Copyright (c) 2026 Andrew Gao and William Gao.
# The original license is retained here and in weight_rounding_LICENSE.txt.
#
# MIT License
#
# Copyright (c) 2026 Andrew Gao and William Gao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Internal BF16 weight materialization around unchanged native AdamW updates.

AdamWConfig owns construction. The hooks keep FP32 master weights separate from
the BF16-grid values held in model parameter storage. Legacy checkpoint field
names are retained; experiment-specific phase resets are not implemented here.
"""

from collections.abc import Callable
from typing import Literal

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

WeightRounding = Literal["nearest", "stochastic"]
# AdamW exposes closures through its external step-hook calling convention.
StepClosure = Callable[[], float | torch.Tensor]
StepArgs = tuple[torch.optim.AdamW | StepClosure | None, ...]
StepKwargs = dict[str, StepClosure | None]


def _stochastic_round_bf16(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Round to the adjacent BF16 values with distance-proportional probabilities.

    Implements the stochastic export rule in arXiv:2610.41073, Section 5.
    Bit conversion is adapted from andrewgcodes/faircast (MIT; weight_rounding_LICENSE.txt).
    The result is unbiased for finite values between the finite BF16 extrema.
    NaNs, infinities, and signed zeros are preserved; overflow follows BF16 limits.

    Args:
        value: Local FP32 tensor of arbitrary shape [...], on CPU or CUDA.
        generator: Generator on the same device; its state advances.

    Returns:
        Independent BF16 tensor of shape [...] on the input device.
    """
    if value.dtype != torch.float32 or isinstance(value, DTensor):
        raise TypeError("stochastic_round_bf16 requires a local float32 tensor")
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(65536, bits.shape, dtype=torch.int32, device=value.device, generator=generator)
    rounded = ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    return torch.where(torch.isfinite(value), rounded, value.to(torch.bfloat16))


@torch.no_grad()
def _refresh_bf16_weights(optimizer: torch.optim.AdamW) -> None:
    """Materialize BF16 working weights q without performing an Adam update.

    Args:
        optimizer: Native AdamW configured by attach_bf16_weight_rounding. Parameters have
            arbitrary global shape [...]; local FP32 storage holds q and
            checkpointed FP32 state["master"] holds theta with the same
            Shard/Replicate layout. Call only between completed training/eval
            operations, after FSDP resharding. This mutates q and the checkpointed
            group quantization_step, leaving theta, moments, Adam step and
            gradients unchanged. Parameters without a master retain their
            initial BF16-grid values. Consumers must use the materialized weights;
            this helper does not synchronize external copies.
    """
    # PyTorch exposes optimizer/subclass and DTensor layout only at this boundary.
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("BF16 weight rounding requires exact torch.optim.AdamW")
    index = 0
    for group in optimizer.param_groups:
        group["quantization_step"] += 1
        for param in group["params"]:
            index += 1
            state = optimizer.state.get(param, {})
            if not group["quantize"] or not state:
                continue
            master = state["master"]
            target = param.to_local() if isinstance(param, DTensor) else param
            local_master = master.to_local() if isinstance(master, DTensor) else master
            if group["rounding"] == "stochastic":
                shard = 0
                if isinstance(param, DTensor):
                    for axis, (placement, coordinate) in enumerate(
                        zip(param.placements, param.device_mesh.get_coordinate())
                    ):
                        if isinstance(placement, Shard):
                            shard = shard * param.device_mesh.size(axis) + coordinate
                seed = (
                    group["rounding_seed"]
                    + index * 1000003
                    + group["quantization_step"] * 1000000007
                    + shard * 100000000003
                ) % (2**63 - 1)
                generator = torch.Generator(device=target.device).manual_seed(seed)
                target.copy_(_stochastic_round_bf16(local_master, generator))
            else:
                target.copy_(local_master.to(torch.bfloat16))


def attach_bf16_weight_rounding(
    optimizer: torch.optim.AdamW, *, rounding: WeightRounding = "nearest", rounding_seed: int = 42
) -> None:
    """Attach master/q switching to a newly constructed native AdamW.

    Model parameters are FP32 tensors of arbitrary shape [...], or DTensors
    retaining their global shape and Shard/Replicate placements. Their local
    storage holds q throughout forward, backward and activation recomputation.
    After backward has completed and FSDP has resharded, hooks temporarily
    restore theta for native AdamW.step and materialize one new BF16-grid q.
    Forward and recomputation must consume the same materialized weights.

    Initial values are left unchanged: callers must start from BF16-grid
    checkpoint values. Native AdamW initializes its own state on the first
    step; subsequent FP32 master tensors are checkpointed under state["master"].
    Save and restore both model q and optimizer state, with the same mesh and
    parameter order. Native checkpoint construction may create zero Adam state
    for unused parameters; its state-dict hook then adds a master copy without
    changing q, RNG, or counters. Hooks themselves are reinstalled by optimizer construction.
    A group may set quantize=False for protected FP32 parameters. An exception
    inside a step cannot roll back native AdamW or restore q automatically;
    discard or restore the optimizer and model before continuing after failure.

    Args:
        optimizer: Exact torch.optim.AdamW instance with FP32 parameters and
            fully reduced gradients. Construction must precede attaching hooks;
            optimizer state must be empty. Closure, graph capture, differentiable,
            fused and foreach variants are not supported.
        rounding: Nearest-even or stochastic BF16 quantization after each update.
        rounding_seed: Private RNG base seed; replicas share streams and shards
            use distinct streams. Global torch RNG state is unchanged.
    """
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("BF16 weight rounding requires exact torch.optim.AdamW")
    if rounding not in ("nearest", "stochastic"):
        raise ValueError(f"Unsupported rounding: {rounding}")
    if optimizer.state:
        raise ValueError("Attach BF16 weight rounding before stepping or loading optimizer state")
    for group in optimizer.param_groups:
        if "rounding_seed" in group:
            raise ValueError("BF16 weight rounding hooks are already attached")
        if any(group[key] for key in ("capturable", "differentiable", "fused", "foreach")):
            raise ValueError("BF16 weight rounding supports eager, non-fused, non-foreach AdamW only")
        # Preserve the legacy checkpoint wire keys; phase resets are owned by callers.
        group.update(
            rounding=rounding,
            rounding_seed=rounding_seed,
            quantization_step=0,
            faircast_phase=0,
            faircast_phase_start_global_step=0,
        )
        group.setdefault("quantize", True)
        for param in group["params"]:
            if param.dtype != torch.float32:
                raise TypeError("BF16 weight rounding requires FP32 parameter storage")
            if isinstance(param, DTensor) and any(
                not isinstance(placement, (Shard, Replicate)) for placement in param.placements
            ):
                raise ValueError("BF16 weight rounding supports only Shard/Replicate parameter placements")

    @torch.no_grad()
    def before_step(opt: torch.optim.AdamW, args: StepArgs, kwargs: StepKwargs) -> None:
        if (len(args) > 1 and args[1] is not None) or kwargs.get("closure") is not None:
            raise ValueError("BF16 weight rounding requires backward before step; closures are unsupported")
        for group in opt.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse or (
                    isinstance(grad, DTensor) and any(not isinstance(p, (Shard, Replicate)) for p in grad.placements)
                ):
                    raise ValueError("BF16 weight rounding requires dense, fully reduced gradients")
                state = opt.state.get(param, {})
                if group["quantize"] and state and "master" not in state:
                    raise ValueError("Initialized BF16 weight rounding state is missing its FP32 master")
                if group["quantize"] and "master" in state:
                    target = param.to_local() if isinstance(param, DTensor) else param
                    master = state["master"]
                    target.copy_(master.to_local() if isinstance(master, DTensor) else master)

    @torch.no_grad()
    def after_step(opt: torch.optim.AdamW, args: StepArgs, kwargs: StepKwargs) -> None:
        for group in opt.param_groups:
            for param in group["params"]:
                if param.grad is None or not group["quantize"]:
                    continue
                state = opt.state[param]
                if "master" not in state:
                    state["master"] = param.detach().clone()
                else:
                    state["master"].copy_(param)
        _refresh_bf16_weights(opt)

    def before_state_dict(opt: torch.optim.Optimizer) -> None:
        # Native DCP materializes zero Adam state before enumerating its keys.
        # Add only the master-shaped destination here, not at construction.
        # Existing masters are authoritative and must never be reconstructed.
        for group in opt.param_groups:
            if not group["quantize"]:
                continue
            for param in group["params"]:
                state = opt.state.get(param, {})
                if not state or "master" in state:
                    continue
                required = {"step", "exp_avg", "exp_avg_sq"}
                if group["amsgrad"]:
                    required.add("max_exp_avg_sq")
                if not required <= state.keys() or any(
                    bool(torch.count_nonzero(state[key].to_local() if isinstance(state[key], DTensor) else state[key]))
                    for key in required
                ):
                    raise ValueError("Cannot reconstruct a missing BF16 master from nonzero or incomplete Adam state")
                state["master"] = param.detach().clone()

    def after_load(opt: torch.optim.Optimizer) -> None:
        for group in opt.param_groups:
            if (
                not {
                    "rounding",
                    "rounding_seed",
                    "quantize",
                    "quantization_step",
                    "faircast_phase",
                    "faircast_phase_start_global_step",
                }
                <= group.keys()
            ):
                raise ValueError("Cannot load a plain AdamW checkpoint into BF16 weight rounding")
            if any(group[key] for key in ("capturable", "differentiable", "fused", "foreach")):
                raise ValueError("Loaded AdamW execution variant is unsupported by BF16 weight rounding")
            if group["rounding"] not in ("nearest", "stochastic"):
                raise ValueError("Loaded BF16 weight rounding mode is invalid")
            if not isinstance(group["quantization_step"], int) or group["quantization_step"] < 0:
                raise ValueError("Loaded BF16 weight rounding quantization counter is invalid")
            for param in group["params"]:
                state = opt.state.get(param, {})
                if group["quantize"] and state and "master" not in state:
                    raise ValueError("Loaded BF16 weight rounding state is missing its FP32 master")

    optimizer.register_state_dict_pre_hook(before_state_dict)
    optimizer.register_load_state_dict_post_hook(after_load)
    optimizer.register_step_pre_hook(before_step)
    optimizer.register_step_post_hook(after_step)
