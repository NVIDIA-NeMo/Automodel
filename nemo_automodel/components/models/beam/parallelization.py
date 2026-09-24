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

"""Beam FSDP2 policy: isolate the fp32 router and preserve fp32 RoPE inputs.

DP/PP use the registered strategy; EP uses the model's _nemo_fully_shard_fn hook.
Beam blocks disable automatic input casts and cast hidden states themselves,
leaving RoPE tables untouched. All other units retain the resolved policy."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    register_parallel_strategy,
)
from nemo_automodel.components.models.beam.layers import Block

# Attribute under which ``torch.distributed.algorithms._checkpoint.checkpoint_wrapper`` stores the wrapped module.
CHECKPOINT_WRAPPER_ATTR = "_checkpoint_wrapped_module"


def wrapper_tolerant_names(names: Iterable[str]) -> tuple[str, ...]:
    """Expand dotted precision pins to match activation-checkpoint wrapper segments."""
    out: list[str] = []
    for name in names:
        parts = name.split(".")
        out.append(name)
        for i in range(1, len(parts)):
            out.append(".".join(parts[:i] + [CHECKPOINT_WRAPPER_ATTR] + parts[i:]))
    return tuple(dict.fromkeys(out))


def is_beam_block(module: nn.Module) -> bool:
    """True for a Beam decoder ``Block``, also when wrapped for activation checkpointing."""
    return isinstance(unwrap_checkpoint_wrapper(module), Block)


def beam_fully_shard_unit(
    module: nn.Module,
    *,
    mp_policy: MixedPrecisionPolicy | None = None,
    fully_shard_fn: Callable[..., nn.Module] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Shard a Beam block without casting its FP32 rotary inputs.

    Clone the policy with cast_forward_inputs=False and record param_dtype on the
    unwrapped block, whose forward casts hidden states. Other modules are unchanged;
    fully_shard_fn selects the underlying sharder."""
    shard = fully_shard_fn or fully_shard
    if mp_policy is not None and is_beam_block(module):
        unwrap_checkpoint_wrapper(module)._fsdp_compute_dtype = mp_policy.param_dtype
        if mp_policy.cast_forward_inputs:
            mp_policy = replace(mp_policy, cast_forward_inputs=False)
    return shard(module, mp_policy=mp_policy, **kwargs)


@register_parallel_strategy(name="BeamForCausalLM")
class BeamParallelizationStrategy(DefaultParallelizationStrategy):
    """Default FSDP2 strategy with dtype-aware sharding for Beam decoder blocks."""

    def parallelize(
        self, model: nn.Module, device_mesh: DeviceMesh, dp_shard_cp_mesh_name: str = "dp_shard_cp", **kwargs: Any
    ) -> nn.Module:
        """Apply the default strategy with Beam's dtype-aware sharder.

        Args:
            model: Native Beam model or pipeline stage, modified in place.
            device_mesh: Framework mesh containing the DP/CP axes.
            dp_shard_cp_mesh_name: FSDP mesh name.
            **kwargs: DefaultParallelizationStrategy options.

        Returns:
            The parallelized model."""
        base_fully_shard = kwargs.pop("fully_shard_fn", None) or fully_shard
        pins = wrapper_tolerant_names(getattr(model, "_keep_in_fp32_modules_strict", None) or ())

        def beam_fully_shard(module, *, mesh, mp_policy=None, offload_policy=None, reshard_after_forward=None, **extra):
            ignored_params = extra.pop("ignored_params", None)
            if pins and is_beam_block(module):
                # The block (or its checkpoint wrapper, exactly the object the default flow would wrap) becomes the
                # bf16 unit; the Gate inside it gets its own fp32 unit. Pin names resolve through wrapper segments.
                # ``beam_fully_shard_unit`` keeps the fp32 RoPE tensors across the block boundary.
                parallelizer_utils.fully_shard_by_dtype(
                    module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    fp32_compute_module_names=pins,
                    reshard_after_forward=reshard_after_forward,
                    ignored_params=ignored_params,
                    fully_shard_fn=lambda m, **kw: beam_fully_shard_unit(m, fully_shard_fn=base_fully_shard, **kw),
                )
                return module
            call_kwargs = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": offload_policy, **extra}
            if reshard_after_forward is not None:
                call_kwargs["reshard_after_forward"] = reshard_after_forward
            if ignored_params is not None:
                call_kwargs["ignored_params"] = ignored_params
            return base_fully_shard(module, **call_kwargs)

        return super().parallelize(
            model, device_mesh, dp_shard_cp_mesh_name=dp_shard_cp_mesh_name, fully_shard_fn=beam_fully_shard, **kwargs
        )
