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

"""Model-owned FSDP2 parallelization for Beam: isolate the fp32 router and preserve fp32 RoPE inputs.

Every FSDP unit goes through ``BeamModelParallelizer._fully_shard_module``: Beam blocks disable FSDP's automatic
forward-input cast and cast hidden states themselves, so the fp32 RoPE tables reach attention unchanged. Decoder
blocks are sharded into dtype-uniform units, so the fp32 router (``_keep_in_fp32_modules_strict``) gets its own
unit: the MoE/EP path does this through the shared ``fully_shard_by_dtype`` block sharding, the dense FSDP2 path
through ``BeamModelParallelizer._apply_fsdp_sharding``."""

from dataclasses import replace
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.distributed.multimodal_fsdp import FrozenMultimodalSharding
from nemo_automodel.components.distributed.parallelizer import apply_fsdp2_sharding_recursively
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.beam.layers import Block


def is_beam_block(module: nn.Module) -> bool:
    """True for a Beam decoder ``Block``, also when wrapped for activation checkpointing."""
    return isinstance(unwrap_checkpoint_wrapper(module), Block)


def beam_fully_shard_unit(
    module: nn.Module, *, mp_policy: MixedPrecisionPolicy | None = None, **kwargs: Any
) -> nn.Module:
    """Shard one FSDP unit without casting a Beam block's FP32 rotary inputs.

    For Beam blocks, clone the policy with cast_forward_inputs=False and record param_dtype on the unwrapped
    block, whose forward casts hidden states itself. Other modules use the policy unchanged.

    Args:
        module: Module to wrap; a Beam block may be wrapped for activation checkpointing.
        mp_policy: Mixed-precision policy for this unit.
        **kwargs: Remaining ``fully_shard`` arguments (mesh, reshard_after_forward, offload_policy, ...).

    Returns:
        The sharded module."""
    if mp_policy is not None and is_beam_block(module):
        unwrap_checkpoint_wrapper(module)._fsdp_compute_dtype = mp_policy.param_dtype
        if mp_policy.cast_forward_inputs:
            mp_policy = replace(mp_policy, cast_forward_inputs=False)
    return fully_shard(module, mp_policy=mp_policy, **kwargs)


class BeamModelParallelizer(ModelParallelizer):
    """Shard Beam with an fp32 router unit and fp32 RoPE tables kept across block boundaries."""

    _customizes_moe_fsdp = True

    def _fully_shard_module(self, module: nn.Module, **kwargs: Any) -> nn.Module:
        """Apply ``beam_fully_shard_unit`` to every FSDP unit."""
        return beam_fully_shard_unit(module, **kwargs)

    def _apply_fsdp_sharding(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        ignored_multimodal_params: set[nn.Parameter] | None = None,
    ) -> None:
        """Run the shared dense FSDP2 walk with Beam blocks sharded into dtype-uniform units."""
        apply_fsdp2_sharding_recursively(
            module,
            mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
            model_parallelizer=_DenseBlockParallelizer(
                self, tuple(getattr(module, "_keep_in_fp32_modules_strict", None) or ())
            ),
            frozen_multimodal_sharding=frozen_multimodal_sharding,
            ignored_multimodal_params=ignored_multimodal_params,
        )


class _DenseBlockParallelizer(ModelParallelizer):
    """Dense FSDP2 walk helper: Beam blocks go through ``fully_shard_by_dtype``, other units unchanged.

    The MoE path already shards each block with ``fully_shard_by_dtype``; the dense walk wraps each layer with a
    single ``_fully_shard_module`` call, which this helper turns into the same dtype-aware block sharding."""

    def __init__(self, unit_parallelizer: BeamModelParallelizer, fp32_compute_module_names: tuple[str, ...]) -> None:
        self._unit_parallelizer = unit_parallelizer
        self._fp32_compute_module_names = fp32_compute_module_names

    def _fully_shard_module(self, module: nn.Module, **kwargs: Any) -> nn.Module:
        """Shard a Beam block into dtype-uniform units; shard any other module as one Beam unit."""
        if not (self._fp32_compute_module_names and is_beam_block(module)):
            return self._unit_parallelizer._fully_shard_module(module, **kwargs)
        fully_shard_by_dtype(
            module,
            mesh=kwargs["mesh"],
            mp_policy=kwargs.get("mp_policy"),
            offload_policy=kwargs.get("offload_policy"),
            fp32_compute_module_names=self._fp32_compute_module_names,
            reshard_after_forward=kwargs.get("reshard_after_forward"),
            ignored_params=kwargs.get("ignored_params"),
            model_parallelizer=self._unit_parallelizer,
        )
        return module


PARALLELIZER = BeamModelParallelizer()

__all__ = ["PARALLELIZER"]
