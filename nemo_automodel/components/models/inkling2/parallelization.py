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

"""Inkling precision policy on Automodel's shared FSDP and expert parallelizers."""

from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer

FP32_MODULES = ("attn_query", "mlp_query", "output_query", "q_l2_clip", "rel_l2_clip", "mlp.gate")


def _is_fp32_module_name(name: str) -> bool:
    """Return whether a dotted module name is one of the model-owned FP32 modules.

    Args:
        name: Module name relative to the module being sharded; activation
            checkpoint wrappers add a ``_checkpoint_wrapped_module`` segment.

    Returns:
        True when the name equals or ends with an ``FP32_MODULES`` entry.
    """
    return any(name == suffix or name.endswith("." + suffix) for suffix in FP32_MODULES)


def iter_fp32_islands(module: nn.Module, owned: set[nn.Parameter]) -> Iterator[nn.Module]:
    """Yield the FP32 modules below ``module`` that need their own FSDP unit.

    Args:
        module: Module about to become an FSDP unit; only its descendants are searched.
        owned: Parameters already owned by nested FSDP units or by expert parallelism.

    Yields:
        Descendants named in ``FP32_MODULES`` whose unowned floating parameters
        are all FP32, outermost first. Nested FSDP units, their descendants and
        descendants of a yielded module are skipped, so buffer-only containers
        such as the layer collection never become units.
    """
    owned_ids = {id(param) for param in owned}
    skipped: list[str] = []
    for name, submodule in module.named_modules():
        if submodule is module or any(name.startswith(prefix + ".") for prefix in skipped):
            continue
        if isinstance(submodule, FSDPModule):
            skipped.append(name)
            continue
        if not _is_fp32_module_name(name):
            continue
        dtypes = {
            param.dtype
            for param in submodule.parameters()
            if torch.is_floating_point(param) and id(param) not in owned_ids
        }
        if dtypes != {torch.float32}:
            continue
        skipped.append(name)
        yield submodule


def _shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    reshard_after_forward: bool | int | None,
    owned: set[nn.Parameter],
) -> None:
    """Make ``module`` one FSDP unit that leaves already-owned parameters alone.

    Args:
        module: Module whose remaining parameters become one FSDP unit.
        mesh: One-dimensional DP shard mesh or two-dimensional HSDP mesh.
        mp_policy: Compute policy of this unit.
        offload_policy: Shared parameter storage/offload policy.
        reshard_after_forward: Shared FSDP reshard setting.
        owned: Parameters owned elsewhere; only those inside ``module`` are passed on.
    """
    module_param_ids = {id(param) for param in module.parameters()}
    ignored = {param for param in owned if id(param) in module_param_ids}
    fully_shard(
        module,
        mesh=mesh,
        mp_policy=mp_policy or MixedPrecisionPolicy(),
        offload_policy=offload_policy or OffloadPolicy(),
        reshard_after_forward=reshard_after_forward,
        ignored_params=ignored or None,
    )


class Inkling2Parallelizer(ModelParallelizer):
    """Keep model-owned casts while reusing shared sharding and communication."""

    _customizes_moe_fsdp = True

    def parallelize(self, model: nn.Module, mesh_context: MeshContext, /) -> nn.Module:
        """Validate qualified text topology before shared parallelization mutates ownership.

        Args:
            model: Inkling text model with shared grouped experts and an untied head.
            mesh_context: Shared meshes and strategy/compute policy.

        Returns:
            The same model after shared FSDP2 and optional expert sharding.
        """
        if mesh_context.tp_size != 1:
            raise ValueError("Inkling2 currently requires TP=1")
        strategy = mesh_context.strategy_config
        if not isinstance(strategy, FSDP2Config):
            raise ValueError("Inkling2 distributed text training requires FSDP2")
        moe = mesh_context.moe_parallel_config
        policies = (strategy.mp_policy, moe.mp_policy if moe is not None else None)
        if any(
            policy is not None and policy.param_dtype not in (None, torch.float32, torch.bfloat16)
            for policy in policies
        ):
            raise ValueError("Inkling2 FSDP compute requires float32 or bfloat16")
        from .attention import Inkling2Attention
        from .layers import ShortConvolution
        from .model import Inkling2ForCausalLM

        if not isinstance(model, Inkling2ForCausalLM):
            raise TypeError("Inkling2Parallelizer requires an Inkling2 model or pipeline stage")
        device_mesh = mesh_context.device_mesh
        cp_mesh = device_mesh["cp"] if device_mesh is not None and "cp" in (device_mesh.mesh_dim_names or ()) else None
        model.cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
        for module in model.modules():
            if isinstance(module, (Inkling2Attention, ShortConvolution)):
                module.cp_mesh = model.cp_mesh
        from .vision import Inkling2VisionEncoder

        visual = getattr(model, "visual", None)
        if isinstance(visual, Inkling2VisionEncoder) and model.cp_mesh is not None:
            assert device_mesh is not None
            visual.image_cp_mesh = device_mesh["dp_cp"] if any(p.requires_grad for p in visual.parameters()) else None
        return super().parallelize(model, mesh_context)

    def _use_full_layer_activation_checkpointing(self, model: nn.Module) -> bool:
        """Residual histories are explicit immutable inputs to each decoder layer."""
        return True

    # Concretize the base hook's untyped **kwargs to its FSDP call contract.
    def _fully_shard_module(  # ty: ignore[invalid-method-override]
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        reshard_after_forward: bool | int | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> nn.Module:
        """Shard the FP32 modules as their own units, then the module around them.

        Args:
            module: Owner of matrices [output, input], expert tensors [experts,
                input, output], and vectors [features]. Nested FSDP units keep ownership.
            mesh: One-dimensional DP shard mesh or two-dimensional HSDP mesh.
            mp_policy: Requested compute policy; model code owns input/output casts.
            offload_policy: Shared parameter storage/offload policy.
            reshard_after_forward: Shared FSDP reshard setting.
            ignored_params: Parameters with the module's tensor layouts already
                owned by another parallelism unit; their placements are preserved.

        Returns:
            The same module with global-shaped DTensor parameters. Dense parameters
            shard on the leading axis; expert ownership remains with the shared EP path.
            Under a low-precision policy every ``FP32_MODULES`` descendant computes and
            reduces in FP32 inside its own unit; an all-FP32 module under an FP32 or
            absent policy is one unit.
        """
        if mp_policy is not None:
            mp_policy = replace(mp_policy, cast_forward_inputs=False, output_dtype=None)
        owned = set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.update(child.parameters())
        owned_ids = {id(param) for param in owned}
        storage_dtypes = {
            param.dtype
            for param in module.parameters()
            if torch.is_floating_point(param) and id(param) not in owned_ids
        }
        compute_dtype = None if mp_policy is None else mp_policy.param_dtype
        if storage_dtypes != {torch.float32} or compute_dtype not in (None, torch.float32):
            fp32_policy = (
                None if mp_policy is None else replace(mp_policy, param_dtype=torch.float32, reduce_dtype=torch.float32)
            )
            for island in iter_fp32_islands(module, owned):
                _shard(
                    island,
                    mesh=mesh,
                    mp_policy=fp32_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    owned=owned,
                )
                owned.update(island.parameters())
        if not isinstance(module, FSDPModule):
            _shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward,
                owned=owned,
            )
        return module
