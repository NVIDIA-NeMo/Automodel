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

"""Model-owned distributed parallelization for dense Qwen3.5 models."""

import logging

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.multimodal_fsdp import (
    FrozenMultimodalSharding,
    is_multimodal_module_name,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
)
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common.utils import _get_strict_fp32_module_keywords

logger = logging.getLogger(__name__)


class Qwen3_5ModelParallelizer(ModelParallelizer):
    """Shard each decoder layer as one FSDP unit with fp32 GatedDeltaNet gate parameters."""

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
        fp32_compute_module_names: tuple[str, ...] | None = None,
    ) -> None:
        """Shard each decoder layer as one FSDP unit whose strict fp32 parameters compute in fp32.

        Args:
            module: Root model on the first call; its ``_keep_in_fp32_modules_strict``
                provides the fp32 parameter-name tokens. Recursive calls receive
                submodules together with the resolved ``fp32_compute_module_names``.
            mesh: Device mesh for FSDP sharding.
            mp_policy: Mixed-precision policy of the enclosing boundary.
            offload_policy: FSDP offload policy.
            enable_fsdp2_prefetch: Unused; layer units rely on the FSDP2 defaults.
            fsdp2_backward_prefetch_depth: Unused.
            fsdp2_forward_prefetch_depth: Unused.
            reshard_after_forward: Optional override for every layer unit.
            frozen_multimodal_sharding: Policy for fully frozen multimodal modules.
            ignored_multimodal_params: Accumulator for replicated frozen multimodal parameters.
            fp32_compute_module_names: Resolved strict fp32 tokens; ``None`` on the root call.
        """
        del enable_fsdp2_prefetch, fsdp2_backward_prefetch_depth, fsdp2_forward_prefetch_depth
        if fp32_compute_module_names is None:
            fp32_compute_module_names = tuple(_get_strict_fp32_module_keywords(module))
        frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
        pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            all_items = list(module.items()) if isinstance(module, nn.ModuleDict) else list(enumerate(module))
            flat_items = [item for item in all_items if not isinstance(item[1], (nn.ModuleList, nn.ModuleDict))]
            nested_items = [item for item in all_items if isinstance(item[1], (nn.ModuleList, nn.ModuleDict))]

            for _, child in nested_items:
                self._apply_fsdp_sharding(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    frozen_multimodal_sharding=frozen_multimodal_sharding,
                    ignored_multimodal_params=ignored_multimodal_params,
                    fp32_compute_module_names=fp32_compute_module_names,
                )

            for index, (_, child) in enumerate(flat_items):
                if reshard_after_forward is not None:
                    layer_reshard_after_forward = reshard_after_forward
                elif pp_enabled:
                    layer_reshard_after_forward = False
                else:
                    layer_reshard_after_forward = index < len(flat_items) - 1
                fully_shard_by_dtype(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    fp32_compute_module_names=fp32_compute_module_names,
                    reshard_after_forward=layer_reshard_after_forward,
                    model_parallelizer=self,
                )
            return

        for name, child in module.named_children():
            if is_multimodal_module_name(name) and module_is_fully_frozen(child):
                if frozen_multimodal_sharding in ("root", "replicate"):
                    logger.info(
                        "Keeping frozen multimodal module %s at FSDP policy %s", name, frozen_multimodal_sharding
                    )
                    if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                        ignored_multimodal_params.update(module_parameters(child))
                    continue
            self._apply_fsdp_sharding(
                child,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward=reshard_after_forward,
                frozen_multimodal_sharding=frozen_multimodal_sharding,
                ignored_multimodal_params=ignored_multimodal_params,
                fp32_compute_module_names=fp32_compute_module_names,
            )

    def _apply(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        """Apply generic TP/AC/FSDP and install Qwen3.5's CP mesh."""
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1
        result = super()._apply(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            **kwargs,
        )
        if cp_enabled:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for module in model.modules():
                if isinstance(module, CPAwareGatedDeltaNet):
                    module._cp_mesh = cp_mesh
            model.cp_mesh = cp_mesh
        return result


PARALLELIZER = Qwen3_5ModelParallelizer()

__all__ = ["PARALLELIZER"]
