# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import warnings
from collections.abc import Callable

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_submodule_checkpointing,
    detect_kv_sharing_and_maybe_disable_cache,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.model_parallelizer import ParallelizeContext, parallelize_model
from nemo_automodel.components.distributed.parallelizer import (
    _extract_model_layer_groups,
    _filter_layer_groups_for_activation_checkpointing,
    _should_use_hf_native_gradient_checkpointing,
    apply_selective_activation_checkpointing,
)

logger = logging.getLogger(__name__)


def fsdp2_sharding_enabled(device_mesh: DeviceMesh) -> bool:
    """Report whether :meth:`FSDP2Manager.parallelize` shards the model for this mesh.

    Parallelization is skipped on a single-rank world or a single-element mesh, which
    also skips every side effect of ``fully_shard`` — most importantly the
    ``MixedPrecisionPolicy`` cast of parameters to the compute dtype. Callers that
    depend on that cast must check this instead of assuming FSDP2 is active.

    Args:
        device_mesh: Device mesh the ``FSDP2Manager`` was constructed with.

    Returns:
        True when ``fully_shard`` is applied, False when parallelization is skipped.
    """
    if get_world_size_safe() == 1 or device_mesh.size() == 1:
        return False
    return True


def _patch_is_packed_sequence_for_training() -> None:
    """Eliminate CPU-GPU sync from flash attention for standard (non-packed) training.

    transformers._is_packed_sequence() returns a GPU bool scalar when batch_size==1,
    which causes Python's ``if`` to call aten::is_nonzero — a CPU-GPU sync — once per
    attention layer per forward pass.  With FSDP+TP+gradient-checkpointing this fires
    hundreds of times per iteration.

    For standard (non-packed) training sequences are never packed, so returning the
    Python False immediately is both correct and avoids the sync.  Do NOT apply this
    patch when using packed-sequence training (multiple sequences concatenated into one
    tensor with position_ids that reset to 0 mid-sequence).
    """
    try:
        import transformers.modeling_flash_attention_utils as _fa_utils

        if getattr(_fa_utils, "_is_packed_sequence_patched", False):
            return  # already patched

        def _is_packed_sequence_no_sync(position_ids, batch_size):
            # Non-packed training: position_ids is always a simple arange -- never packed.
            return False

        _fa_utils._is_packed_sequence = _is_packed_sequence_no_sync
        _fa_utils._is_packed_sequence_patched = True
    except (ImportError, AttributeError):
        pass


class FSDP2Manager:
    """Deprecated compatibility wrapper for FSDP2 parallelization.

    .. deprecated:: 0.6
        Pass :class:`FSDP2Config` through the config-driven infrastructure and
        provide model-specific behavior with :class:`ModelParallelizer`. This
        compatibility class is scheduled for removal in 0.7.

    This manager applies parallelization to the model using a prescribed
    TP sharding plan. It supports mixed precision and CPU offloading options.

    The device mesh must be created externally and passed in.

    Args:
        config (FSDP2Config): Configuration for FSDP2 distributed training.
        device_mesh (DeviceMesh): Device mesh for distributed operations.
        moe_mesh (Optional[DeviceMesh]): Optional device mesh for expert parallelism.
        moe_config: Optional expert-parallel policy included in the model's
            :class:`ParallelizeContext`.

    """

    def __init__(
        self,
        config: FSDP2Config,
        device_mesh: DeviceMesh,
        moe_mesh: DeviceMesh | None = None,
        moe_config: MoEParallelizerConfig | None = None,
    ):
        warnings.warn(
            "FSDP2Manager is deprecated and will be removed in 0.7; pass FSDP2Config through the "
            "config-driven infrastructure and use ModelParallelizer for model-owned behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.device_mesh = device_mesh
        self.moe_mesh = moe_mesh
        self.moe_config = moe_config

        # Extract config fields for easy access
        self.sequence_parallel = config.sequence_parallel
        self.tp_plan = config.tp_plan
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.activation_checkpointing = config.activation_checkpointing
        self.activation_checkpointing_scope = config.activation_checkpointing_scope
        self.defer_fsdp_grad_sync = config.defer_fsdp_grad_sync
        self.reshard_after_forward = config.reshard_after_forward
        self.enable_async_tensor_parallel = config.enable_async_tensor_parallel
        self.enable_compile = config.enable_compile
        self.enable_fsdp2_prefetch = config.enable_fsdp2_prefetch
        self.fsdp2_backward_prefetch_depth = config.fsdp2_backward_prefetch_depth
        self.fsdp2_forward_prefetch_depth = config.fsdp2_forward_prefetch_depth
        self.frozen_multimodal_sharding = config.multimodal.frozen_sharding

    def parallelize(
        self,
        model: nn.Module,
        reapply_trainability: Callable[[nn.Module], None] | None = None,
    ) -> nn.Module:
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        Args:
            model (nn.Module): The model to be parallelized.
            reapply_trainability: Optional callback that re-resolves parameter
                trainability after model surgery and before FSDP construction.

        Returns:
            The parallelized model.
        """
        if not fsdp2_sharding_enabled(self.device_mesh):
            logger.info("World size or FSDP mesh size is 1, skipping parallelization.")
            if self.activation_checkpointing:
                if is_selective_activation_checkpointing(self.activation_checkpointing):
                    # Selective AC works on a plain model (no FSDP required), so
                    # honor it on a single GPU instead of silently falling back
                    # to full HF gradient checkpointing.
                    apply_selective_activation_checkpointing(
                        model,
                        enable_compile=self.enable_compile,
                        activation_checkpointing_scope=self.activation_checkpointing_scope,
                    )
                else:
                    layer_groups = _extract_model_layer_groups(model)
                    layers, ac_scopes = _filter_layer_groups_for_activation_checkpointing(
                        layer_groups,
                        self.activation_checkpointing_scope,
                    )
                    if _should_use_hf_native_gradient_checkpointing(
                        model,
                        layer_groups,
                        ac_scopes,
                        enable_compile=self.enable_compile,
                    ):
                        model.gradient_checkpointing_enable()
                    else:
                        apply_submodule_checkpointing(layers, detect_kv_sharing_and_maybe_disable_cache(model))
            if reapply_trainability is not None:
                reapply_trainability(model)
            return model

        if self.config.patch_is_packed_sequence:
            _patch_is_packed_sequence_for_training()

        context = ParallelizeContext(
            mesh=MeshContext.from_meshes(self.device_mesh, self.moe_mesh),
            strategy=self.config,
            moe=self.moe_config,
            activation_checkpointing=self.activation_checkpointing,
            reapply_trainability=reapply_trainability,
        )
        model = parallelize_model(
            model,
            context,
        )

        return model

    def maybe_compile(self, model):
        """Apply per-layer compile after sharding, alongside whole-model compile_model()."""
        if self.enable_compile or (self.enable_async_tensor_parallel and self.device_mesh["tp"].size() > 1):
            from nemo_automodel.components.distributed.parallelizer import _apply_per_layer_compile

            _apply_per_layer_compile(model)
