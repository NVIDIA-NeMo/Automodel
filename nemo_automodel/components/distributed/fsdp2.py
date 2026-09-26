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

import warnings
from collections.abc import Callable

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.model_parallelizer import (
    ParallelizeContext,
    compile_parallelized_model,
    parallelize_model,
)


def fsdp2_sharding_enabled(device_mesh: DeviceMesh) -> bool:
    """Report whether :class:`ModelParallelizer` shards the model for this mesh.

    Parallelization is skipped on a single-rank world or a single-element mesh, which
    also skips every side effect of ``fully_shard`` — most importantly the
    ``MixedPrecisionPolicy`` cast of parameters to the compute dtype. Callers that
    depend on that cast must check this instead of assuming FSDP2 is active.

    Args:
        device_mesh: Device mesh from the model's parallelization context.

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

    .. deprecated:: 0.7
        Pass :class:`FSDP2Config` through the config-driven infrastructure and
        provide model-specific behavior with :class:`ModelParallelizer`. This
        compatibility class is scheduled for removal in 0.8.

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
            "FSDP2Manager is deprecated and will be removed in 0.8; pass FSDP2Config through the "
            "config-driven infrastructure and use ModelParallelizer for model-owned behavior.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.device_mesh = device_mesh
        self.moe_mesh = moe_mesh
        self.moe_config = moe_config

    def __getattr__(self, name):
        if name == "frozen_multimodal_sharding":
            return self.config.multimodal.frozen_sharding
        return getattr(self.config, name)

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
        context = ParallelizeContext(
            mesh=MeshContext.from_meshes(self.device_mesh, self.moe_mesh),
            strategy=self.config,
            moe=self.moe_config,
            activation_checkpointing=self.activation_checkpointing,
        )
        compile_parallelized_model(model, context)
