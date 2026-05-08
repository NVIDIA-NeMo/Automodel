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
from typing import Optional

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.parallelizer import (
    _extract_model_layers,
    _get_parallel_plan,
    fsdp2_strategy_parallelize,
)

logger = logging.getLogger(__name__)


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

def _apply_activation_checkpointing_single_gpu(model) -> None:
    """Apply activation checkpointing for single-GPU training.

    Mirrors the logic in ``fsdp2_strategy_parallelize`` so that models which
    don't set ``supports_gradient_checkpointing`` (e.g. Gemma4) still get
    sub-module-level checkpoint wrapping instead of crashing (required for 
    larger gemma4 models lora run on single DGX Spark GPU)
    """
    if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
        try:
            model.config.use_cache = False
        except Exception:
            pass

    _use_hf_native_grad_ckpt = False
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _HFGradLayer

        layers = _extract_model_layers(model)
        _use_hf_native_grad_ckpt = (
            bool(layers)
            and layers[0].__class__.__module__.startswith("transformers.")
            and isinstance(layers[0], _HFGradLayer)
            and getattr(model, "supports_gradient_checkpointing", False)
            and hasattr(model, "gradient_checkpointing_enable")
        )
    except (ImportError, Exception):
        layers = None

    if _use_hf_native_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    else:
        if layers is None:
            try:
                layers = _extract_model_layers(model)
            except Exception:
                logger.error("Model does not support gradient checkpointing and layer extraction failed.")
                return
        for i, layer in enumerate(layers):
            if hasattr(layer, "mlp"):
                layers[i].mlp = checkpoint_wrapper(layer.mlp)
            if hasattr(layer, "self_attn"):
                layers[i].self_attn = checkpoint_wrapper(layers[i].self_attn)
            if hasattr(layer, "input_layernorm"):
                layers[i].input_layernorm = checkpoint_wrapper(layers[i].input_layernorm)
            if hasattr(layer, "post_attention_layernorm"):
                layers[i].post_attention_layernorm = checkpoint_wrapper(layers[i].post_attention_layernorm)
        logger.info("Applied sub-module activation checkpointing for single-GPU training.")


class FSDP2Manager:
    """
    Manager for parallelizing models using FSDP2 with TP, DP, CP sharding.

    This manager applies parallelization to the model using a prescribed
    TP sharding plan. It supports mixed precision and CPU offloading options.

    The device mesh must be created externally and passed in.

    Args:
        config (FSDP2Config): Configuration for FSDP2 distributed training.
        device_mesh (DeviceMesh): Device mesh for distributed operations.
        moe_mesh (Optional[DeviceMesh]): Optional device mesh for expert parallelism.

    Example:
        from nemo_automodel.components.distributed.config import FSDP2Config

        config = FSDP2Config(sequence_parallel=True, activation_checkpointing=True)
        # device_mesh created externally via create_device_mesh()
        manager = FSDP2Manager(config, device_mesh=device_mesh, moe_mesh=moe_mesh)
        model = manager.parallelize(model)
    """

    def __init__(
        self,
        config: FSDP2Config,
        device_mesh: DeviceMesh,
        moe_mesh: Optional[DeviceMesh] = None,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.moe_mesh = moe_mesh

        # Extract config fields for easy access
        self.sequence_parallel = config.sequence_parallel
        self.tp_plan = config.tp_plan
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.activation_checkpointing = config.activation_checkpointing
        self.defer_fsdp_grad_sync = config.defer_fsdp_grad_sync
        self.backend = config.backend
        self.enable_async_tensor_parallel = config.enable_async_tensor_parallel
        self.enable_compile = config.enable_compile
        self.enable_fsdp2_prefetch = config.enable_fsdp2_prefetch
        self.fsdp2_backward_prefetch_depth = config.fsdp2_backward_prefetch_depth
        self.fsdp2_forward_prefetch_depth = config.fsdp2_forward_prefetch_depth

    def parallelize(self, model):
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        Args:
            model (nn.Module): The model to be parallelized.

        Returns:
            The parallelized model.
        """
        if get_world_size_safe() == 1:
            logger.info("World size is 1, skipping parallelization.")
            if self.activation_checkpointing:
                _apply_activation_checkpointing_single_gpu(model)
            return model

        if self.config.patch_is_packed_sequence:
            _patch_is_packed_sequence_for_training()

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=self.tp_plan,
            offload_policy=self.offload_policy,
            sequence_parallel=bool(self.sequence_parallel),
            activation_checkpointing=self.activation_checkpointing,
            enable_async_tensor_parallel=self.enable_async_tensor_parallel,
            enable_compile=self.enable_compile,
            enable_fsdp2_prefetch=self.enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth=self.fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth=self.fsdp2_forward_prefetch_depth,
        )

        return model

    def maybe_compile(self, model):
        """Apply per-layer compile after sharding, alongside whole-model compile_model()."""
        if self.enable_compile or (self.enable_async_tensor_parallel and self.device_mesh["tp"].size() > 1):
            from nemo_automodel.components.distributed.parallelizer import _apply_per_layer_compile

            _apply_per_layer_compile(model)
