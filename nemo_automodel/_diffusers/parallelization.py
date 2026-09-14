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

"""Parallelization strategies for the diffusers transformers NeMo AutoModel trains."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Dict, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module

from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    ParallelizationStrategy,
    apply_fsdp2_sharding_recursively,
)

logger = logging.getLogger(__name__)


class WanParallelizationStrategy(ParallelizationStrategy):
    """Parallelization strategy for Wan-style transformer modules used in Diffusers.

    Applies TP to condition embedders, FFN projections in each block, and final projection,
    then applies FSDP sharding similarly to other strategies.
    """

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        # Not using custom tp_shard_plan; apply Wan-specific plan
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        # Apply TP only when TP group size > 1
        if tp_mesh.size() > 1:
            # Condition embedders if present
            try:
                if hasattr(model, "condition_embedder"):
                    cond = model.condition_embedder
                    if hasattr(cond, "text_embedder"):
                        cond.text_embedder = parallelize_module(
                            cond.text_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_embedder"):
                        cond.time_embedder = parallelize_module(
                            cond.time_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_proj"):
                        cond.time_proj = parallelize_module(
                            cond.time_proj,
                            tp_mesh,
                            {"": ColwiseParallel()},
                        )
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP condition embedders: {e}")

            # Blocks FFN and final projection
            try:
                if hasattr(model, "blocks"):
                    for block in model.blocks:
                        if hasattr(block, "ffn"):
                            block.ffn = parallelize_module(
                                block.ffn,
                                tp_mesh,
                                {
                                    "net.0.proj": ColwiseParallel(),
                                    "net.2": RowwiseParallel(),
                                },
                            )
                if hasattr(model, "proj_out"):
                    model.proj_out = parallelize_module(model.proj_out, tp_mesh, {"": RowwiseParallel()})
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP blocks/proj_out: {e}")

        # Activation checkpointing wraps every WanTransformerBlock so its
        # forward activations are recomputed on backward instead of being
        # held in memory. Critical for Wan2.2-A14B (14B params, ~30k-token
        # video sequence) — without this, fp32 layer-norm casts in the block
        # forward will OOM even on 8x80GB H100.
        if activation_checkpointing and hasattr(model, "blocks"):
            for idx in range(len(model.blocks)):
                model.blocks[idx] = checkpoint_wrapper(
                    model.blocks[idx],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        # Mixed precision default like Default strategy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        if reapply_trainability is not None:
            reapply_trainability(model)

        # Apply FSDP sharding recursively and to root
        apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            kwargs.get("enable_fsdp2_prefetch", True),
            kwargs.get("fsdp2_backward_prefetch_depth", 2),
            kwargs.get("fsdp2_forward_prefetch_depth", 1),
        )

        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


class HunyuanParallelizationStrategy(ParallelizationStrategy):
    """Parallelization strategy for Hunyuan-style transformer modules used in HunyuanVideo."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = True,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        # Mixed precision default like Default strategy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        # Apply activation checkpointing to transformer blocks if requested
        if activation_checkpointing:
            for idx in range(len(model.transformer_blocks)):
                model.transformer_blocks[idx] = checkpoint_wrapper(
                    model.transformer_blocks[idx],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        if reapply_trainability is not None:
            reapply_trainability(model)

        # Apply FSDP sharding recursively and to root
        apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            kwargs.get("enable_fsdp2_prefetch", True),
            kwargs.get("fsdp2_backward_prefetch_depth", 2),
            kwargs.get("fsdp2_forward_prefetch_depth", 1),
        )

        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


class LTX2ParallelizationStrategy(HunyuanParallelizationStrategy):
    """Parallelization strategy for the LTX-2 video+audio transformer.

    ``LTX2VideoTransformer3DModel`` exposes its layers as ``transformer_blocks``
    but names the attention/FFN submodules ``attn1``/``attn2``/``ff``, which the
    Default strategy's submodule-level activation checkpointing does not
    recognize — leaving attention and MLP activations un-checkpointed and OOM-ing
    on the long combined video+audio token sequence. Wrapping each whole block
    (as the HunyuanVideo strategy does) restores the expected memory profile.
    """


def _unwrap_qwen_checkpointed_block(block: nn.Module) -> nn.Module:
    """Return the Qwen transformer block held by a checkpoint wrapper."""
    if isinstance(block, CheckpointWrapper):
        return block._checkpoint_wrapped_module
    return block


def _validate_qwen_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Validate the upstream Qwen dual-stream block structure.

    Args:
        model: Upstream Qwen image transformer whose ``transformer_blocks``
            container must hold dual-stream image/text blocks.

    Returns:
        The model's ``transformer_blocks`` ModuleList. Returned modules alias
        the model-owned blocks.
    """
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Qwen image FSDP2 requires a non-empty transformer_blocks nn.ModuleList")

    required_branches = ("attn", "img_mlp", "txt_mlp")
    for index, wrapped_block in enumerate(blocks):
        block = _unwrap_qwen_checkpointed_block(wrapped_block)
        missing = [name for name in required_branches if not isinstance(getattr(block, name, None), nn.Module)]
        if missing:
            raise TypeError(
                f"Qwen transformer_blocks[{index}] is missing required dual-stream modules: {', '.join(missing)}"
            )
    return blocks


def _apply_qwen_block_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint complete Qwen blocks, including both image and text MLPs.

    Args:
        model: Upstream Qwen image transformer. Each block consumes image
            hidden states with shape [batch, image_tokens, hidden], text hidden
            states with shape [batch, text_tokens, hidden], a text mask with
            shape [batch, text_tokens], timestep embeddings with shape [batch,
            hidden], and rotary tensors whose leading layout is owned by
            Diffusers. Its block outputs preserve the image/text layouts.
    """
    blocks = _validate_qwen_transformer_blocks(model)
    wrapped_count = 0
    for index, block in enumerate(blocks):
        if isinstance(block, CheckpointWrapper):
            continue
        blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        wrapped_count += 1
    logger.info("Applied whole-block activation checkpointing to %d Qwen image transformer blocks", wrapped_count)


class QwenImageEditParallelizationStrategy(DefaultParallelizationStrategy):
    """Shard upstream Qwen image transformer blocks as complete FSDP2 units.

    Applies whole-block activation checkpointing that covers the attention,
    image-MLP, and text-MLP branches of each dual-stream block, then delegates
    TP planning and FSDP2 sharding to the default strategy.
    """

    def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        """Apply Qwen block checkpointing followed by the standard FSDP2 flow.

        Args:
            model: Upstream Qwen image transformer. Tensor layouts are unchanged
                by sharding; each dual-stream block consumes image tensors of
                shape [batch, image_tokens, hidden] and text tensors of shape
                [batch, text_tokens, hidden].
            *args: Positional arguments forwarded to the default strategy.
            **kwargs: Keyword arguments accepted by
                :meth:`DefaultParallelizationStrategy.parallelize`.

        Returns:
            The same upstream model with its parameters represented by FSDP2
            DTensors on distributed runs. Global tensor shapes and upstream
            Diffusers state-dict keys are preserved.
        """
        _validate_qwen_transformer_blocks(model)
        activation_checkpointing = kwargs.get("activation_checkpointing", False)
        selective_checkpointing = (
            isinstance(activation_checkpointing, str)
            and activation_checkpointing.lower().replace("-", "_") == "selective"
        )
        if activation_checkpointing and not selective_checkpointing:
            _apply_qwen_block_activation_checkpointing(model)
            kwargs["activation_checkpointing"] = False

        return super().parallelize(model, *args, **kwargs)


DIFFUSERS_PARALLEL_SPECS: dict[str, ParallelSpec] = {
    "WanTransformer3DModel": ParallelSpec(strategy=WanParallelizationStrategy()),
    "HunyuanVideo15Transformer3DModel": ParallelSpec(strategy=HunyuanParallelizationStrategy()),
    "LTX2VideoTransformer3DModel": ParallelSpec(strategy=LTX2ParallelizationStrategy()),
    "QwenImageTransformer2DModel": ParallelSpec(strategy=QwenImageEditParallelizationStrategy()),
}


def attach_parallel_spec(module: nn.Module) -> nn.Module:
    """Bind the known ``ParallelSpec`` for a diffusers transformer onto ``module``'s class.

    Mirrors the HF bridge: ``module.__class__`` becomes a dynamic subclass carrying
    ``parallel_spec``, so ``query_parallel_spec`` finds it without touching the upstream class.
    Modules without a known contract are returned unchanged.
    """
    cls = type(module)
    if hasattr(cls, "parallel_spec"):
        return module
    for base in cls.__mro__:
        spec = DIFFUSERS_PARALLEL_SPECS.get(base.__name__)
        if spec is not None:
            namespace = {"parallel_spec": spec, "__module__": cls.__module__, "__qualname__": cls.__qualname__}
            module.__class__ = type(cls.__name__, (cls,), namespace)
            break
    return module
