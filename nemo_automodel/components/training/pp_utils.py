# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import functools
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.distributed.init_utils import get_rank_safe
from nemo_automodel.components.distributed.nvfsdp import NVFSDPManager
from nemo_automodel.components.distributed.pipeline import pipeline_model_hf
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.rng import StatefulRNG

logger = logging.getLogger(__name__)


@dataclass
class PipelineInfo:
    """Holds information about the pipeline parallel configuration and state.

    Attributes:
        enabled (bool): Whether pipeline parallelism is enabled.
        config_prefix (str): The dotted prefix for accessing PP config (e.g., "distributed.pipeline_parallel").
        cfg (Optional[Any]): Reference to the full ConfigNode for dotted access.
        schedule (Optional[object]): The pipeline schedule object (None if PP not enabled).
        has_first_stage (bool): Whether this rank has the first pipeline stage.
        has_last_stage (bool): Whether this rank has the last pipeline stage.
        model_parts (Optional[list[nn.Module]]): List of model parts for this rank (None if PP not enabled).
    """

    enabled: bool = False
    cfg: Optional[Any] = None
    schedule: Optional[object] = None
    has_first_stage: bool = True
    has_last_stage: bool = True
    model_parts: Optional[list[nn.Module]] = None

    def build_config_dict(self) -> dict[str, Any]:
        """Build a configuration dictionary from the ConfigNode for functions that need it.

        Returns:
            Dictionary with all PP configuration values
        """
        if not self.enabled or self.cfg is None:
            return {}

        return {
            "pp_size": self.cfg.get("pp_size", 1),
            "pp_schedule": self.cfg.get("pp_schedule", "1f1b"),
            "layers_per_stage": self.cfg.get("layers_per_stage", None),
            "init_weights": self.cfg.get("init_weights", True),
            "empty_weights": self.cfg.get("empty_weights", False),
            "input_weight": self.cfg.get("input_weight", 2),
            "output_weight": self.cfg.get("output_weight", 1),
            "module_fqns_per_model_part": self.cfg.get("module_fqns_per_model_part", None),
        }


def validate_model_for_pipeline_support(model: nn.Module, model_wrapper: Optional[Any]) -> None:
    """
    Validate if a model configuration is compatible with pipeline parallel training.

    Args:
        cfg_model: Model configuration object

    Raises:
        ValueError: If the model configuration is incompatible with pipeline parallelism
    """
    # Get the model config to check compatibility
    if hasattr(model.config, "pretrained_model_name_or_path"):
        model_name = model.config.pretrained_model_name_or_path
    else:
        model_name = "Unknown model"

    config = model.config
    # List of known incompatibilities
    incompatibilities = []

    if isinstance(model_wrapper, NVFSDPManager):
        incompatibilities.append(
            "nvFSDP is not currentlysupported with pipeline parallelism. Please use FSDP2 instead."
        )

    # Check if we can access the config to validate
    try:
        if getattr(config, "tie_word_embeddings", False):
            incompatibilities.append(
                "tie_word_embeddings=True is not supported with pipeline parallelism. "
                "The input and output embeddings must be separate for proper stage assignment."
            )

        # Check for models that use shared embeddings/parameters across layers
        model_type = getattr(config, "model_type", "")
        if model_type in ["albert", "reformer"]:
            incompatibilities.append(
                f"Model type '{model_type}' uses parameter sharing across layers, "
                "which is incompatible with pipeline parallelism."
            )

        # Check if model has cross-attention (encoder-decoder models)
        if getattr(config, "is_encoder_decoder", False):
            incompatibilities.append(
                "Encoder-decoder models are not yet supported with pipeline parallelism "
                "due to cross-attention dependencies between encoder and decoder."
            )

        # Additional architecture-specific checks
        if hasattr(config, "use_cache") and getattr(config, "gradient_checkpointing", False):
            logger.warning(
                "Using both gradient checkpointing and KV-cache with pipeline parallelism "
                "may lead to unexpected behavior. Consider disabling one of them."
            )

    except Exception as e:
        logger.warning(
            f"Could not validate model configuration for pipeline parallel compatibility: {e}. "
            "Proceeding without validation."
        )
        return

    # Raise error if any incompatibilities found
    if incompatibilities:
        error_msg = f"Model '{model_name}' is not compatible with pipeline parallelism:\n\n"
        for i, issue in enumerate(incompatibilities, 1):
            error_msg += f"{i}. {issue}\n"
        error_msg += (
            "\nTo use pipeline parallelism, please either:\n"
            "- Choose a different model\n"
            "- Modify the model configuration (if possible)\n"
            "- Disable pipeline parallelism"
        )
        raise ValueError(error_msg)


def pipeline_parallel_forward_backward_step(
    pp_schedule: _PipelineSchedule,
    pp_has_first_stage: bool,
    pp_has_last_stage: bool,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    train_ctx: Callable,
    device: torch.device,
) -> torch.Tensor:
    """
    Execute pipeline parallel forward and backward pass.

    Args:
        pp_schedule: Pipeline schedule object
        pp_has_first_stage: Whether this rank has the first pipeline stage
        pp_has_last_stage: Whether this rank has the last pipeline stage
        batch: Input batch dictionary
        labels: Target labels
        loss_mask: Optional loss mask (1 for tokens to include, 0 for tokens to ignore)
        train_ctx: Training context manager
        device: Target device

    Returns:
        Loss tensor (averaged across microbatches for last stage, 0 for others)
    """
    with train_ctx():
        # Create a list to collect losses from pipeline stages
        losses = [] if pp_has_last_stage else None

        # For the last stage, we need to handle both labels and mask
        if pp_has_last_stage:
            # Apply mask to labels before passing to pipeline
            masked_labels = labels.clone()
            if loss_mask is not None:
                # Set positions where mask is 0 to -100 (ignore_index)
                masked_labels[loss_mask == 0] = -100
            targets = masked_labels
        else:
            targets = None

        input_ids = batch.pop("input_ids")
        # Run pipeline schedule step
        if pp_has_first_stage:
            # First stage receives input
            pp_schedule.step(input_ids, target=targets, losses=losses, **batch)
        else:
            # Non-first stages don't receive input
            pp_schedule.step(target=targets, losses=losses, **batch)

    # Accumulate losses across pipeline microbatches
    if pp_has_last_stage:
        loss = torch.sum(torch.stack(losses))
    else:
        # Non-last stages don't compute loss
        loss = torch.tensor(0.0, device=device)

    return loss


def check_pipeline_parallel_validation_support(pp_schedule: Optional[_PipelineSchedule]) -> bool:
    """
    Check if validation is supported with pipeline parallelism.

    Args:
        pp_schedule: Pipeline schedule object (None if PP is not enabled)

    Returns:
        True if validation can proceed, False otherwise
    """
    if pp_schedule is not None:
        logger.warning("Pipeline parallel validation is not yet supported. Skipping validation.")
        return False
    return True


def initialize_meta_model(cfg_model: Any, use_hf_fa2: bool = False) -> nn.Module:
    """
    Initialize a model on meta device to save memory during pipeline parallel setup.

    Args:
        cfg_model: Model configuration object
        use_hf_fa2: Whether to use HuggingFace's flash_attention_2

    Returns:
        Model initialized on meta device
    """
    kwargs = {}
    if use_hf_fa2:
        kwargs["attn_implementation"] = "flash_attention_2"
        logger.warning(
            "Packed sequence is supported only with Flash Attention. "
            "Setting model's attn_implementation to flash_attention_2"
        )

    # Initialize model on meta device
    with torch.device("meta"):
        # kwargs["device_map"] = "meta"
        model = cfg_model.instantiate(**kwargs)

    return model


def materialize_meta_model(
    model_parts: list[nn.Module],
    device: torch.device,
    init_weights: bool = True,
    empty_weights: bool = False,
) -> None:
    """
    Materialize model parts from meta device to actual device.

    Args:
        model_parts: List of model parts to materialize
        device: Target device
        init_weights: Whether to call init_weights on the model
        empty_weights: Whether to use empty weights (zeros) instead of init_weights
    """
    for mp in model_parts:
        logger.debug(f"Rank {get_rank_safe()}: Materializing model part {mp}")
        # Move from meta to device with empty weights
        mp.to_empty(device=device)

        if not empty_weights and init_weights:
            # Initialize weights using model's init_weights method
            with torch.no_grad():
                mp.apply(mp._init_weights)

        mp.bfloat16()
        mp.train()
        logger.debug(
            f"Rank {get_rank_safe()}: Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )
        # Move to target device
        mp.to(device)


def parallelize_for_pp(
    model: nn.Module,
    *,
    world_mesh: DeviceMesh,
    moe_mesh: Optional[DeviceMesh] = None,
    pp_enabled: bool = False,
    dp_axis_names: Union[tuple[str, ...], str] = ("data_parallel",),
    cp_axis_name: Optional[str] = None,
    tp_axis_name: Optional[str] = None,
    ep_axis_name: Optional[str] = None,
    ep_shard_axis_names: Optional[tuple[str, ...]] = None,
    model_wrapper: Optional[Any] = None,
) -> nn.Module:
    if model_wrapper is not None:
        if callable(getattr(model_wrapper, "parallelize", None)):
            model = model_wrapper.parallelize(model)
    return model


def build_model_and_optimizer_for_pp(
    device: torch.device,
    cfg_model: Any,
    cfg_opt: Any,
    use_hf_fa2: bool,
    cfg_peft: Optional[Any],
    model_wrapper: Optional[Any],
    seed: int,
    pp_config: dict[str, Any],
    device_mesh: DeviceMesh,
    tp_size: int = 1,
    freeze_embeddings: bool = False,
) -> tuple[list[nn.Module], optim.Optimizer, _PipelineSchedule, tuple[bool, bool]]:
    """
    Build and initialize a model and optimizer for pipeline parallelism.

    Args:
        device: The target device
        cfg_model: Configuration for model instantiation
        cfg_opt: Configuration for optimizer instantiation
        use_hf_fa2: Whether to use HF's flash_attention_2
        cfg_peft: Configuration for PEFT
        model_wrapper: Optional parallelism wrapper
        seed: Random seed
        pp_config: Pipeline parallel configuration containing pp_size, pp_schedule, etc.
        device_mesh: Device mesh for distributed training
        tp_size: Tensor parallel size
        freeze_embeddings: Whether to freeze embeddings

    Returns:
        Tuple of (model_parts, optimizer, pp_schedule, (pp_has_first_stage, pp_has_last_stage))
    """
    # Extract PP configuration
    pp_schedule = pp_config.get("pp_schedule", "1f1b")
    microbatch_size = pp_config.get("microbatch_size", 1)
    local_batch_size = pp_config.get("local_batch_size", 1)

    with StatefulRNG(seed=seed, ranked=True):
        # Initialize model on meta device
        model = initialize_meta_model(cfg_model, use_hf_fa2)

        if freeze_embeddings:
            logger.info("Marking embeddings for freezing")
            for m in model.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.requires_grad_(False)

        # Apply PEFT if configured
        if cfg_peft is not None:
            apply_lora_to_linear_modules(model, cfg_peft)

    # Validate model compatibility with pipeline parallelism
    validate_model_for_pipeline_support(model, model_wrapper)

    # Build loss function for PP (will be passed to pipeline_model_hf)
    loss_fn = build_loss_fn_for_pp(pp_config.get("loss_fn", None))

    # Apply pipeline model splitting
    pp_schedule_obj, model_parts, pp_has_first_stage, pp_has_last_stage = pipeline_model_hf(
        model,
        world_mesh=device_mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=(
            ("dp_replicate", "dp_shard")
            if "dp_replicate" in device_mesh.mesh_dim_names and "dp_shard" in device_mesh.mesh_dim_names
            else ("dp_shard",)
        ),
        cp_axis_name="cp" if "cp" in device_mesh.mesh_dim_names else None,
        tp_axis_name="tp" if "tp" in device_mesh.mesh_dim_names else None,
        ep_axis_name=None,
        layers_per_stage=pp_config.get("layers_per_stage", None),
        pipeline_parallel_schedule_csv=None,
        pipeline_parallel_schedule=pp_schedule,
        microbatch_size=microbatch_size,
        local_batch_size=local_batch_size,
        device=device,
        loss_fn=loss_fn,
        parallelize_fn=functools.partial(parallelize_for_pp, model_wrapper=model_wrapper),
        module_fqns_per_model_part=pp_config.get("module_fqns_per_model_part", None),
    )

    # Materialize model parts from meta device
    materialize_meta_model(
        model_parts,
        device=device,
        init_weights=pp_config.get("init_weights", True),
        empty_weights=pp_config.get("empty_weights", False),
    )

    # Create optimizer for all model parts
    trainable_params = []
    for i, model_part in enumerate(model_parts):
        trainable_params.append(
            {
                "params": list(filter(lambda x: x.requires_grad, model_part.parameters())),
                "name": f"rank_{get_rank_safe()}_model_part_{i}",
            }
        )
    assert len(trainable_params) > 0, "trainable_params cannot be empty"
    if tp_size > 1:
        cfg_opt.foreach = False
    optimizer = cfg_opt.instantiate(params=trainable_params)

    return model_parts, optimizer, pp_schedule_obj, (pp_has_first_stage, pp_has_last_stage)


def build_loss_fn_for_pp(loss_fn_config: Optional[Any] = None) -> Callable:
    """
    Build a loss function suitable for pipeline parallelism.

    Note: Loss masking is handled by setting labels to ignore_index (-100) in
    pipeline_parallel_forward_backward_step before passing to the loss function.

    Args:
        loss_fn_config: Optional loss function configuration

    Returns:
        Loss function callable for pipeline parallelism
    """
    ignore_index = -100  # Default PyTorch ignore index

    # Check if a specific loss function was configured
    if loss_fn_config is not None and hasattr(loss_fn_config, "instantiate"):
        loss_fn_instance = loss_fn_config.instantiate()

        # Get ignore_index from the configured loss if available
        ignore_index = getattr(loss_fn_instance, "ignore_index", -100)

        # Warn if using FusedLinearCrossEntropy
        if isinstance(loss_fn_instance, FusedLinearCrossEntropy):
            logger.warning(
                "FusedLinearCrossEntropy is currently not supported with pipeline parallelism, "
                "replacing with standard cross entropy loss."
            )

        return loss_fn_instance

    # Return a simple cross entropy loss function that respects ignore_index
    # The masking is already handled by setting labels to ignore_index
    def pp_ce_loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.view(-1, pred.size(-1)), labels.view(-1), ignore_index=ignore_index
        )

    return pp_ce_loss_fn


@torch.no_grad()
def rescale_gradients_for_pp(
    pp_schedule: _PipelineSchedule,
    num_tokens_for_grad_scaling: torch.Tensor,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Rescale gradients for pipeline parallel models.

    This function rescales gradients by the total number of tokens used in the loss calculation
    across the data parallel group. It uses the stage's scale_grads method to apply the scaling.

    Args:
        pp_schedule: The pipeline schedule object containing the stages
        num_tokens_for_grad_scaling: The number of tokens used for loss calculation
        dp_group: The data parallel process group for all-reduce
    """
    # Calculate the scaling factor similar to rescale_gradients
    num_tokens_for_grad_scaling = num_tokens_for_grad_scaling.clone().detach().item()

    # Access stages based on schedule type
    stages = []
    if hasattr(pp_schedule, "_stages"):
        # PipelineScheduleMulti
        stages = pp_schedule._stages
    elif hasattr(pp_schedule, "_stage"):
        # PipelineScheduleSingle
        stages = [pp_schedule._stage]
    else:
        logger.warning("Pipeline schedule does not have _stage or _stages attribute. Cannot rescale gradients.")
        return

    # Apply gradient scaling to each stage
    for stage in stages:
        if hasattr(stage, "scale_grads"):
            stage.scale_grads(num_tokens_for_grad_scaling)
        else:
            logger.warning(f"Stage {getattr(stage, 'stage_index', 'unknown')} does not have scale_grads method")


# Adapted from torchtitan
@torch.no_grad()
def clip_grad_norm_for_pp(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_dense_params_mesh_ndim: int | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: Pipeline Parallel device mesh. If not None, will reduce gradient norm across PP stages.
        ep_dense_params_mesh_ndim: Mesh ndim of the dense params when EP is used. If EP is not used,
            set it to ``None``.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if ep_dense_params_mesh_ndim is not None:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_dense_params_mesh_ndim,
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
    dense_params_mesh_ndim: int,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if p.device_mesh.ndim == dense_params_mesh_ndim:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
        else:
            ep_params.append(p)
            ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(ep_grads, norm_type, error_if_nonfinite, foreach).full_tensor()
    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    ).full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm
