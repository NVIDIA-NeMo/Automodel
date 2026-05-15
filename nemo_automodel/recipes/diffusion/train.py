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

from __future__ import annotations

import inspect
import logging
import os
import time
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from huggingface_hub.constants import HF_HUB_CACHE
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline, create_adapter
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import build_distributed, build_wandb
from nemo_automodel.shared.import_utils import safe_import_from

_OPTIMIZER_DEFAULT_TARGET = "torch.optim.AdamW"
_TORCH_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _get_diffusion_microbatch_size(batch: Dict[str, Any]) -> int:
    """Return the number of samples in one local diffusion micro-batch."""
    for key in ("video_latents", "image_latents", "latents", "text_embeddings", "text_embeddings_2"):
        value = batch.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])
    return 0


def _count_local_batch_group_samples(batch_group: list[Dict[str, Any]]) -> int:
    """Count local samples processed by one optimizer step."""
    return sum(_get_diffusion_microbatch_size(batch) for batch in batch_group)


def _calculate_throughput_metrics(
    *,
    elapsed_seconds: float,
    optimizer_steps: int,
    global_samples: int,
    world_size: int,
) -> Dict[str, float]:
    """Calculate directly measured training throughput metrics."""
    elapsed_seconds = max(float(elapsed_seconds), 1e-12)
    optimizer_steps = max(int(optimizer_steps), 0)
    global_samples = max(int(global_samples), 0)
    world_size = max(int(world_size), 1)
    nonzero_steps = max(optimizer_steps, 1)

    samples_per_sec = global_samples / elapsed_seconds
    return {
        "step_time": elapsed_seconds / nonzero_steps,
        "optimizer_steps_per_sec": optimizer_steps / elapsed_seconds,
        "samples_per_sec": samples_per_sec,
        "samples_per_sec_per_gpu": samples_per_sec / world_size,
        "samples_per_step": global_samples / nonzero_steps,
        "log_window_seconds": elapsed_seconds,
        "log_window_steps": float(optimizer_steps),
        "log_window_samples": float(global_samples),
    }


def _normalize_optimizer_value(value: Any) -> Any:
    """Convert CLI-friendly optimizer scalar values into Python objects."""
    if isinstance(value, str):
        normalized = value.removeprefix("torch.").lower()
        return _TORCH_DTYPE_ALIASES.get(normalized, value)
    return value


def _get_optimizer_target_name(target: Any) -> str:
    """Return a stable display name for an optimizer target."""
    if isinstance(target, str):
        return target
    module_name = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if module_name and qualname:
        return f"{module_name}.{qualname}"
    return repr(target)


def _resolve_optimizer_class(target: Any) -> Any:
    """Resolve an optimizer class from a fully qualified `_target_` string."""
    if not isinstance(target, str):
        if callable(target):
            return target
        raise ValueError(f"Optimizer target must be a fully qualified import path or callable, got {target!r}.")

    if target == _OPTIMIZER_DEFAULT_TARGET:
        return torch.optim.AdamW

    module_name, _, symbol_name = target.rpartition(".")
    if not module_name or not symbol_name:
        raise ValueError(
            f"Optimizer target must be a fully qualified import path, got {target!r}. "
            f"Example: {_OPTIMIZER_DEFAULT_TARGET!r}."
        )

    available, optimizer_cls = safe_import_from(
        module_name,
        symbol_name,
        msg=f"Optimizer target {target!r} could not be imported",
    )
    if not available:
        raise ImportError(f"Optimizer target {target!r} could not be imported")
    return optimizer_cls


def _filter_optimizer_kwargs(target: str, optimizer_cls: Any, optimizer_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs unsupported by an opt-in optimizer target."""
    try:
        signature = inspect.signature(optimizer_cls)
    except (TypeError, ValueError):
        logging.info("[INFO] Could not inspect optimizer target %s; passing all optimizer kwargs", target)
        return optimizer_kwargs

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return optimizer_kwargs

    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    accepted.discard("params")

    filtered_kwargs = {key: value for key, value in optimizer_kwargs.items() if key in accepted}
    ignored_kwargs = sorted(set(optimizer_kwargs) - set(filtered_kwargs))
    if ignored_kwargs:
        logging.info("[INFO] Optimizer target %s does not accept kwargs %s; ignoring them", target, ignored_kwargs)
    return filtered_kwargs


def _build_transformer_engine_fp8_recipe(
    recipe_name: str,
    *,
    amax_history_len: int,
    amax_compute_algo: str,
) -> Any:
    """Build a Transformer Engine FP8 recipe from CLI-friendly config values."""
    normalized_recipe_name = recipe_name.replace("-", "_").lower()
    if normalized_recipe_name in {"delayed", "delayed_scaling"}:
        available, delayed_scaling = safe_import_from(
            "transformer_engine.common.recipe",
            "DelayedScaling",
            msg="model.transformer_engine_fp8=true requires Transformer Engine DelayedScaling",
        )
        if not available:
            raise ImportError("model.transformer_engine_fp8=true requires Transformer Engine DelayedScaling")
        return delayed_scaling(amax_history_len=amax_history_len, amax_compute_algo=amax_compute_algo)

    if normalized_recipe_name in {"current", "current_scaling"}:
        available, current_scaling = safe_import_from(
            "transformer_engine.common.recipe",
            "Float8CurrentScaling",
            msg="model.transformer_engine_fp8_recipe=current requires Transformer Engine Float8CurrentScaling",
        )
        if not available:
            raise ImportError("model.transformer_engine_fp8_recipe=current requires Float8CurrentScaling")
        return current_scaling()

    if normalized_recipe_name in {"mxfp8", "mx", "mx_fp8"}:
        available, mxfp8_block_scaling = safe_import_from(
            "transformer_engine.common.recipe",
            "MXFP8BlockScaling",
            msg="model.transformer_engine_fp8_recipe=mxfp8 requires Transformer Engine MXFP8BlockScaling",
        )
        if not available:
            raise ImportError("model.transformer_engine_fp8_recipe=mxfp8 requires MXFP8BlockScaling")
        return mxfp8_block_scaling()

    raise ValueError(
        f"model.transformer_engine_fp8_recipe must be one of 'delayed', 'current', or 'mxfp8', got {recipe_name!r}"
    )


def _resolve_transformer_engine_autocast() -> Any:
    """Resolve Transformer Engine's quantization autocast context manager."""
    available, te_autocast = safe_import_from(
        "transformer_engine.pytorch.quantization",
        "autocast",
        msg="model.transformer_engine_fp8=true requires transformer_engine.pytorch.quantization.autocast",
    )
    if not available:
        raise ImportError("model.transformer_engine_fp8=true requires transformer_engine.pytorch.quantization.autocast")
    return te_autocast


def build_model_and_optimizer(
    *,
    model_id: str,
    finetune_mode: bool,
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_cfg: Optional[Dict[str, Any]] = None,
    ddp_cfg: Optional[Dict[str, Any]] = None,
    attention_backend: Optional[str] = None,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
    transformer_engine_linear: bool = False,
    transformer_engine_fp8_safe_only: bool = False,
    fuse_qkv_projections: bool = False,
    compact_fused_qkv_projections: bool = False,
    torchao_float8_linear: bool = False,
    torchao_float8_fsdp_all_gather: bool = True,
    torchao_float8_recipe_name: Optional[str] = None,
    torchao_float8_force_recompute_weight_in_bwd: bool = False,
    pipeline_spec: Optional[Dict[str, Any]] = None,
    peft_cfg=None,
    model_type=None,
) -> tuple[NeMoAutoDiffusionPipeline, torch.optim.Optimizer, Any]:
    """Build the diffusion model, parallel scheme, and optimizer.

    Args:
        model_id: Pretrained model name or path.
        finetune_mode: Whether to load for finetuning (True) or pretraining (False).
        learning_rate: Learning rate for optimizer.
        device: Target device.
        dtype: Model dtype.
        cpu_offload: Whether to enable CPU offload (FSDP only).
        fsdp_cfg: FSDP configuration dict. Mutually exclusive with ddp_cfg.
        ddp_cfg: DDP configuration dict. Mutually exclusive with fsdp_cfg.
        attention_backend: Optional attention backend override.
        optimizer_cfg: Optional optimizer configuration.
        transformer_engine_linear: Whether to replace transformer torch.nn.Linear modules with Transformer Engine Linear.
        transformer_engine_fp8_safe_only: Whether to skip TE conversion for known FP8-incompatible modules.
        fuse_qkv_projections: Whether to call Diffusers QKV projection fusion on the transformer before FSDP.
        compact_fused_qkv_projections: Whether to remove original projection modules after QKV fusion.
        torchao_float8_linear: Whether to replace transformer torch.nn.Linear modules with torchao Float8Linear.
        torchao_float8_fsdp_all_gather: Whether torchao Float8Linear should use FP8 FSDP2 all-gather.
        torchao_float8_recipe_name: Optional torchao Float8Linear recipe name.
        torchao_float8_force_recompute_weight_in_bwd: Whether to recompute FP8 weight in backward.
        pipeline_spec: Pipeline specification for pretraining (from_config).
            Required when finetune_mode is False. Should contain:
            - transformer_cls: str (e.g., "WanTransformer3DModel", "FluxTransformer2DModel")
            - subfolder: str (e.g., "transformer")
            - Optional: pipeline_cls, load_full_pipeline
        peft_cfg: PeftConfig instance or None. When provided, only LoRA params
            are trained; base weights are frozen and sharded by FSDP2 for memory.
        model_type: "flux" | "wan" | "hunyuan". Required when peft_cfg is provided.

    Returns:
        Tuple of (pipeline, optimizer, device_mesh or None).

    Raises:
        ValueError: If both fsdp_cfg and ddp_cfg are provided.
        ValueError: If finetune_mode is False and pipeline_spec is not provided.
    """
    # Validate mutually exclusive configs
    if fsdp_cfg is not None and ddp_cfg is not None:
        raise ValueError(
            "Cannot specify both 'fsdp' and 'ddp' configurations. "
            "Please provide only one distributed training strategy."
        )

    logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

    if not dist.is_initialized():
        logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

    world_size = dist.get_world_size() if dist.is_initialized() else 1

    lora_enabled = peft_cfg is not None
    # param_dtype=None when LoRA: FSDP2 does not cast any parameter.
    # bf16 base weights stay bf16 (loaded dtype).
    # bf16 LoRA weights stay bf16 (set via peft_cfg.lora_dtype in pipeline).
    # param_dtype=dtype when full fine-tune: FSDP2 casts everything to dtype (bf16).
    param_dtype = None if lora_enabled else dtype

    # Build manager args based on which config is provided
    if ddp_cfg is not None:
        # DDP configuration
        logging.info("[INFO] Using DDP (DistributedDataParallel) for training")
        manager_args: Dict[str, Any] = {
            "_manager_type": "ddp",
            "backend": ddp_cfg.get("backend", "nccl"),
            "world_size": world_size,
            "activation_checkpointing": ddp_cfg.get("activation_checkpointing", False),
        }
    else:
        # FSDP configuration (default)
        fsdp_cfg = fsdp_cfg or {}
        logging.info("[INFO] Using FSDP2 (Fully Sharded Data Parallel) for training")

        dp_size = fsdp_cfg.get("dp_size")
        tp_size = fsdp_cfg.get("tp_size", 1)
        cp_size = fsdp_cfg.get("cp_size", 1)
        pp_size = fsdp_cfg.get("pp_size", 1)

        if dp_size is None:
            denom = tp_size * cp_size * pp_size
            if world_size % denom != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"tp_size*cp_size*pp_size ({tp_size}*{cp_size}*{pp_size}={denom})"
                )
            dp_size = world_size // denom

        manager_args: Dict[str, Any] = {
            "_manager_type": "fsdp2",
            "dp_size": dp_size,
            "dp_replicate_size": fsdp_cfg.get("dp_replicate_size", None),
            "tp_size": tp_size,
            "cp_size": cp_size,
            "pp_size": pp_size,
            "backend": "nccl",
            "world_size": world_size,
            "use_hf_tp_plan": fsdp_cfg.get("use_hf_tp_plan", False),
            "sequence_parallel": fsdp_cfg.get("sequence_parallel", False),
            "tp_plan": fsdp_cfg.get("tp_plan", None),
            "patch_is_packed_sequence": fsdp_cfg.get("patch_is_packed_sequence", False),
            "activation_checkpointing": fsdp_cfg.get("activation_checkpointing", True),
            "defer_fsdp_grad_sync": fsdp_cfg.get("defer_fsdp_grad_sync", True),
            "enable_async_tensor_parallel": fsdp_cfg.get("enable_async_tensor_parallel", False),
            "enable_compile": fsdp_cfg.get("enable_compile", False),
            "enable_fsdp2_prefetch": fsdp_cfg.get("enable_fsdp2_prefetch", True),
            "fsdp2_backward_prefetch_depth": fsdp_cfg.get("fsdp2_backward_prefetch_depth", 2),
            "fsdp2_forward_prefetch_depth": fsdp_cfg.get("fsdp2_forward_prefetch_depth", 1),
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=torch.float32,
                output_dtype=dtype,
            ),
        }

    parallel_scheme = {"transformer": manager_args}

    if finetune_mode:
        # Finetuning: load from pretrained weights
        logging.info("[INFO] Loading pretrained model for finetuning")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=True,
            low_cpu_mem_usage=True,
            peft_cfg=peft_cfg,
            model_type=model_type,
            transformer_engine_linear=transformer_engine_linear,
            transformer_engine_fp8_safe_only=transformer_engine_fp8_safe_only,
            fuse_qkv_projections=fuse_qkv_projections,
            compact_fused_qkv_projections=compact_fused_qkv_projections,
            torchao_float8_linear=torchao_float8_linear,
            torchao_float8_fsdp_all_gather=torchao_float8_fsdp_all_gather,
            torchao_float8_recipe_name=torchao_float8_recipe_name,
            torchao_float8_force_recompute_weight_in_bwd=torchao_float8_force_recompute_weight_in_bwd,
        )
    else:
        # Pretraining: initialize with random weights using pipeline_spec
        if pipeline_spec is None:
            raise ValueError(
                "pipeline_spec is required for pretraining (finetune_mode=False). "
                "Please provide pipeline_spec in your YAML config with at least:\n"
                "  pipeline_spec:\n"
                "    transformer_cls: 'WanTransformer3DModel'  # or 'FluxTransformer2DModel', etc.\n"
                "    subfolder: 'transformer'"
            )
        logging.info("[INFO] Initializing model with random weights for pretraining")
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_config(
            model_id,
            pipeline_spec=pipeline_spec,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            transformer_engine_linear=transformer_engine_linear,
            transformer_engine_fp8_safe_only=transformer_engine_fp8_safe_only,
            fuse_qkv_projections=fuse_qkv_projections,
            compact_fused_qkv_projections=compact_fused_qkv_projections,
            torchao_float8_linear=torchao_float8_linear,
            torchao_float8_fsdp_all_gather=torchao_float8_fsdp_all_gather,
            torchao_float8_recipe_name=torchao_float8_recipe_name,
            torchao_float8_force_recompute_weight_in_bwd=torchao_float8_force_recompute_weight_in_bwd,
        )
    fsdp2_manager = created_managers["transformer"]
    transformer_module = pipe.transformer
    transformer_module_for_attrs = getattr(transformer_module, "module", transformer_module)
    if attention_backend is not None:
        logging.info(f"[INFO] Setting attention backend to {attention_backend}")
        transformer_module_for_attrs.set_attention_backend(attention_backend)

    if lora_enabled:
        # Collect lora_params AFTER FSDP2 wrapping from the live wrapped module.
        # Pre-FSDP2 refs (pipe._lora_params) are stale after fully_shard() —
        # FSDP2 replaces parameter storage, so AdamW with stale refs never commits
        # updates to the actual sharded parameters. Mirrors the LLM pattern in
        # nemo_automodel/recipes/llm/train_ft.py line 313.
        trainable_params = [p for n, p in transformer_module.named_parameters() if "lora_" in n and p.requires_grad]
        if not trainable_params:
            raise RuntimeError(
                "peft_cfg is set but no LoRA params found. "
                "Check that peft.target_modules match module names in the transformer."
            )
        logging.info(
            "[LoRA] Optimizer: %d param tensors, %s elements",
            len(trainable_params),
            f"{sum(p.numel() for p in trainable_params):,}",
        )
    else:
        trainable_params = [p for p in transformer_module.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found in transformer module!")

    optimizer_cfg = optimizer_cfg or {}
    optimizer_cfg = optimizer_cfg.to_dict() if hasattr(optimizer_cfg, "to_dict") else dict(optimizer_cfg)
    optimizer_target = optimizer_cfg.get("_target_", _OPTIMIZER_DEFAULT_TARGET)
    optimizer_target_name = _get_optimizer_target_name(optimizer_target)
    optimizer_cls = _resolve_optimizer_class(optimizer_target)

    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)) or (0.9, 0.999))
    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": optimizer_cfg.get("eps", 1e-8),
        "amsgrad": optimizer_cfg.get("amsgrad", False),
    }
    for key, value in optimizer_cfg.items():
        if key in {"_target_", "lr", "learning_rate", "weight_decay", "betas", "eps", "amsgrad"}:
            continue
        if value is not None:
            optimizer_kwargs[key] = _normalize_optimizer_value(value)

    if (
        optimizer_cls is torch.optim.AdamW
        and optimizer_kwargs.get("foreach", False)
        and optimizer_kwargs.get("fused", False)
    ):
        raise ValueError("torch.optim.AdamW does not support foreach=True and fused=True at the same time")
    optimizer_kwargs = _filter_optimizer_kwargs(optimizer_target_name, optimizer_cls, optimizer_kwargs)
    optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)

    logging.info("[INFO] Optimizer target: %s", optimizer_target_name)
    logging.info("[INFO] Optimizer config: %s", optimizer_kwargs)

    trainable_count = sum(1 for p in transformer_module.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in transformer_module.parameters() if not p.requires_grad)
    logging.info(f"[INFO] Trainable parameters: {trainable_count}, Frozen parameters: {frozen_count}")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        logging.info(f"[INFO] GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    logging.info("[INFO] NeMoAutoDiffusion setup complete (pipeline + optimizer)")

    return pipe, optimizer, getattr(fsdp2_manager, "device_mesh", None)


def build_lr_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> Optional[OptimizerParamScheduler]:
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler from YAML. If None, no scheduler
            is created and constant LR is used. Supports:
            - lr_decay_style: constant, linear, cosine, inverse-square-root, WSD
            - lr_warmup_steps: Number of warmup steps (or fraction < 1 for percentage)
            - min_lr: Minimum LR after decay
            - init_lr: Initial LR for warmup (defaults to 10% of max_lr if warmup enabled)
            - wd_incr_style: constant, linear, cosine (for weight decay scheduling)
            - wsd_decay_steps: WSD-specific decay steps
            - lr_wsd_decay_style: WSD-specific decay style (cosine, linear, exponential, minus_sqrt)
        optimizer: The optimizer to be scheduled.
        total_steps: Total number of optimizer steps for the training run.

    Returns:
        OptimizerParamScheduler instance, or None if cfg is None.
    """
    if cfg is None:
        return None

    user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    base_lr = optimizer.param_groups[0]["lr"]
    base_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    # Compute defaults from runtime values
    default_cfg: Dict[str, Any] = {
        "optimizer": optimizer,
        "lr_warmup_steps": min(1000, total_steps // 10),
        "lr_decay_steps": total_steps,
        "lr_decay_style": "cosine",
        "init_lr": base_lr * 0.1,
        "max_lr": base_lr,
        "min_lr": base_lr * 0.01,
        "start_wd": base_wd,
        "end_wd": base_wd,
        "wd_incr_steps": total_steps,
        "wd_incr_style": "constant",
    }

    # Handle warmup as fraction before merging
    if "lr_warmup_steps" in user_cfg:
        warmup = user_cfg["lr_warmup_steps"]
        if isinstance(warmup, float) and 0 < warmup < 1:
            user_cfg["lr_warmup_steps"] = int(warmup * total_steps)

    # WSD defaults if user specifies WSD style
    if user_cfg.get("lr_decay_style") == "WSD":
        default_cfg["wsd_decay_steps"] = max(1, total_steps // 10)
        default_cfg["lr_wsd_decay_style"] = "cosine"

    # User config overrides defaults
    default_cfg.update(user_cfg)

    # If user disabled warmup, set init_lr = max_lr
    if default_cfg["lr_warmup_steps"] == 0:
        default_cfg["init_lr"] = default_cfg["max_lr"]

    # Ensure warmup < decay steps
    if default_cfg["lr_warmup_steps"] >= default_cfg["lr_decay_steps"]:
        default_cfg["lr_warmup_steps"] = max(0, default_cfg["lr_decay_steps"] - 1)

    logging.info(
        f"[INFO] LR Scheduler: style={default_cfg['lr_decay_style']}, "
        f"warmup={default_cfg['lr_warmup_steps']}, total={default_cfg['lr_decay_steps']}, "
        f"max_lr={default_cfg['max_lr']}, min_lr={default_cfg['min_lr']}"
    )

    return OptimizerParamScheduler(
        optimizer=default_cfg["optimizer"],
        init_lr=default_cfg["init_lr"],
        max_lr=default_cfg["max_lr"],
        min_lr=default_cfg["min_lr"],
        lr_warmup_steps=default_cfg["lr_warmup_steps"],
        lr_decay_steps=default_cfg["lr_decay_steps"],
        lr_decay_style=default_cfg["lr_decay_style"],
        start_wd=default_cfg["start_wd"],
        end_wd=default_cfg["end_wd"],
        wd_incr_steps=default_cfg["wd_incr_steps"],
        wd_incr_style=default_cfg["wd_incr_style"],
        wsd_decay_steps=default_cfg.get("wsd_decay_steps"),
        lr_wsd_decay_style=default_cfg.get("lr_wsd_decay_style"),
    )


def is_main_process():
    """Return whether the current process should perform rank-zero work."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


class TrainDiffusionRecipe(BaseRecipe):
    """Training recipe for diffusion models."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            if run is not None:
                logging.info("🚀 View run at {}".format(run.url))

        self.seed = self.cfg.get("seed", 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)

        self.model_id = self.cfg.get("model.pretrained_model_name_or_path")
        self.attention_backend = self.cfg.get("model.attention_backend")
        self.transformer_engine_linear = bool(self.cfg.get("model.transformer_engine_linear", False))
        self.transformer_engine_fp8 = bool(self.cfg.get("model.transformer_engine_fp8", False))
        self.transformer_engine_fp8_recipe_name = str(self.cfg.get("model.transformer_engine_fp8_recipe", "delayed"))
        self.transformer_engine_fp8_amax_history_len = int(
            self.cfg.get("model.transformer_engine_fp8_amax_history_len", 1024)
        )
        self.transformer_engine_fp8_amax_compute_algo = str(
            self.cfg.get("model.transformer_engine_fp8_amax_compute_algo", "max")
        )
        self.fuse_qkv_projections = bool(self.cfg.get("model.fuse_qkv_projections", False))
        self.compact_fused_qkv_projections = bool(self.cfg.get("model.compact_fused_qkv_projections", False))
        self.torchao_float8_linear = bool(self.cfg.get("model.torchao_float8_linear", False))
        self.torchao_float8_fsdp_all_gather = bool(self.cfg.get("model.torchao_float8_fsdp_all_gather", True))
        self.torchao_float8_recipe_name = self.cfg.get("model.torchao_float8_recipe_name", "tensorwise")
        self.torchao_float8_force_recompute_weight_in_bwd = bool(
            self.cfg.get("model.torchao_float8_force_recompute_weight_in_bwd", False)
        )
        self.torchao_float8_precompute_scale_for_fsdp = bool(
            self.cfg.get("model.torchao_float8_precompute_scale_for_fsdp", False)
        )
        if self.transformer_engine_fp8:
            self.transformer_engine_linear = True
        if self.transformer_engine_linear and self.torchao_float8_linear:
            raise ValueError("Use only one of model.transformer_engine_linear or model.torchao_float8_linear")
        if self.transformer_engine_fp8 and self.torchao_float8_linear:
            raise ValueError("Use only one of model.transformer_engine_fp8 or model.torchao_float8_linear")
        if self.compact_fused_qkv_projections and not self.fuse_qkv_projections:
            raise ValueError("model.compact_fused_qkv_projections=true requires model.fuse_qkv_projections=true")
        if self.torchao_float8_precompute_scale_for_fsdp and not self.torchao_float8_linear:
            raise ValueError(
                "model.torchao_float8_precompute_scale_for_fsdp=true requires model.torchao_float8_linear=true"
            )
        if self.torchao_float8_precompute_scale_for_fsdp and not self.torchao_float8_fsdp_all_gather:
            raise ValueError(
                "model.torchao_float8_precompute_scale_for_fsdp=true requires model.torchao_float8_fsdp_all_gather=true"
            )
        self.learning_rate = self.cfg.get("optim.learning_rate", 5e-6)
        self.clip_grad_max_norm = float(self.cfg.get("optim.clip_grad", 1.0))
        self.bf16 = torch.bfloat16
        performance_cfg = self.cfg.get("performance", {}) or {}
        self.check_loss = bool(performance_cfg.get("check_loss", False))
        self.grad_clip_foreach = bool(performance_cfg.get("grad_clip_foreach", True))

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", self.world_size))
        self.local_world_size = max(self.local_world_size, 1)
        self.num_nodes = max(1, self.world_size // self.local_world_size)
        self.node_rank = dist.get_rank() // self.local_world_size if dist.is_initialized() else 0
        self._te_fp8_autocast = None
        self._te_fp8_recipe = None
        self._te_fp8_group = None
        if self.transformer_engine_fp8:
            self._te_fp8_autocast = _resolve_transformer_engine_autocast()
            self._te_fp8_recipe = _build_transformer_engine_fp8_recipe(
                self.transformer_engine_fp8_recipe_name,
                amax_history_len=self.transformer_engine_fp8_amax_history_len,
                amax_compute_algo=self.transformer_engine_fp8_amax_compute_algo,
            )
            self._te_fp8_group = dist.group.WORLD if dist.is_initialized() else None

        self._torchao_precompute_float8_dynamic_scale_for_fsdp = None
        if self.torchao_float8_precompute_scale_for_fsdp:
            available, precompute_float8_dynamic_scale_for_fsdp = safe_import_from(
                "torchao.float8",
                "precompute_float8_dynamic_scale_for_fsdp",
                msg="model.torchao_float8_precompute_scale_for_fsdp=true requires torchao.float8",
            )
            if not available:
                raise ImportError("model.torchao_float8_precompute_scale_for_fsdp=true requires torchao.float8")
            self._torchao_precompute_float8_dynamic_scale_for_fsdp = precompute_float8_dynamic_scale_for_fsdp

        logging.info("[INFO] Diffusion Trainer with Flow Matching")
        logging.info(
            f"[INFO] Total GPUs: {self.world_size}, GPUs per node: {self.local_world_size}, Num nodes: {self.num_nodes}"
        )
        logging.info(f"[INFO] Node rank: {self.node_rank}, Local rank: {self.local_rank}")
        logging.info(f"[INFO] Learning rate: {self.learning_rate}")
        logging.info("[INFO] Transformer Engine Linear: %s", self.transformer_engine_linear)
        logging.info(
            "[INFO] Transformer Engine FP8: %s (recipe=%s, amax_history_len=%s, amax_compute_algo=%s)",
            self.transformer_engine_fp8,
            self.transformer_engine_fp8_recipe_name,
            self.transformer_engine_fp8_amax_history_len,
            self.transformer_engine_fp8_amax_compute_algo,
        )
        logging.info("[INFO] Fuse QKV projections: %s", self.fuse_qkv_projections)
        logging.info("[INFO] Compact fused QKV projections: %s", self.compact_fused_qkv_projections)
        logging.info(
            "[INFO] torchao Float8Linear: %s (recipe=%s, fsdp_float8_all_gather=%s, "
            "force_recompute_weight_in_bwd=%s, precompute_scale_for_fsdp=%s)",
            self.torchao_float8_linear,
            self.torchao_float8_recipe_name,
            self.torchao_float8_fsdp_all_gather,
            self.torchao_float8_force_recompute_weight_in_bwd,
            self.torchao_float8_precompute_scale_for_fsdp,
        )
        logging.info(
            "[INFO] Performance config: check_loss=%s, grad_clip_foreach=%s",
            self.check_loss,
            self.grad_clip_foreach,
        )

        # Get distributed training configs (mutually exclusive)
        fsdp_cfg = self.cfg.get("fsdp", None)
        ddp_cfg = self.cfg.get("ddp", None)
        fm_cfg = self.cfg.get("flow_matching", {})

        # Validate mutually exclusive distributed configs
        if fsdp_cfg is not None and ddp_cfg is not None:
            raise ValueError(
                "Cannot specify both 'fsdp' and 'ddp' configurations in YAML. "
                "Please provide only one distributed training strategy."
            )

        self.cpu_offload = fsdp_cfg.get("cpu_offload", False) if fsdp_cfg else False

        # Flow matching configuration
        self.adapter_type = fm_cfg.get("adapter_type", "simple")
        self.timestep_sampling = fm_cfg.get("timestep_sampling", "logit_normal")
        self.logit_mean = fm_cfg.get("logit_mean", 0.0)
        self.logit_std = fm_cfg.get("logit_std", 1.0)
        self.flow_shift = fm_cfg.get("flow_shift", 3.0)
        self.mix_uniform_ratio = fm_cfg.get("mix_uniform_ratio", 0.1)
        self.use_sigma_noise = fm_cfg.get("use_sigma_noise", True)
        self.sigma_min = fm_cfg.get("sigma_min", 0.0)
        self.sigma_max = fm_cfg.get("sigma_max", 1.0)
        self.num_train_timesteps = fm_cfg.get("num_train_timesteps", 1000)
        self.i2v_prob = fm_cfg.get("i2v_prob", 0.3)
        self.cfg_dropout_prob = fm_cfg.get("cfg_dropout_prob", 0.1)
        self.use_loss_weighting = fm_cfg.get("use_loss_weighting", True)
        self.loss_weighting_scheme = fm_cfg.get("loss_weighting_scheme", "linear")
        self.log_interval = fm_cfg.get("log_interval", 100)
        self.summary_log_interval = fm_cfg.get("summary_log_interval", 10)

        # Adapter-specific configuration
        adapter_kwargs = fm_cfg.get("adapter_kwargs", {})
        self.adapter_kwargs = (
            adapter_kwargs.to_dict() if hasattr(adapter_kwargs, "to_dict") else dict(adapter_kwargs or {})
        )

        logging.info("[INFO] Flow Matching V2 Pipeline")
        logging.info(f"[INFO]   - Adapter type: {self.adapter_type}")
        logging.info(f"[INFO]   - Timestep sampling: {self.timestep_sampling}")
        logging.info(f"[INFO]   - Flow shift: {self.flow_shift}")
        logging.info(f"[INFO]   - Mix uniform ratio: {self.mix_uniform_ratio}")
        logging.info(f"[INFO]   - Use sigma noise: {self.use_sigma_noise}")
        logging.info(f"[INFO]   - CFG dropout prob: {self.cfg_dropout_prob}")
        logging.info(f"[INFO]   - Use loss weighting: {self.use_loss_weighting}")
        logging.info(f"[INFO]   - Loss weighting scheme: {self.loss_weighting_scheme}")

        # Get pipeline_spec for pretraining mode (required when mode != "finetune")
        pipeline_spec_cfg = self.cfg.get("model.pipeline_spec", None)
        pipeline_spec = pipeline_spec_cfg.to_dict() if pipeline_spec_cfg is not None else None

        # ── PEFT / LoRA configuration ─────────────────────────────────────────
        # Mirrors the LLM recipe pattern: peft block in YAML with _target_ pointing
        # to PeftConfig is instantiated directly, no intermediate wrapper class.
        self.peft_cfg = None
        if self.cfg.get("peft", None) is not None:
            self.peft_cfg = self.cfg.peft.instantiate()

        # model_type is explicit in yaml — no fragile string detection.
        # Required when peft block is present.
        self.model_type = self.cfg.get("model.model_type", None)
        if self.peft_cfg is not None and not self.model_type:
            raise ValueError(
                "model.model_type must be set when peft config is provided. Options: 'flux', 'wan', 'hunyuan'"
            )

        lora_status = (
            f"enabled (dim={self.peft_cfg.dim}, alpha={self.peft_cfg.alpha})"
            if self.peft_cfg is not None
            else "disabled (full fine-tune)"
        )
        logging.info(f"[INFO] LoRA: {lora_status}")

        (self.pipe, self.optimizer, self.device_mesh) = build_model_and_optimizer(
            model_id=self.model_id,
            finetune_mode=self.cfg.get("model.mode", "finetune").lower() == "finetune",
            learning_rate=self.learning_rate,
            device=self.device,
            dtype=self.bf16,
            cpu_offload=self.cpu_offload,
            fsdp_cfg=fsdp_cfg,
            ddp_cfg=ddp_cfg,
            optimizer_cfg=self.cfg.get("optim.optimizer", {}),
            transformer_engine_linear=self.transformer_engine_linear,
            transformer_engine_fp8_safe_only=self.transformer_engine_fp8,
            fuse_qkv_projections=self.fuse_qkv_projections,
            compact_fused_qkv_projections=self.compact_fused_qkv_projections,
            torchao_float8_linear=self.torchao_float8_linear,
            torchao_float8_fsdp_all_gather=self.torchao_float8_fsdp_all_gather,
            torchao_float8_recipe_name=self.torchao_float8_recipe_name,
            torchao_float8_force_recompute_weight_in_bwd=self.torchao_float8_force_recompute_weight_in_bwd,
            attention_backend=self.attention_backend,
            pipeline_spec=pipeline_spec,
            peft_cfg=self.peft_cfg,
            model_type=self.model_type,
        )

        self.model = self.pipe.transformer
        self.peft_config = getattr(self.pipe, "_peft_config", None)

        checkpoint_cfg = self.cfg.get("checkpoint", None)

        self.num_epochs = self.cfg.step_scheduler.num_epochs
        self.log_every = self.cfg.get("step_scheduler.log_every", 5)

        # Strictly require checkpoint config from YAML (no fallback)
        if checkpoint_cfg is None:
            raise ValueError(
                "checkpoint config is required in YAML (enabled, checkpoint_dir, model_save_format, save_consolidated)"
            )

        # Build BaseRecipe-style checkpointing configuration (DCP/TORCH_SAVE) from YAML
        model_state_dict_keys = list(self.model.state_dict().keys())
        model_cache_dir = self.cfg.get("model.cache_dir", None)
        self.checkpoint_config = CheckpointingConfig(
            enabled=checkpoint_cfg.get("enabled"),
            checkpoint_dir=checkpoint_cfg.get("checkpoint_dir"),
            model_save_format=checkpoint_cfg.get("model_save_format"),
            model_cache_dir=model_cache_dir if model_cache_dir is not None else HF_HUB_CACHE,
            model_repo_id=self.model_id,
            save_consolidated=checkpoint_cfg.get("save_consolidated"),
            is_peft=self.peft_cfg is not None,
            model_state_dict_keys=model_state_dict_keys,
            diffusers_compatible=checkpoint_cfg.get("diffusers_compatible", False),
        )
        self.restore_from = checkpoint_cfg.get("restore_from", None)
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=None,
        )

        dataloader_cfg = self.cfg.get("data.dataloader")
        if not hasattr(dataloader_cfg, "instantiate"):
            raise RuntimeError("data.dataloader must be a config node with instantiate()")

        self.dataloader, self.sampler = dataloader_cfg.instantiate(
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            batch_size=self.cfg.step_scheduler.local_batch_size,
        )

        self.raw_steps_per_epoch = len(self.dataloader)
        if self.raw_steps_per_epoch == 0:
            raise RuntimeError("Training dataloader is empty; cannot proceed with training")

        # Derive DP size consistent with model parallel config
        if ddp_cfg is not None:
            # DDP uses pure data parallelism across all ranks
            self.dp_size = self.world_size
        else:
            # FSDP may have TP/CP/PP dimensions
            _fsdp_cfg = fsdp_cfg or {}
            tp_size = _fsdp_cfg.get("tp_size", 1)
            cp_size = _fsdp_cfg.get("cp_size", 1)
            pp_size = _fsdp_cfg.get("pp_size", 1)
            denom = max(1, tp_size * cp_size * pp_size)
            self.dp_size = _fsdp_cfg.get("dp_size", None)
            if self.dp_size is None:
                self.dp_size = max(1, self.world_size // denom)

        # Infer local micro-batch size from dataloader if available
        self.local_batch_size = self.cfg.step_scheduler.local_batch_size
        # Desired global effective batch size across all DP ranks and nodes
        self.global_batch_size = self.cfg.step_scheduler.global_batch_size
        # Steps per epoch after gradient accumulation
        grad_acc_steps = max(1, self.global_batch_size // max(1, self.local_batch_size * self.dp_size))
        self.steps_per_epoch = ceil(self.raw_steps_per_epoch / grad_acc_steps)

        # Calculate total optimizer steps for LR scheduler
        total_steps = self.num_epochs * self.steps_per_epoch
        max_steps = self.cfg.get("step_scheduler.max_steps", None)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)

        # Build LR scheduler (returns None if lr_scheduler not in config)
        # Wrap in list for compatibility with checkpointing (OptimizerState expects list)
        lr_scheduler = build_lr_scheduler(
            self.cfg.get("lr_scheduler", None),
            self.optimizer,
            total_steps,
        )
        self.lr_scheduler = [lr_scheduler] if lr_scheduler is not None else None

        self.global_step = 0
        self.start_epoch = 0
        # Initialize StepScheduler for gradient accumulation and step/epoch bookkeeping
        self.step_scheduler = StepScheduler(
            global_batch_size=self.cfg.step_scheduler.global_batch_size,
            local_batch_size=self.cfg.step_scheduler.local_batch_size,
            dp_size=int(self.dp_size),
            ckpt_every_steps=self.cfg.step_scheduler.ckpt_every_steps,
            save_checkpoint_every_epoch=self.cfg.get("step_scheduler.save_checkpoint_every_epoch", False),
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=int(self.global_step),
            start_epoch=int(self.start_epoch),
            num_epochs=int(self.num_epochs),
            max_steps=max_steps,
        )

        self.load_checkpoint(self.restore_from)

        # Init Flow Matching Pipeline V2 with model adapter
        model_adapter = create_adapter(self.adapter_type, **self.adapter_kwargs)
        self.flow_matching_pipeline = FlowMatchingPipeline(
            model_adapter=model_adapter,
            num_train_timesteps=self.num_train_timesteps,
            timestep_sampling=self.timestep_sampling,
            flow_shift=self.flow_shift,
            i2v_prob=self.i2v_prob,
            cfg_dropout_prob=self.cfg_dropout_prob,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mix_uniform_ratio=self.mix_uniform_ratio,
            use_sigma_noise=self.use_sigma_noise,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            use_loss_weighting=self.use_loss_weighting,
            loss_weighting_scheme=self.loss_weighting_scheme,
            log_interval=self.log_interval,
            summary_log_interval=self.summary_log_interval,
            device=self.device,
        )
        logging.info(f"[INFO] Flow Matching Pipeline V2 initialized with {self.adapter_type} adapter")

        if is_main_process():
            os.makedirs(self.checkpoint_config.checkpoint_dir, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

    def _transformer_engine_fp8_context(self) -> Any:
        """Return the per-forward Transformer Engine FP8 context."""
        if not self.transformer_engine_fp8:
            return nullcontext()
        return self._te_fp8_autocast(
            enabled=True,
            recipe=self._te_fp8_recipe,
            amax_reduction_group=self._te_fp8_group,
        )

    def _precompute_torchao_float8_scales(self) -> None:
        """Precompute torchao FP8 FSDP scales after optimizer updates when enabled."""
        if self._torchao_precompute_float8_dynamic_scale_for_fsdp is None:
            return
        self._torchao_precompute_float8_dynamic_scale_for_fsdp(self.model)

    def _get_torch_profiler_config(self) -> Dict[str, Any]:
        """Return the optional torch profiler config."""
        profiling_cfg = self.cfg.get("profiling", {}) or {}
        if hasattr(profiling_cfg, "to_dict"):
            profiling_cfg = profiling_cfg.to_dict()
        torch_profiler_cfg = profiling_cfg.get("torch_profiler", profiling_cfg) or {}
        if hasattr(torch_profiler_cfg, "to_dict"):
            torch_profiler_cfg = torch_profiler_cfg.to_dict()
        return dict(torch_profiler_cfg)

    def _should_profile_current_rank(self, profiler_cfg: Dict[str, Any]) -> bool:
        """Return whether torch profiling should run on this rank."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        ranks = profiler_cfg.get("ranks", [0])
        if ranks in ("all", "*"):
            return True
        if isinstance(ranks, int):
            return rank == ranks
        return rank in {int(profile_rank) for profile_rank in ranks}

    def _build_torch_profiler(self) -> Optional[Any]:
        """Build an opt-in torch profiler for bounded performance experiments."""
        profiler_cfg = self._get_torch_profiler_config()
        if not profiler_cfg.get("enabled", False):
            return None
        if not self._should_profile_current_rank(profiler_cfg):
            return None

        rank = dist.get_rank() if dist.is_initialized() else 0
        output_dir = Path(str(profiler_cfg.get("output_dir", "profile_default"))).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        wait_steps = int(profiler_cfg.get("wait_steps", 1))
        warmup_steps = int(profiler_cfg.get("warmup_steps", 1))
        active_steps = int(profiler_cfg.get("active_steps", 3))
        repeat = int(profiler_cfg.get("repeat", 1))
        row_limit = int(profiler_cfg.get("row_limit", 40))
        sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"

        def trace_handler(profiler: torch.profiler.profile) -> None:
            step = int(self.step_scheduler.step)
            trace_path = output_dir / f"rank{rank}_step{step}.pt.trace.json"
            summary_path = output_dir / f"rank{rank}_step{step}_key_averages.txt"
            profiler.export_chrome_trace(str(trace_path))
            summary = profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit)
            summary_path.write_text(summary + "\n", encoding="utf-8")
            logging.info("[PROFILE] Wrote torch profiler trace to %s", trace_path)
            logging.info("[PROFILE] Wrote torch profiler summary to %s", summary_path)

        logging.info(
            "[PROFILE] Torch profiler enabled on rank %s: output_dir=%s wait=%s warmup=%s active=%s repeat=%s",
            rank,
            output_dir,
            wait_steps,
            warmup_steps,
            active_steps,
            repeat,
        )
        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=repeat,
            ),
            on_trace_ready=trace_handler,
            record_shapes=bool(profiler_cfg.get("record_shapes", True)),
            profile_memory=bool(profiler_cfg.get("profile_memory", True)),
            with_stack=bool(profiler_cfg.get("with_stack", False)),
            with_flops=bool(profiler_cfg.get("with_flops", False)),
        )

    def run_train_validation_loop(self):
        logging.info("[INFO] Starting T2V training with Flow Matching")
        logging.info(f"[INFO] Global Batch size: {self.global_batch_size}; Local Batch size: {self.local_batch_size}")
        logging.info(f"[INFO] Num nodes: {self.num_nodes}; DP size: {self.dp_size}")

        # Keep global_step synchronized with scheduler
        global_step = int(self.step_scheduler.step)
        self._sync_device()
        perf_window_start_time = time.perf_counter()
        perf_window_steps = 0
        perf_window_local_samples = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        profiler = self._build_torch_profiler()
        if profiler is not None:
            profiler.start()

        for epoch in self.step_scheduler.epochs:
            if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

            # Optionally wrap dataloader with tqdm for rank-0
            if is_main_process():
                from tqdm import tqdm

                self.step_scheduler.dataloader = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            else:
                self.step_scheduler.dataloader = self.dataloader

            epoch_loss = 0.0
            num_steps = 0

            for batch_group in self.step_scheduler:
                self.optimizer.zero_grad(set_to_none=True)

                micro_losses = []
                prepare_for_grad_accumulation([self.model], pp_enabled=False)
                num_microbatches = len(batch_group)
                for microbatch_idx, micro_batch in enumerate(batch_group):
                    if microbatch_idx == num_microbatches - 1:
                        prepare_for_final_backward([self.model], pp_enabled=False)

                    try:
                        with self._transformer_engine_fp8_context():
                            _, average_weighted_loss, _, _ = self.flow_matching_pipeline.step(
                                model=self.model,
                                batch=micro_batch,
                                device=self.device,
                                dtype=self.bf16,
                                global_step=global_step,
                                collect_metrics=False,
                                check_loss=self.check_loss,
                            )
                    except Exception as exc:
                        logging.info(f"[ERROR] Training step failed at epoch {epoch}, step {num_steps}: {exc}")
                        video_shape = micro_batch.get("video_latents", torch.tensor([])).shape
                        text_shape = micro_batch.get("text_embeddings", torch.tensor([])).shape
                        logging.info(f"[DEBUG] Batch shapes - video: {video_shape}, text: {text_shape}")
                        raise

                    # Use average_weighted_loss for backprop (scalar for gradient accumulation)
                    (average_weighted_loss / num_microbatches).backward()
                    micro_losses.append(average_weighted_loss.detach())

                    if microbatch_idx == 0:
                        prepare_after_first_microbatch()

                grad_norm = clip_grad_norm(self.clip_grad_max_norm, [self.model], foreach=self.grad_clip_foreach)
                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm

                # ── LoRA gradient diagnostic (step 1 only) ───────────────────
                if global_step == 1 and self.peft_cfg is not None:
                    for n, p in self.model.named_parameters():
                        if "lora_B" in n:
                            try:
                                grad_val = p.grad.to_local().float().norm().item() if p.grad is not None else None
                            except Exception:
                                grad_val = p.grad.float().norm().item() if p.grad is not None else None
                            logging.info(
                                f"[GRAD CHECK] {n}: grad_norm={grad_val}, param_norm={p.data.float().norm().item():.6f}"
                            )
                            break

                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler[0].step(1)
                self._precompute_torchao_float8_scales()

                perf_window_steps += 1
                perf_window_local_samples += _count_local_batch_group_samples(batch_group)
                group_loss_mean = float(torch.stack(micro_losses).mean().item())
                epoch_loss += group_loss_mean
                num_steps += 1
                global_step = int(self.step_scheduler.step)

                should_log = self.log_every and self.log_every > 0 and global_step % self.log_every == 0
                if should_log:
                    elapsed_seconds, perf_window_end_time = self._elapsed_seconds_since(perf_window_start_time)
                    perf_window_global_samples = self._count_global_samples(perf_window_local_samples)
                    throughput_metrics = _calculate_throughput_metrics(
                        elapsed_seconds=elapsed_seconds,
                        optimizer_steps=perf_window_steps,
                        global_samples=perf_window_global_samples,
                        world_size=self.world_size,
                    )
                    memory_metrics = self._get_memory_metrics()
                    perf_window_start_time = perf_window_end_time
                    perf_window_steps = 0
                    perf_window_local_samples = 0

                if should_log and is_main_process():
                    avg_loss = epoch_loss / num_steps
                    log_dict = {
                        "train_loss": group_loss_mean,
                        "train_avg_loss": avg_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "epoch": epoch,
                        "global_step": global_step,
                        **throughput_metrics,
                        **memory_metrics,
                    }
                    if wandb.run is not None:
                        wandb.log(log_dict, step=global_step)
                    logging.info(
                        "[TRAIN] step=%s epoch=%s loss=%.6f avg_loss=%.6f lr=%.3e grad_norm=%.3f "
                        "step_time=%.3fs samples_per_sec=%.2f samples_per_sec_per_gpu=%.2f mem=%.2fGB",
                        global_step,
                        epoch,
                        group_loss_mean,
                        avg_loss,
                        self.optimizer.param_groups[0]["lr"],
                        grad_norm,
                        throughput_metrics["step_time"],
                        throughput_metrics["samples_per_sec"],
                        throughput_metrics["samples_per_sec_per_gpu"],
                        memory_metrics["max_memory_allocated_gb"],
                    )

                    # Update tqdm if present
                    if hasattr(self.step_scheduler.dataloader, "set_postfix"):
                        self.step_scheduler.dataloader.set_postfix(
                            {
                                "loss": f"{group_loss_mean:.4f}",
                                "avg": f"{(avg_loss):.4f}",
                                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                "gn": f"{grad_norm:.2f}",
                                "s/s": f"{throughput_metrics['samples_per_sec']:.1f}",
                                "s/s/gpu": f"{throughput_metrics['samples_per_sec_per_gpu']:.2f}",
                            }
                        )

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)

                if profiler is not None:
                    profiler.step()

            if num_steps == 0:
                logging.info(f"[INFO] Epoch {epoch + 1} skipped (already completed in previous run)")
                continue
            avg_loss = epoch_loss / num_steps
            logging.info(f"[INFO] Epoch {epoch + 1} complete. avg_loss={avg_loss:.6f}")

            if is_main_process() and wandb.run is not None:
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch + 1}, step=global_step)

        if is_main_process():
            logging.info(f"[INFO] Saved final checkpoint at step {global_step}")
            if wandb.run is not None:
                wandb.finish()

        if profiler is not None:
            profiler.stop()

        logging.info("[INFO] Training complete!")

    def _get_dp_rank(self, include_cp: bool = False) -> int:
        """Get data parallel rank, handling DDP mode where device_mesh is None."""
        # In DDP mode, device_mesh is None, so use torch.distributed directly
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return dist.get_rank() if dist.is_initialized() else 0
        # Otherwise, use the parent implementation
        return super()._get_dp_rank(include_cp=include_cp)

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        """Get data parallel world size, handling DDP mode where device_mesh is None."""
        # In DDP mode, device_mesh is None, so use torch.distributed directly
        device_mesh = getattr(self, "device_mesh", None)
        if device_mesh is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        # Otherwise, use the parent implementation
        return super()._get_dp_group_size(include_cp=include_cp)

    def _sync_device(self) -> None:
        """Wait for queued CUDA work so timing reflects completed training work."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _get_collective_device(self) -> torch.device:
        """Return a tensor device compatible with the active distributed backend."""
        if dist.is_initialized() and str(dist.get_backend()).lower() == "nccl" and torch.cuda.is_available():
            return self.device
        return torch.device("cpu")

    def _elapsed_seconds_since(self, start_time: float) -> tuple[float, float]:
        """Return the max elapsed wall-clock seconds across ranks since start_time."""
        self._sync_device()
        end_time = time.perf_counter()
        elapsed_seconds = max(end_time - start_time, 1e-12)
        if dist.is_initialized():
            elapsed = torch.tensor(elapsed_seconds, device=self._get_collective_device(), dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            elapsed_seconds = float(elapsed.item())
        return elapsed_seconds, end_time

    def _count_global_samples(self, local_samples: int) -> int:
        """Count samples processed across the data-parallel group."""
        global_samples = int(local_samples)
        if dist.is_initialized():
            sample_count = torch.tensor(global_samples, device=self._get_collective_device(), dtype=torch.long)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM, group=self._get_dp_group())
            global_samples = int(sample_count.item())
        return global_samples

    def _get_memory_metrics(self) -> Dict[str, float]:
        """Return PyTorch CUDA allocator memory counters, max-reduced across ranks."""
        if not torch.cuda.is_available():
            return {
                "mem": 0.0,
                "memory_allocated_gb": 0.0,
                "memory_reserved_gb": 0.0,
                "max_memory_allocated_gb": 0.0,
                "max_memory_reserved_gb": 0.0,
            }

        scale = 1024**3
        memory = torch.tensor(
            [
                torch.cuda.memory_allocated(self.device) / scale,
                torch.cuda.memory_reserved(self.device) / scale,
                torch.cuda.max_memory_allocated(self.device) / scale,
                torch.cuda.max_memory_reserved(self.device) / scale,
            ],
            device=self._get_collective_device(),
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)

        allocated, reserved, max_allocated, max_reserved = memory.tolist()
        return {
            "mem": max_allocated,
            "memory_allocated_gb": allocated,
            "memory_reserved_gb": reserved,
            "max_memory_allocated_gb": max_allocated,
            "max_memory_reserved_gb": max_reserved,
        }
