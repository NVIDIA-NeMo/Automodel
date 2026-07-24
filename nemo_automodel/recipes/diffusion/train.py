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
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import suppress_wandb_log_messages
from nemo_automodel.components.optim.precision_warnings import warn_if_torch_adam_with_bf16_params
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.recipes._dist_utils import parse_distributed_section
from nemo_automodel.recipes._typed_config import RecipeConfig, _model_name_from_cfg
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.utils import dtype_from_str

_OPTIMIZER_DEFAULT_TARGET = "torch.optim.AdamW"
_TORCH_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _resolve_optimizer_dtype_strings(optimizer_cfg: Any) -> None:
    """Resolve dtype strings in optimizer config objects in place."""
    for attr in ("master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        val = getattr(optimizer_cfg, attr, None)
        if isinstance(val, str):
            setattr(optimizer_cfg, attr, dtype_from_str(val))


def _resolve_model_dtypes(cfg: Any) -> tuple[torch.dtype, torch.dtype]:
    """Resolve model storage and compute dtypes from the recipe config."""
    return (
        dtype_from_str(cfg.get("model.torch_dtype", None), default=torch.bfloat16),
        dtype_from_str(cfg.get("model.compute_dtype", None), default=torch.bfloat16),
    )


def _validate_precision_configuration(
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
    *,
    ddp_cfg: Optional[Dict[str, Any]],
    peft_cfg: Any,
) -> None:
    """Reject split storage/compute dtypes on paths without FSDP param casting."""
    if dtype == compute_dtype:
        return

    unsupported_modes = []
    if ddp_cfg is not None:
        unsupported_modes.append("DDP")
    if peft_cfg is not None:
        unsupported_modes.append("PEFT/LoRA")
    if not unsupported_modes:
        return

    modes = " and ".join(unsupported_modes)
    raise ValueError(
        f"model.torch_dtype ({dtype}) and model.compute_dtype ({compute_dtype}) must match for {modes}. "
        "Split storage/compute dtypes require FSDP full-parameter training, where FSDP can cast gathered "
        "parameters to the compute dtype."
    )


def _build_optimizer(
    trainable_params: list[torch.nn.Parameter],
    optimizer_cfg: Any,
    learning_rate: float,
    is_peft: bool = False,
) -> torch.optim.Optimizer:
    """Build optimizer from config, falling back to AdamW for legacy configs."""
    optimizer_cfg = optimizer_cfg or {}
    if isinstance(optimizer_cfg, dict) and "_target_" in optimizer_cfg:
        optimizer_cfg = ConfigNode(optimizer_cfg)

    if hasattr(optimizer_cfg, "_target_"):
        _resolve_optimizer_dtype_strings(optimizer_cfg)
        optimizer_target = optimizer_cfg._target_
        optimizer_target_name = _get_optimizer_target_name(optimizer_target)
        optimizer_cls = _resolve_optimizer_class(optimizer_target)
        optimizer_dict = optimizer_cfg.to_dict() if hasattr(optimizer_cfg, "to_dict") else dict(optimizer_cfg)

        optimizer_kwargs = {"lr": learning_rate}
        for key, value in optimizer_dict.items():
            if key in {"_target_", "lr", "learning_rate"}:
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
        warn_if_torch_adam_with_bf16_params(
            optimizer=optimizer,
            parameters=trainable_params,
            is_peft=is_peft,
            context="diffusion",
            logger=logging.getLogger(__name__),
        )
        return optimizer

    optimizer_dict = optimizer_cfg.to_dict() if hasattr(optimizer_cfg, "to_dict") else dict(optimizer_cfg)
    weight_decay = optimizer_dict.get("weight_decay", 0.01)
    betas = tuple(optimizer_dict.get("betas", (0.9, 0.999)))
    adamw_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": betas,
        "eps": optimizer_dict.get("eps", 1e-8),
        "amsgrad": optimizer_dict.get("amsgrad", False),
    }
    for key in ("foreach", "fused", "capturable", "maximize"):
        value = optimizer_dict.get(key, None)
        if value is not None:
            adamw_kwargs[key] = value
    if adamw_kwargs.get("foreach", False) and adamw_kwargs.get("fused", False):
        raise ValueError("torch.optim.AdamW does not support foreach=True and fused=True at the same time")
    optimizer = torch.optim.AdamW(trainable_params, **adamw_kwargs)

    logging.info("[INFO] Optimizer config: %s", adamw_kwargs)
    warn_if_torch_adam_with_bf16_params(
        optimizer=optimizer,
        parameters=trainable_params,
        is_peft=is_peft,
        context="diffusion",
        logger=logging.getLogger(__name__),
    )
    return optimizer


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


class _PhaseTimer:
    """Accumulate optional CPU or CUDA-event phase timings for one log window."""

    def __init__(self, *, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self._use_cuda_events = enabled and device.type == "cuda" and torch.cuda.is_available()
        self._cpu_seconds: Dict[str, float] = {}
        self._cuda_events: Dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self) -> float | torch.cuda.Event | None:
        """Start one phase without synchronizing the accelerator."""
        if not self.enabled:
            return None
        if self._use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def stop(self, name: str, started: float | torch.cuda.Event | None) -> None:
        """Stop and accumulate one named phase without an eager device sync."""
        if started is None:
            return
        if self._use_cuda_events:
            if not isinstance(started, torch.cuda.Event):
                raise TypeError(f"CUDA phase {name!r} must start with torch.cuda.Event")
            ended = torch.cuda.Event(enable_timing=True)
            ended.record(torch.cuda.current_stream(self.device))
            self._cuda_events.setdefault(name, []).append((started, ended))
            return
        if not isinstance(started, float):
            raise TypeError(f"CPU phase {name!r} must start with a perf-counter timestamp")
        self._cpu_seconds[name] = self._cpu_seconds.get(name, 0.0) + (time.perf_counter() - started)

    def elapsed_seconds(self) -> Dict[str, float]:
        """Return accumulated seconds after the caller has synchronized CUDA."""
        elapsed = dict(self._cpu_seconds)
        for name, event_pairs in self._cuda_events.items():
            elapsed[name] = sum(start.elapsed_time(end) for start, end in event_pairs) / 1000.0
        return elapsed

    def reset(self) -> None:
        """Clear completed measurements for the next log window."""
        self._cpu_seconds.clear()
        self._cuda_events.clear()


def _build_diffusion_parallel_manager_args(
    *,
    fsdp_cfg: Optional[Dict[str, Any]],
    ddp_cfg: Optional[Dict[str, Any]],
    world_size: int,
    dtype: torch.dtype,
    compute_dtype: Optional[torch.dtype] = None,
    lora_enabled: bool,
) -> Dict[str, Any]:
    """Build diffusion transformer manager args through the shared distributed parser."""
    if compute_dtype is None:
        compute_dtype = dtype

    # The recipe passes ConfigNode sections, which support .to_dict() but not dict().
    if hasattr(fsdp_cfg, "to_dict"):
        fsdp_cfg = fsdp_cfg.to_dict()
    if hasattr(ddp_cfg, "to_dict"):
        ddp_cfg = ddp_cfg.to_dict()

    if fsdp_cfg is not None and ddp_cfg is not None:
        raise ValueError(
            "Cannot specify both 'fsdp' and 'ddp' configurations. "
            "Please provide only one distributed training strategy."
        )

    if ddp_cfg is not None:
        ddp_options = dict(ddp_cfg)
        ddp_options.pop("backend", None)
        parsed = parse_distributed_section({"strategy": "ddp", **ddp_options})
        return {
            "_manager_type": "ddp",
            "world_size": world_size,
            **parsed["strategy_config"].to_dict(),
            "activation_checkpointing": parsed["activation_checkpointing"],
        }

    fsdp_options = dict(fsdp_cfg or {})
    ignored_options = {"use_hf_tp_plan": fsdp_options.pop("use_hf_tp_plan", False)}
    fsdp_options.pop("backend", None)
    cpu_offload = bool(fsdp_options.pop("cpu_offload", False))
    reduce_dtype = dtype_from_str(fsdp_options.pop("reduce_dtype", None), default=torch.float32)

    param_dtype = None if lora_enabled else compute_dtype
    parsed = parse_distributed_section(
        {
            "strategy": "fsdp2",
            "activation_checkpointing": True,
            "defer_fsdp_grad_sync": True,
            "enable_fsdp2_prefetch": True,
            **fsdp_options,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                output_dtype=compute_dtype,
            ),
            # CPU offload: sharded params + optimizer state live on host RAM and are
            # paged to GPU per-block during forward/backward (saves GPU memory, adds H2D).
            "offload_policy": CPUOffloadPolicy(pin_memory=True) if cpu_offload else None,
        }
    )

    return {
        "_manager_type": "fsdp2",
        "world_size": world_size,
        "dp_size": parsed["dp_size"],
        "dp_replicate_size": parsed["dp_replicate_size"],
        "tp_size": parsed["tp_size"],
        "cp_size": parsed["cp_size"],
        "pp_size": parsed["pp_size"],
        "ep_size": parsed["ep_size"],
        **parsed["strategy_config"].to_dict(),
        "activation_checkpointing": parsed["activation_checkpointing"],
        **ignored_options,
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
    model_revision: str | None = None,
    finetune_mode: bool,
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
    compute_dtype: Optional[torch.dtype] = None,
    cpu_offload: bool = False,
    fsdp_cfg: Optional[Dict[str, Any]] = None,
    ddp_cfg: Optional[Dict[str, Any]] = None,
    attention_backend: Optional[str] = None,
    optimizer_cfg: Optional[Dict[str, Any]] = None,
    transformer_engine_linear: bool = False,
    transformer_engine_fp8_safe_only: bool = False,
    fuse_qkv_projections: bool = False,
    compact_fused_qkv_projections: bool = False,
    pipeline_spec: Optional[Dict[str, Any]] = None,
    peft_cfg=None,
    model_type=None,
    active_transformer: Optional[str] = None,
) -> tuple[NeMoAutoDiffusionPipeline, torch.optim.Optimizer, Any]:
    """Build the diffusion model, parallel scheme, and optimizer.

    Args:
        model_id: Pretrained model name or path.
        model_revision: Optional immutable Hugging Face revision for the pretrained model.
        finetune_mode: Whether to load for finetuning (True) or pretraining (False).
        learning_rate: Learning rate for optimizer.
        device: Target device.
        dtype: Model parameter storage dtype.
        compute_dtype: Forward/FSDP compute dtype. Defaults to dtype when unset.
        cpu_offload: Whether to enable CPU offload (FSDP only).
        fsdp_cfg: FSDP configuration dict. Mutually exclusive with ddp_cfg.
        ddp_cfg: DDP configuration dict. Mutually exclusive with fsdp_cfg.
        attention_backend: Optional attention backend override.
        optimizer_cfg: Optional optimizer configuration.
        transformer_engine_linear: Whether to replace transformer torch.nn.Linear modules with Transformer Engine Linear.
        transformer_engine_fp8_safe_only: Whether to skip TE conversion for known FP8-incompatible modules.
        fuse_qkv_projections: Whether to call Diffusers QKV projection fusion on the transformer before FSDP.
        compact_fused_qkv_projections: Whether to remove original projection modules after QKV fusion.
        pipeline_spec: Pipeline specification for pretraining (from_config).
            Required when finetune_mode is False. Should contain:
            - transformer_cls: str (e.g., "WanTransformer3DModel", "FluxTransformer2DModel")
            - subfolder: str (e.g., "transformer")
            - Optional: pipeline_cls, load_full_pipeline
        peft_cfg: PeftConfig instance or None. When provided, only LoRA params
            are trained; base weights are frozen and sharded by FSDP2 for memory.
        model_type: "flux" | "wan" | "hunyuan". Required when peft_cfg is provided.
        active_transformer: For two-transformer pipelines (Wan2.2), select which
            transformer to finetune. ``"transformer"`` (default for Wan2.2 = high-noise)
            or ``"transformer_2"`` (low-noise). The unused transformer is dropped
            before device placement so only one transformer lives on GPU.

    Returns:
        Tuple of (pipeline, optimizer, device_mesh or None).

    Raises:
        ValueError: If both fsdp_cfg and ddp_cfg are provided.
        ValueError: If finetune_mode is False and pipeline_spec is not provided.
    """
    logging.info("[INFO] Building NeMoAutoDiffusionPipeline with transformer parallel scheme...")

    if not dist.is_initialized():
        logging.info("[WARN] torch.distributed not initialized; proceeding in single-process mode")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if compute_dtype is None:
        compute_dtype = dtype

    lora_enabled = peft_cfg is not None
    _validate_precision_configuration(dtype, compute_dtype, ddp_cfg=ddp_cfg, peft_cfg=peft_cfg)

    if ddp_cfg is not None:
        logging.info("[INFO] Using DDP (DistributedDataParallel) for training")
    else:
        logging.info("[INFO] Using FSDP2 (Fully Sharded Data Parallel) for training")
    manager_args = _build_diffusion_parallel_manager_args(
        fsdp_cfg=fsdp_cfg,
        ddp_cfg=ddp_cfg,
        world_size=world_size,
        dtype=dtype,
        compute_dtype=compute_dtype,
        lora_enabled=lora_enabled,
    )

    parallel_scheme = {"transformer": manager_args}

    if finetune_mode:
        # Finetuning: load from pretrained weights
        logging.info("[INFO] Loading pretrained model for finetuning")
        if active_transformer is not None:
            logging.info("[INFO] Active transformer: %s", active_transformer)
        pipe, created_managers = NeMoAutoDiffusionPipeline.from_pretrained(
            model_id,
            revision=model_revision,
            torch_dtype=dtype,
            device=device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=True,
            low_cpu_mem_usage=True,
            peft_cfg=peft_cfg,
            model_type=model_type,
            active_transformer=active_transformer,
            transformer_engine_linear=transformer_engine_linear,
            transformer_engine_fp8_safe_only=transformer_engine_fp8_safe_only,
            fuse_qkv_projections=fuse_qkv_projections,
            compact_fused_qkv_projections=compact_fused_qkv_projections,
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

    optimizer = _build_optimizer(trainable_params, optimizer_cfg, learning_rate, is_peft=lora_enabled)

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
        self.cfg = cfg if isinstance(cfg, RecipeConfig) else RecipeConfig(cfg)

    def setup(self):
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 1),
        )
        setup_logging()

        if self.dist_env.is_main and self.cfg.wandb is not None:
            suppress_wandb_log_messages()
            # For two-stage Wan2.2 finetuning, suffix the wandb run name with the
            # active stage so high-noise and low-noise runs are distinguishable.
            # Normalize to lowercase to match self.stage so the wandb suffix and the
            # internal stage name stay consistent. Mutating the cached WandbConfig
            # before build() makes the suffix take effect (build() uses self.name).
            stage_for_wandb = self.cfg.get("model.stage", None)
            if stage_for_wandb is not None:
                stage_for_wandb = str(stage_for_wandb).lower()
                current_name = self.cfg.get("wandb.name", None)
                if current_name is not None and not str(current_name).endswith(f"_{stage_for_wandb}"):
                    self.cfg.wandb.name = f"{current_name}_{stage_for_wandb}"
            model_name = _model_name_from_cfg(self.cfg.model) if "model" in self.cfg else None
            run = self.cfg.wandb.build(run_config=self.cfg.to_dict(), model_name=model_name)
            if run is not None:
                logging.info("🚀 View run at {}".format(run.url))

        self.seed = self.cfg.get("seed", 42)
        self.rng = StatefulRNG(seed=self.seed, ranked=True)

        self.model_id = self.cfg.get("model.pretrained_model_name_or_path")
        self.model_revision = self.cfg.get("model.revision", None)
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
        self.optimize_hunyuan_flash_varlen_mask = bool(self.cfg.get("model.optimize_hunyuan_flash_varlen_mask", False))
        if self.transformer_engine_fp8:
            self.transformer_engine_linear = True
        if self.compact_fused_qkv_projections and not self.fuse_qkv_projections:
            raise ValueError("model.compact_fused_qkv_projections=true requires model.fuse_qkv_projections=true")
        self.learning_rate = self.cfg.get("optim.learning_rate", 5e-6)
        self.clip_grad_max_norm = float(self.cfg.get("optim.clip_grad", 1.0))
        self.model_dtype, self.compute_dtype = _resolve_model_dtypes(self.cfg)
        performance_cfg = self.cfg.get("performance", {}) or {}
        self.check_loss = bool(performance_cfg.get("check_loss", False))
        self.grad_clip_foreach = bool(performance_cfg.get("grad_clip_foreach", True))
        self.throughput_warmup_steps = int(performance_cfg.get("throughput_warmup_steps", 0))
        self.phase_timing = bool(performance_cfg.get("phase_timing", False))
        if self.throughput_warmup_steps < 0:
            raise ValueError("performance.throughput_warmup_steps must be non-negative")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.flow_generator = self.rng.generator(self.device)

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
        logging.info("[INFO] Optimize Hunyuan flash-varlen mask: %s", self.optimize_hunyuan_flash_varlen_mask)
        logging.info("[INFO] Precision: model_dtype=%s, compute_dtype=%s", self.model_dtype, self.compute_dtype)
        logging.info(
            "[INFO] Performance config: check_loss=%s, grad_clip_foreach=%s, warmup_steps=%s, phase_timing=%s",
            self.check_loss,
            self.grad_clip_foreach,
            self.throughput_warmup_steps,
            self.phase_timing,
        )

        # Get distributed training configs (mutually exclusive)
        fsdp_cfg = self.cfg.get("fsdp", None)
        ddp_cfg = self.cfg.get("ddp", None)
        self.flow_matching_config = self.cfg.flow_matching

        # Validate mutually exclusive distributed configs
        if fsdp_cfg is not None and ddp_cfg is not None:
            raise ValueError(
                "Cannot specify both 'fsdp' and 'ddp' configurations in YAML. "
                "Please provide only one distributed training strategy."
            )

        self.cpu_offload = fsdp_cfg.get("cpu_offload", False) if fsdp_cfg else False
        self.defer_fsdp_grad_sync = fsdp_cfg.get("defer_fsdp_grad_sync", True) if fsdp_cfg else True

        # Flow matching configuration. Construct the adapter before the model so
        # model-owned parallel strategies are registered before FSDP planning.
        self.adapter_type = self.flow_matching_config.adapter_type or (
            "target" if self.flow_matching_config.adapter is not None else "simple"
        )
        self.timestep_sampling = self.flow_matching_config.timestep_sampling
        self.logit_mean = self.flow_matching_config.logit_mean
        self.logit_std = self.flow_matching_config.logit_std
        self.flow_shift = self.flow_matching_config.flow_shift
        self.mix_uniform_ratio = self.flow_matching_config.mix_uniform_ratio
        self.use_sigma_noise = self.flow_matching_config.use_sigma_noise
        self.sigma_min = self.flow_matching_config.sigma_min
        self.sigma_max = self.flow_matching_config.sigma_max
        self.num_train_timesteps = self.flow_matching_config.num_train_timesteps
        self.i2v_prob = self.flow_matching_config.i2v_prob
        self.cfg_dropout_prob = self.flow_matching_config.cfg_dropout_prob
        self.use_loss_weighting = self.flow_matching_config.use_loss_weighting
        self.loss_weighting_scheme = self.flow_matching_config.loss_weighting_scheme
        self.log_interval = self.flow_matching_config.log_interval
        self.summary_log_interval = self.flow_matching_config.summary_log_interval
        self.model_adapter = self.flow_matching_config.build_adapter()
        self.model_adapter.register_parallel_strategy()
        if self.optimize_hunyuan_flash_varlen_mask:
            if self.adapter_type != "hunyuan":
                raise ValueError(
                    "model.optimize_hunyuan_flash_varlen_mask=true requires flow_matching.adapter_type=hunyuan"
                )
            if self.attention_backend != "flash_varlen":
                raise ValueError(
                    "model.optimize_hunyuan_flash_varlen_mask=true requires model.attention_backend=flash_varlen"
                )

        # Two-stage finetuning (Wan2.2 T2V-A14B): each stage trains only one
        # transformer against a restricted timestep range. The stage knob both
        # selects the active transformer and clamps the sigma sampling window so
        # this run only sees noise levels its transformer is responsible for.
        self.stage = self.cfg.get("model.stage", None)
        self.boundary_ratio = self.cfg.get("model.boundary_ratio", None)
        self.active_transformer = None
        if self.stage is not None:
            stage = str(self.stage).lower()
            if stage not in ("high_noise", "low_noise"):
                raise ValueError(f"model.stage must be 'high_noise' or 'low_noise', got {self.stage!r}")
            self.stage = stage
            self.active_transformer = "transformer" if stage == "high_noise" else "transformer_2"

        logging.info("[INFO] Flow Matching V2 Pipeline")
        logging.info(f"[INFO]   - Adapter type: {self.adapter_type}")
        logging.info(f"[INFO]   - Timestep sampling: {self.timestep_sampling}")
        if self.timestep_sampling == "beta":
            logging.info(
                "[INFO]   - Beta parameters: alpha=%s, beta=%s",
                self.flow_matching_config.beta_alpha,
                self.flow_matching_config.beta_beta,
            )
        logging.info(f"[INFO]   - Flow shift: {self.flow_shift}")
        logging.info(f"[INFO]   - Mix uniform ratio: {self.mix_uniform_ratio}")
        logging.info(f"[INFO]   - Use sigma noise: {self.use_sigma_noise}")
        logging.info(f"[INFO]   - CFG dropout prob: {self.cfg_dropout_prob}")
        logging.info(f"[INFO]   - Use loss weighting: {self.use_loss_weighting}")
        logging.info(f"[INFO]   - Loss weighting scheme: {self.loss_weighting_scheme}")
        if self.stage is not None:
            logging.info(f"[INFO]   - Two-stage finetune: stage={self.stage}, active={self.active_transformer}")

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
                "model.model_type must be set when peft config is provided. Options: 'flux', 'flux2', 'wan', 'hunyuan'"
            )

        lora_status = (
            f"enabled (dim={self.peft_cfg.dim}, alpha={self.peft_cfg.alpha})"
            if self.peft_cfg is not None
            else "disabled (full fine-tune)"
        )
        logging.info(f"[INFO] LoRA: {lora_status}")

        (self.pipe, self.optimizer, self.device_mesh) = build_model_and_optimizer(
            model_id=self.model_id,
            model_revision=self.model_revision,
            finetune_mode=self.cfg.get("model.mode", "finetune").lower() == "finetune",
            learning_rate=self.learning_rate,
            device=self.device,
            dtype=self.model_dtype,
            compute_dtype=self.compute_dtype,
            cpu_offload=self.cpu_offload,
            fsdp_cfg=fsdp_cfg,
            ddp_cfg=ddp_cfg,
            optimizer_cfg=self.cfg.get("optim.optimizer", {}),
            transformer_engine_linear=self.transformer_engine_linear,
            transformer_engine_fp8_safe_only=self.transformer_engine_fp8,
            fuse_qkv_projections=self.fuse_qkv_projections,
            compact_fused_qkv_projections=self.compact_fused_qkv_projections,
            attention_backend=self.attention_backend,
            pipeline_spec=pipeline_spec,
            peft_cfg=self.peft_cfg,
            model_type=self.model_type,
            active_transformer=self.active_transformer,
        )

        self.model = self.pipe.transformer
        if self.optimize_hunyuan_flash_varlen_mask:
            from nemo_automodel.components.flow_matching.adapters.hunyuan import (
                enable_hunyuan_flash_varlen_mask_optimization,
            )

            if not enable_hunyuan_flash_varlen_mask_optimization():
                raise RuntimeError("Failed to enable Hunyuan flash-varlen mask optimization")
            logging.info("[INFO] Enabled Hunyuan flash-varlen 2D mask optimization")

        self.peft_config = getattr(self.pipe, "_peft_config", None)

        # Resolve sigma range for two-stage finetuning now that the pipeline
        # is loaded and we can read its boundary_ratio config.
        if self.stage is not None:
            if self.boundary_ratio is None:
                pipe_cfg = getattr(self.pipe, "config", None)
                self.boundary_ratio = pipe_cfg.get("boundary_ratio") if pipe_cfg is not None else None
            if self.boundary_ratio is None:
                raise ValueError(
                    "model.stage is set but no boundary_ratio could be resolved. "
                    "Set model.boundary_ratio in YAML, or use a pipeline whose config "
                    "carries boundary_ratio (e.g. Wan-AI/Wan2.2-T2V-A14B-Diffusers)."
                )
            self.boundary_ratio = float(self.boundary_ratio)
            # A boundary outside (0, 1) collapses the stage sigma window to an empty
            # or degenerate range (e.g. boundary_ratio=0.0 with stage=low_noise gives
            # sigma_min=sigma_max=0.0), silently yielding a useless model.
            if not (0.0 < self.boundary_ratio < 1.0):
                raise ValueError(f"model.boundary_ratio must be in (0, 1), got {self.boundary_ratio}")
            if self.stage == "high_noise":
                self.sigma_min = self.boundary_ratio
                self.sigma_max = 1.0
            else:
                self.sigma_min = 0.0
                self.sigma_max = self.boundary_ratio
            logging.info(
                "[INFO]   - Stage sigma range: [%.4f, %.4f] (boundary_ratio=%.4f)",
                self.sigma_min,
                self.sigma_max,
                self.boundary_ratio,
            )

        checkpoint_cfg = self.cfg.get("checkpoint", None)

        self.num_epochs = self.cfg.get("step_scheduler.num_epochs")
        self.log_every = self.cfg.get(
            "step_scheduler.log_every",
            self.cfg.get("step_scheduler.log_remote_every_steps", 5),
        )

        # Strictly require checkpoint config from YAML (no fallback)
        if checkpoint_cfg is None:
            raise ValueError(
                "checkpoint config is required in YAML (enabled, checkpoint_dir, model_save_format, save_consolidated)"
            )

        # Keep declarative checkpoint construction at the typed-config boundary;
        # pre-shard model keys are runtime state owned by the built checkpointer.
        model_state_dict_keys = list(self.model.state_dict().keys())
        self.checkpoint_config = self.cfg.checkpoint
        self.restore_from = checkpoint_cfg.get("restore_from", None)
        self.checkpointer = self.checkpoint_config.build(
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=None,
            model_state_dict_keys=model_state_dict_keys,
        )

        dataloader_config = self.cfg.diffusion_dataloader
        if dataloader_config is None:
            raise ValueError("Diffusion training requires a data.dataloader config")
        dataloader_build = dataloader_config.build(
            dp_rank=self._get_dp_rank(),
            dp_world_size=self._get_dp_group_size(),
            batch_size=self.cfg.get("step_scheduler.local_batch_size"),
        )
        self.dataloader = dataloader_build.dataloader
        self.sampler = dataloader_build.sampler

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
        self.local_batch_size = self.cfg.get("step_scheduler.local_batch_size")
        # Desired global effective batch size across all DP ranks and nodes
        self.global_batch_size = self.cfg.get("step_scheduler.global_batch_size")
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
            global_batch_size=self.cfg.get("step_scheduler.global_batch_size"),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size"),
            dp_size=int(self.dp_size),
            ckpt_every_steps=self.cfg.get("step_scheduler.ckpt_every_steps"),
            save_checkpoint_every_epoch=self.cfg.get("step_scheduler.save_checkpoint_every_epoch", False),
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=int(self.global_step),
            start_epoch=int(self.start_epoch),
            num_epochs=int(self.num_epochs),
            max_steps=max_steps,
        )

        self.load_checkpoint(self.restore_from)

        self.flow_matching_pipeline = self.flow_matching_config.build(
            model_adapter=self.model_adapter,
            device=self.device,
            generator=self.flow_generator,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
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
        perf_window_local_target_tokens = 0
        perf_window_local_context_tokens = 0
        perf_window_local_text_tokens = 0
        perf_window_local_total_tokens = 0
        perf_window_local_dataloader_wait = 0.0
        warmup_steps = max(int(getattr(self, "throughput_warmup_steps", 0)), 0)
        measurement_started = global_step >= warmup_steps
        phase_timer = _PhaseTimer(enabled=bool(getattr(self, "phase_timing", False)), device=self.device)
        batch_wait_started = time.perf_counter()
        if measurement_started and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
                dataloader_wait = max(time.perf_counter() - batch_wait_started, 0.0)
                measure_step = measurement_started
                if measure_step:
                    perf_window_local_dataloader_wait += dataloader_wait
                self.optimizer.zero_grad(set_to_none=True)

                micro_losses = []
                micro_metrics: list[Dict[str, Any]] = []
                prepare_for_grad_accumulation([self.model], pp_enabled=False)
                num_microbatches = len(batch_group)
                for microbatch_idx, micro_batch in enumerate(batch_group):
                    is_final_microbatch = microbatch_idx == num_microbatches - 1
                    if is_final_microbatch:
                        prepare_for_final_backward([self.model], pp_enabled=False)

                    sync_context = get_sync_ctx(
                        self.model,
                        is_final_microbatch,
                        defer_fsdp_grad_sync=getattr(self, "defer_fsdp_grad_sync", True),
                    )
                    with sync_context:
                        try:
                            phase_started = phase_timer.start() if measure_step else None
                            with torch.profiler.record_function("diffusion.forward"):
                                with self._transformer_engine_fp8_context():
                                    _, average_weighted_loss, _, metrics = self.flow_matching_pipeline.step(
                                        model=self.model,
                                        batch=micro_batch,
                                        device=self.device,
                                        dtype=self.compute_dtype,
                                        global_step=global_step,
                                        collect_metrics=False,
                                        check_loss=self.check_loss,
                                    )
                            phase_timer.stop("forward", phase_started)
                        except Exception as exc:
                            logging.info(f"[ERROR] Training step failed at epoch {epoch}, step {num_steps}: {exc}")
                            video_shape = micro_batch.get("video_latents", torch.tensor([])).shape
                            image_shape = micro_batch.get("image_latents", torch.tensor([])).shape
                            text_shape = micro_batch.get("text_embeddings", torch.tensor([])).shape
                            logging.info(
                                "[DEBUG] Batch shapes - video: %s, image: %s, text: %s",
                                video_shape,
                                image_shape,
                                text_shape,
                            )
                            raise

                        # Use average_weighted_loss for backprop (scalar for gradient accumulation)
                        phase_started = phase_timer.start() if measure_step else None
                        with torch.profiler.record_function("diffusion.backward_and_communication"):
                            (average_weighted_loss / num_microbatches).backward()
                        phase_timer.stop("backward", phase_started)
                    micro_losses.append(average_weighted_loss.detach())
                    micro_metrics.append(metrics)

                    if microbatch_idx == 0:
                        prepare_after_first_microbatch()

                phase_started = phase_timer.start() if measure_step else None
                with torch.profiler.record_function("diffusion.grad_clip"):
                    grad_norm = clip_grad_norm(self.clip_grad_max_norm, [self.model], foreach=self.grad_clip_foreach)
                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm
                phase_timer.stop("grad_clip", phase_started)

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

                phase_started = phase_timer.start() if measure_step else None
                with torch.profiler.record_function("diffusion.optimizer"):
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler[0].step(1)
                phase_timer.stop("optimizer", phase_started)

                if measure_step:
                    perf_window_steps += 1
                    perf_window_local_samples += _count_local_batch_group_samples(batch_group)
                    perf_window_local_target_tokens += sum(
                        int(metrics.get("perf/target_token_count", 0)) for metrics in micro_metrics
                    )
                    perf_window_local_context_tokens += sum(
                        int(metrics.get("perf/context_token_count", 0)) for metrics in micro_metrics
                    )
                    perf_window_local_text_tokens += sum(
                        int(metrics.get("perf/text_token_count", 0)) for metrics in micro_metrics
                    )
                    perf_window_local_total_tokens += sum(
                        int(metrics.get("perf/total_token_count", 0)) for metrics in micro_metrics
                    )
                group_loss_mean = float(torch.stack(micro_losses).mean().item())
                epoch_loss += group_loss_mean
                num_steps += 1
                # StepScheduler increments after yielding, so its current value still
                # identifies the update that just ran. Metrics and checkpoint names
                # use the number of completed optimizer steps instead.
                global_step = int(self.step_scheduler.step) + 1

                should_log = (
                    measurement_started
                    and self.log_every
                    and self.log_every > 0
                    and perf_window_steps >= self.log_every
                )
                if should_log:
                    elapsed_seconds, _ = self._elapsed_seconds_since(perf_window_start_time)
                    perf_window_global_samples = self._count_global_samples(perf_window_local_samples)
                    throughput_metrics = _calculate_throughput_metrics(
                        elapsed_seconds=elapsed_seconds,
                        optimizer_steps=perf_window_steps,
                        global_samples=perf_window_global_samples,
                        world_size=self.world_size,
                    )
                    global_token_counts = self._count_global_token_metrics(
                        target=perf_window_local_target_tokens,
                        context=perf_window_local_context_tokens,
                        text=perf_window_local_text_tokens,
                        total=perf_window_local_total_tokens,
                    )
                    throughput_metrics.update(
                        {
                            "target_latent_tokens_per_sec": global_token_counts["target"] / elapsed_seconds,
                            "total_latent_tokens_per_sec": (
                                global_token_counts["target"] + global_token_counts["context"]
                            )
                            / elapsed_seconds,
                            "text_tokens_per_sec": global_token_counts["text"] / elapsed_seconds,
                            "total_tokens_per_sec": global_token_counts["total"] / elapsed_seconds,
                        }
                    )
                    local_phase_seconds = {
                        "dataloader_wait": perf_window_local_dataloader_wait,
                        **phase_timer.elapsed_seconds(),
                    }
                    global_phase_seconds = self._max_global_timing_metrics(local_phase_seconds)
                    timed_steps = max(perf_window_steps, 1)
                    throughput_metrics.update(
                        {
                            "dataloader_wait_time": global_phase_seconds.get("dataloader_wait", 0.0) / timed_steps,
                            "forward_time": global_phase_seconds.get("forward", 0.0) / timed_steps,
                            "backward_time": global_phase_seconds.get("backward", 0.0) / timed_steps,
                            "grad_clip_time": global_phase_seconds.get("grad_clip", 0.0) / timed_steps,
                            "optimizer_time": global_phase_seconds.get("optimizer", 0.0) / timed_steps,
                        }
                    )
                    memory_metrics = self._get_memory_metrics()
                    perf_window_steps = 0
                    perf_window_local_samples = 0
                    perf_window_local_target_tokens = 0
                    perf_window_local_context_tokens = 0
                    perf_window_local_text_tokens = 0
                    perf_window_local_total_tokens = 0
                    perf_window_local_dataloader_wait = 0.0
                    phase_timer.reset()

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
                        "step_time=%.3fs samples_per_sec=%.2f samples_per_sec_per_gpu=%.2f "
                        "target_tokens_per_sec=%.2f total_latent_tokens_per_sec=%.2f "
                        "wait=%.3fs fwd=%.3fs bwd=%.3fs grad_clip=%.3fs opt=%.3fs "
                        "max_allocated=%.2fGB max_reserved=%.2fGB",
                        global_step,
                        epoch,
                        group_loss_mean,
                        avg_loss,
                        self.optimizer.param_groups[0]["lr"],
                        grad_norm,
                        throughput_metrics["step_time"],
                        throughput_metrics["samples_per_sec"],
                        throughput_metrics["samples_per_sec_per_gpu"],
                        throughput_metrics["target_latent_tokens_per_sec"],
                        throughput_metrics["total_latent_tokens_per_sec"],
                        throughput_metrics["dataloader_wait_time"],
                        throughput_metrics["forward_time"],
                        throughput_metrics["backward_time"],
                        throughput_metrics["grad_clip_time"],
                        throughput_metrics["optimizer_time"],
                        memory_metrics["max_memory_allocated_gb"],
                        memory_metrics["max_memory_reserved_gb"],
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

                started_after_warmup = not measurement_started and global_step >= warmup_steps
                if started_after_warmup:
                    measurement_started = True
                    perf_window_steps = 0
                    perf_window_local_samples = 0
                    perf_window_local_target_tokens = 0
                    perf_window_local_context_tokens = 0
                    perf_window_local_text_tokens = 0
                    perf_window_local_total_tokens = 0
                    perf_window_local_dataloader_wait = 0.0
                    phase_timer.reset()

                if should_log or started_after_warmup:
                    perf_window_start_time = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                batch_wait_started = time.perf_counter()

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

    def _count_global_token_metrics(self, *, target: int, context: int, text: int, total: int) -> Dict[str, int]:
        """Sum cached CPU token counts across the data-parallel group."""
        names = ("target", "context", "text", "total")
        counts = torch.tensor(
            [target, context, text, total],
            device=self._get_collective_device(),
            dtype=torch.long,
        )
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self._get_dp_group())
        return dict(zip(names, (int(value) for value in counts.tolist())))

    def _max_global_timing_metrics(self, timings: Dict[str, float]) -> Dict[str, float]:
        """Max-reduce phase seconds across all training ranks."""
        if not timings:
            return {}
        names = tuple(sorted(timings))
        values = torch.tensor(
            [timings[name] for name in names],
            device=self._get_collective_device(),
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX, group=None)
        return dict(zip(names, (float(value) for value in values.tolist())))

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
