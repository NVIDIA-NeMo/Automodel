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

"""
NeMo Auto Diffusion Pipeline - Unified pipeline wrapper for all diffusion models.

This module provides a single pipeline class that handles:
- Loading from pretrained weights (finetuning) via DiffusionPipeline auto-detection
- Loading from config with random weights (pretraining) via YAML-specified transformer class
- FSDP2/DDP parallelization for distributed training
- Gradient checkpointing for memory efficiency

Usage:
    # Finetuning (from_pretrained) - no pipeline_spec needed
    pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        load_for_training=True,
        parallel_scheme={"transformer": manager_args},
    )

    # Pretraining (from_config) - pipeline_spec required in YAML
    pipe, managers = NeMoAutoDiffusionPipeline.from_config(
        "black-forest-labs/FLUX.1-dev",
        pipeline_spec={
            "transformer_cls": "FluxTransformer2DModel",
            "subfolder": "transformer",
        },
        parallel_scheme={"transformer": manager_args},
    )
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.parallelizer import (
    HunyuanParallelizationStrategy,
    WanParallelizationStrategy,
)
from nemo_automodel.shared.utils import dtype_from_str

# diffusers is an optional dependency
try:
    from diffusers import DiffusionPipeline

    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline = object


logger = logging.getLogger(__name__)

# Type alias for parallel managers
ParallelManager = Union[FSDP2Manager, DDPManager]


@dataclass
class PipelineSpec:
    """
    YAML-driven specification for loading a diffusion pipeline.

    This is required for from_config (pretraining with random weights).
    Not needed for from_pretrained (finetuning).

    Example YAML:
        pipeline_spec:
            transformer_cls: "FluxTransformer2DModel"
            pipeline_cls: "FluxPipeline"  # Optional
            subfolder: "transformer"
            load_full_pipeline: false
            enable_gradient_checkpointing: true
    """

    # Required for from_config: transformer class name from diffusers
    transformer_cls: str = ""

    # Optional: full pipeline class name (for loading VAE, text encoders, etc.)
    pipeline_cls: Optional[str] = None

    # Subfolder for transformer weights in HF repo
    subfolder: str = "transformer"

    # For from_config: whether to load full pipeline or just transformer
    load_full_pipeline: bool = False

    # Training optimizations
    enable_gradient_checkpointing: bool = True
    low_cpu_mem_usage: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PipelineSpec":
        """Create PipelineSpec from YAML dict."""
        if d is None:
            return cls()
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def validate_for_from_config(self):
        """Validate spec has required fields for from_config."""
        if not self.transformer_cls:
            raise ValueError(
                "pipeline_spec.transformer_cls is required for from_config. "
                "Example YAML:\n"
                "  pipeline_spec:\n"
                "    transformer_cls: 'FluxTransformer2DModel'\n"
                "    subfolder: 'transformer'"
            )


def _import_diffusers_class(class_name: str):
    """Dynamically import a class from diffusers by name."""
    import diffusers

    if not hasattr(diffusers, class_name):
        raise ImportError(
            f"Class '{class_name}' not found in diffusers. Check pipeline_spec.transformer_cls in your YAML config."
        )
    return getattr(diffusers, class_name)


def _init_parallelizer():
    """Register custom parallelization strategies."""
    parallelizer.PARALLELIZATION_STRATEGIES["WanTransformer3DModel"] = WanParallelizationStrategy()
    parallelizer.PARALLELIZATION_STRATEGIES["HunyuanVideo15Transformer3DModel"] = HunyuanParallelizationStrategy()


def _choose_device(device: Optional[torch.device]) -> torch.device:
    """Choose device, defaulting to CUDA with LOCAL_RANK if available."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _iter_pipeline_modules(pipe) -> Iterable[Tuple[str, nn.Module]]:
    """Iterate over nn.Module components in a pipeline."""
    # Prefer Diffusers' components registry when available
    if hasattr(pipe, "components") and isinstance(pipe.components, dict):
        for name, value in pipe.components.items():
            if isinstance(value, nn.Module):
                yield name, value
        return

    # Fallback: inspect attributes
    for name in dir(pipe):
        if name.startswith("_"):
            continue
        try:
            value = getattr(pipe, name)
        except Exception:
            continue
        if isinstance(value, nn.Module):
            yield name, value


def _move_module_to_device(module: nn.Module, device: torch.device, torch_dtype: Any) -> None:
    """Move module to device with specified dtype."""
    dtype: Optional[torch.dtype]
    if torch_dtype == "auto":
        dtype = None
    else:
        dtype = dtype_from_str(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    if dtype is not None:
        module.to(device=device, dtype=dtype)
    else:
        module.to(device=device)


def _ensure_params_trainable(module: nn.Module, module_name: Optional[str] = None) -> int:
    """
    Ensure that all parameters in the given module are trainable.

    Returns the number of parameters marked trainable.
    """
    num_trainable_parameters = 0
    for parameter in module.parameters():
        parameter.requires_grad = True
        num_trainable_parameters += parameter.numel()
    if module_name is None:
        module_name = module.__class__.__name__
    logger.info("[Trainable] %s: %s parameters set requires_grad=True", module_name, f"{num_trainable_parameters:,}")
    return num_trainable_parameters


def _create_parallel_manager(manager_args: Dict[str, Any]) -> ParallelManager:
    """
    Factory function to create the appropriate parallel manager based on config.

    The manager type is determined by the '_manager_type' key in manager_args:
    - 'ddp': Creates a DDPManager for standard Distributed Data Parallel
    - 'fsdp2' (default): Creates an FSDP2Manager for Fully Sharded Data Parallel

    Args:
        manager_args: Dictionary of arguments for the manager. Must include '_manager_type'
                     key to specify which manager to create. The '_manager_type' key is
                     removed before passing args to the manager constructor.

    Returns:
        Either an FSDP2Manager or DDPManager instance.

    Raises:
        ValueError: If an unknown manager type is specified.
    """
    # Make a copy to avoid modifying the original dict
    args = manager_args.copy()
    manager_type = args.pop("_manager_type", "fsdp2").lower()

    if manager_type == "ddp":
        logger.info("[Parallel] Creating DDPManager with args: %s", args)
        return DDPManager(**args)
    elif manager_type == "fsdp2":
        logger.info("[Parallel] Creating FSDP2Manager with args: %s", args)
        return FSDP2Manager(**args)
    else:
        raise ValueError(f"Unknown manager type: '{manager_type}'. Expected 'ddp' or 'fsdp2'.")


def _apply_parallelization(
    pipe,
    parallel_scheme: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, ParallelManager]:
    """Apply FSDP2/DDP parallelization to pipeline components."""
    created_managers: Dict[str, ParallelManager] = {}
    if parallel_scheme is None:
        return created_managers

    assert torch.distributed.is_initialized(), "Distributed environment must be initialized for parallelization"
    _init_parallelizer()

    for comp_name, comp_module in _iter_pipeline_modules(pipe):
        manager_args = parallel_scheme.get(comp_name)
        if manager_args is None:
            continue
        logger.info("[INFO] Applying parallelization to %s", comp_name)
        manager = _create_parallel_manager(manager_args)
        created_managers[comp_name] = manager
        parallel_module = manager.parallelize(comp_module)
        setattr(pipe, comp_name, parallel_module)

    return created_managers


class NeMoAutoDiffusionPipeline:
    """
    Unified diffusion pipeline wrapper for all model types.

    This class serves dual purposes:
    1. Provides class methods (from_pretrained, from_config) for loading pipelines
    2. Acts as a minimal wrapper when load_full_pipeline=False (transformer-only mode)

    Two loading paths:
    - from_pretrained: Uses DiffusionPipeline auto-detection (for finetuning)
      No pipeline_spec needed - pipeline type is auto-detected from model_index.json

    - from_config: Uses YAML-specified transformer class (for pretraining)
      Requires pipeline_spec with transformer_cls in YAML config

    Features:
    - Accepts a per-component mapping from component name to parallel manager init args
    - Moves all nn.Module components to the chosen device/dtype
    - Parallelizes only components present in the mapping by constructing a manager per component
    - Supports both FSDP2Manager and DDPManager via '_manager_type' key in config
    - Gradient checkpointing support for memory efficiency

    parallel_scheme:
    - Dict[str, Dict[str, Any]]: component name -> kwargs for parallel manager
    - Each component's kwargs should include '_manager_type': 'fsdp2' or 'ddp' (defaults to 'fsdp2')
    """

    def __init__(self, transformer=None, **components):
        """
        Initialize NeMoAutoDiffusionPipeline.

        Args:
            transformer: The transformer model instance
            **components: Additional pipeline components (vae, text_encoder, etc.)
        """
        self.transformer = transformer
        for k, v in components.items():
            setattr(self, k, v)
        # Create components dict for compatibility with _iter_pipeline_modules
        self._components = {"transformer": transformer, **components}

    @property
    def components(self) -> Dict[str, Any]:
        """Return components dict for compatibility."""
        return {k: v for k, v in self._components.items() if v is not None}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        parallel_scheme: Optional[Dict[str, Dict[str, Any]]] = None,
        device: Optional[torch.device] = None,
        torch_dtype: Any = torch.bfloat16,
        move_to_device: bool = True,
        load_for_training: bool = False,
        components_to_load: Optional[Iterable[str]] = None,
        enable_gradient_checkpointing: bool = True,
        **kwargs,
    ) -> Tuple[DiffusionPipeline, Dict[str, ParallelManager]]:
        """
        Load pipeline from pretrained weights using DiffusionPipeline auto-detection.

        This method auto-detects the pipeline type from model_index.json and loads
        all components. Use this for finetuning existing models.

        No pipeline_spec is needed - the pipeline type is determined automatically.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            parallel_scheme: Dict mapping component names to parallel manager kwargs.
                           Each component's kwargs should include '_manager_type': 'fsdp2' or 'ddp'
            device: Device to load model to
            torch_dtype: Data type for model parameters
            move_to_device: Whether to move modules to device
            load_for_training: Whether to make parameters trainable
            components_to_load: Which components to process (default: all)
            enable_gradient_checkpointing: Enable gradient checkpointing for transformer
            **kwargs: Additional arguments passed to DiffusionPipeline.from_pretrained

        Returns:
            Tuple of (DiffusionPipeline, Dict[str, ParallelManager])
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "diffusers is required for NeMoAutoDiffusionPipeline.from_pretrained. "
                "Install with: pip install nemo_automodel[diffusion]"
            )

        logger.info("[INFO] Loading pipeline from pretrained: %s", pretrained_model_name_or_path)

        # Use DiffusionPipeline.from_pretrained for auto-detection
        pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        logger.info("[INFO] Loaded pipeline type: %s", type(pipe).__name__)

        # Decide device
        dev = _choose_device(device)

        # Move modules to device/dtype first (helps avoid initial OOM during sharding)
        if move_to_device:
            for name, module in _iter_pipeline_modules(pipe):
                if not components_to_load or name in components_to_load:
                    logger.info("[INFO] Moving module: %s to device/dtype", name)
                    _move_module_to_device(module, dev, torch_dtype)

        # Enable gradient checkpointing if configured
        if enable_gradient_checkpointing:
            if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
                pipe.transformer.enable_gradient_checkpointing()
                logger.info("[INFO] Enabled gradient checkpointing for transformer")

        # If loading for training, ensure the target module parameters are trainable
        if load_for_training:
            for name, module in _iter_pipeline_modules(pipe):
                if not components_to_load or name in components_to_load:
                    logger.info("[INFO] Ensuring params trainable: %s", name)
                    _ensure_params_trainable(module, module_name=name)

        # Apply parallelization (FSDP2 or DDP)
        created_managers = _apply_parallelization(pipe, parallel_scheme)

        return pipe, created_managers

    @classmethod
    def from_config(
        cls,
        model_id: str,
        pipeline_spec: Dict[str, Any],
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        parallel_scheme: Optional[Dict[str, Dict[str, Any]]] = None,
        move_to_device: bool = True,
        components_to_load: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Tuple["NeMoAutoDiffusionPipeline", Dict[str, ParallelManager]]:
        """
        Initialize pipeline with random weights using YAML-specified transformer class.

        This method uses the transformer_cls from pipeline_spec to create a model
        with random weights. Use this for pretraining from scratch.

        Requires pipeline_spec in YAML config with at least:
            pipeline_spec:
                transformer_cls: "FluxTransformer2DModel"  # or WanTransformer3DModel, etc.
                subfolder: "transformer"

        Args:
            model_id: HuggingFace model ID or local path (for loading config)
            pipeline_spec: Dict from YAML config with transformer_cls, subfolder, etc.
            torch_dtype: Data type for model parameters
            device: Device to load model to
            parallel_scheme: Dict mapping component names to parallel manager kwargs
            move_to_device: Whether to move modules to device
            components_to_load: Which components to process (default: all)
            **kwargs: Additional arguments

        Returns:
            Tuple of (NeMoAutoDiffusionPipeline or DiffusionPipeline, Dict[str, ParallelManager])
        """
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError(
                "diffusers is required for NeMoAutoDiffusionPipeline.from_config. "
                "Install with: pip install nemo_automodel[diffusion]"
            )

        # Parse and validate pipeline spec
        spec = PipelineSpec.from_dict(pipeline_spec)
        spec.validate_for_from_config()

        logger.info("[INFO] Initializing pipeline from config with random weights")
        logger.info("[INFO] Model ID: %s", model_id)
        logger.info("[INFO] Transformer class: %s", spec.transformer_cls)

        # Dynamically import transformer class from diffusers
        TransformerCls = _import_diffusers_class(spec.transformer_cls)

        # Load config from the model_id
        logger.info("[INFO] Loading config from %s/%s", model_id, spec.subfolder)
        config = TransformerCls.load_config(model_id, subfolder=spec.subfolder)

        # Initialize transformer with random weights
        logger.info("[INFO] Creating %s with random weights", spec.transformer_cls)
        transformer = TransformerCls.from_config(config)
        transformer = transformer.to(torch_dtype)

        # Decide device
        dev = _choose_device(device)

        # Either load full pipeline or just use transformer
        if spec.load_full_pipeline and spec.pipeline_cls:
            # Load full pipeline with random transformer injected
            PipelineCls = _import_diffusers_class(spec.pipeline_cls)
            logger.info("[INFO] Loading full pipeline %s with random transformer", spec.pipeline_cls)
            pipe = PipelineCls.from_pretrained(
                model_id,
                transformer=transformer,
                torch_dtype=torch_dtype,
            )

            # Move all modules to device
            if move_to_device:
                for name, module in _iter_pipeline_modules(pipe):
                    if not components_to_load or name in components_to_load:
                        logger.info("[INFO] Moving module: %s to device/dtype", name)
                        _move_module_to_device(module, dev, torch_dtype)
        else:
            # Transformer only mode - use this class as minimal wrapper
            if move_to_device:
                transformer = transformer.to(dev)
            pipe = cls(transformer=transformer)

        # Enable gradient checkpointing if configured
        if spec.enable_gradient_checkpointing:
            target_transformer = getattr(pipe, "transformer", transformer)
            if hasattr(target_transformer, "enable_gradient_checkpointing"):
                target_transformer.enable_gradient_checkpointing()
                logger.info("[INFO] Enabled gradient checkpointing for transformer")

        # Make parameters trainable (always true for from_config / pretraining)
        for name, module in _iter_pipeline_modules(pipe):
            if not components_to_load or name in components_to_load:
                _ensure_params_trainable(module, module_name=name)

        # Apply parallelization (FSDP2 or DDP)
        created_managers = _apply_parallelization(pipe, parallel_scheme)

        return pipe, created_managers
