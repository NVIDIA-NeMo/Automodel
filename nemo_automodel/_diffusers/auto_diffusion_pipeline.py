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
from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
from nemo_automodel.components.distributed.parallelizer import (
    HunyuanParallelizationStrategy,
    WanParallelizationStrategy,
)
from nemo_automodel.shared.import_utils import safe_import_te
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


def _is_fp8_training_safe_linear(name: str, linear: nn.Linear) -> bool:
    """Return whether a Linear has shapes that are safe for FP8 training kernels."""
    if linear.weight.shape[0] % 16 != 0 or linear.weight.shape[1] % 16 != 0:
        return False
    if name.startswith(("time_text_embed.", "norm_out.")):
        return False
    if name.endswith(".linear") and any(norm_name in name for norm_name in (".norm.", ".norm1.", ".norm1_context.")):
        return False
    return True


def _replace_linear_with_transformer_engine(
    module: nn.Module,
    module_name: str,
    *,
    fp8_safe_only: bool = False,
) -> int:
    """Replace torch Linear modules with Transformer Engine Linear modules."""
    has_te, transformer_engine = safe_import_te()
    if not has_te:
        raise ImportError("model.transformer_engine_linear=true requires transformer_engine.pytorch to be importable")

    te_linear_cls = transformer_engine.pytorch.Linear
    converted = 0
    skipped = 0

    def replace_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, skipped
        for child_name, child in list(parent.named_children()):
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if fp8_safe_only and not _is_fp8_training_safe_linear(child_fqn, child):
                    skipped += 1
                    logger.info(
                        "[TE] Keeping %s.%s as torch.nn.Linear for FP8 shape compatibility; weight shape=%s",
                        module_name,
                        child_fqn,
                        tuple(child.weight.shape),
                    )
                    continue
                te_linear = te_linear_cls(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    params_dtype=child.weight.dtype,
                )
                te_linear.train(child.training)
                with torch.no_grad():
                    te_linear.weight.copy_(child.weight)
                    te_linear.weight.requires_grad_(child.weight.requires_grad)
                    if child.bias is not None:
                        te_linear.bias.copy_(child.bias)
                        te_linear.bias.requires_grad_(child.bias.requires_grad)
                setattr(parent, child_name, te_linear)
                converted += 1
            elif not isinstance(child, te_linear_cls):
                replace_children(child, child_fqn)

    replace_children(module)
    logger.info(
        "[TE] Replaced %d torch.nn.Linear modules with Transformer Engine Linear in %s; skipped=%d fp8_safe_only=%s",
        converted,
        module_name,
        skipped,
        fp8_safe_only,
    )
    return converted


def _remove_child_module(parent: nn.Module, child_name: str) -> int:
    """Remove a child module and return the number of parameters it owned."""
    child = getattr(parent, child_name, None)
    if not isinstance(child, nn.Module):
        return 0

    parameter_count = sum(parameter.numel() for parameter in child.parameters())
    delattr(parent, child_name)
    return parameter_count


def _compact_fused_qkv_projection_modules(module: nn.Module, module_name: str) -> tuple[int, int]:
    """Remove original projection modules that are unused after Diffusers QKV fusion."""
    removed_modules = 0
    removed_parameters = 0

    for child in list(module.modules()):
        if not getattr(child, "fused_projections", False):
            continue

        if hasattr(child, "to_qkv"):
            for projection_name in ("to_q", "to_k", "to_v"):
                parameter_count = _remove_child_module(child, projection_name)
                if parameter_count:
                    removed_modules += 1
                    removed_parameters += parameter_count

        if hasattr(child, "to_added_qkv"):
            for projection_name in ("add_q_proj", "add_k_proj", "add_v_proj"):
                parameter_count = _remove_child_module(child, projection_name)
                if parameter_count:
                    removed_modules += 1
                    removed_parameters += parameter_count

    logger.info(
        "[QKV] Removed %d original projection modules with %s parameters after fused QKV setup in %s",
        removed_modules,
        f"{removed_parameters:,}",
        module_name,
    )
    return removed_modules, removed_parameters


def _fuse_transformer_qkv_projections(module: nn.Module, module_name: str, *, compact: bool = False) -> int:
    """Fuse Diffusers attention QKV projections when the transformer exposes that hook."""
    if not hasattr(module, "fuse_qkv_projections"):
        raise AttributeError(f"model.fuse_qkv_projections=true requires {module_name} to expose fuse_qkv_projections()")

    module.fuse_qkv_projections()
    fused = sum(1 for child in module.modules() if getattr(child, "fused_projections", False))
    logger.info("[QKV] Fused QKV projections for %d attention modules in %s", fused, module_name)
    if compact:
        _compact_fused_qkv_projection_modules(module, module_name)
    return fused


def _create_parallel_manager(manager_args: Dict[str, Any]) -> ParallelManager:
    """
    Factory function to create the appropriate parallel manager based on config.

    Constructs the proper config objects (FSDP2Config / DDPConfig) and, for FSDP2,
    creates the required device mesh before instantiating the manager.  This mirrors
    the pattern used by ``_instantiate_distributed`` in the transformers infrastructure.

    The manager type is determined by the ``_manager_type`` key in *manager_args*:
    - ``'ddp'``: Creates :class:`DDPConfig` + :class:`DDPManager`
    - ``'fsdp2'`` (default): Creates :class:`FSDP2Config`, builds a
      :class:`DeviceMesh` via :func:`create_device_mesh`, then creates
      :class:`FSDP2Manager`

    Args:
        manager_args: Flat dictionary of arguments.  Recognised keys:

            Common:
                ``_manager_type`` (str): ``'fsdp2'`` or ``'ddp'``.
                ``activation_checkpointing`` (bool): Enable activation checkpointing.
                ``backend`` (str): Distributed backend (default ``'nccl'``).

            FSDP2-specific (mesh creation):
                ``world_size`` (int): Total number of processes.
                ``dp_size``, ``dp_replicate_size``, ``tp_size``, ``cp_size``,
                ``pp_size``, ``ep_size`` (int): Parallelism dimensions.

            FSDP2-specific (config):
                ``mp_policy``: :class:`MixedPrecisionPolicy` instance.
                ``sequence_parallel`` (bool), ``tp_plan`` (dict),
                ``offload_policy``, ``defer_fsdp_grad_sync`` (bool).

    Returns:
        Either an FSDP2Manager or DDPManager instance.

    Raises:
        ValueError: If an unknown manager type is specified.
    """
    args = manager_args.copy()
    manager_type = args.pop("_manager_type", "fsdp2").lower()

    if manager_type == "ddp":
        config = DDPConfig(
            activation_checkpointing=args.get("activation_checkpointing", False),
            backend=args.get("backend", "nccl"),
        )
        logger.info("[Parallel] Creating DDPManager with config: %s", config)
        return DDPManager(config)

    elif manager_type == "fsdp2":
        config = FSDP2Config(
            activation_checkpointing=args.get("activation_checkpointing", False),
            mp_policy=args.get("mp_policy", None),
            backend=args.get("backend", "nccl"),
            sequence_parallel=args.get("sequence_parallel", False),
            tp_plan=args.get("tp_plan", None),
            patch_is_packed_sequence=args.get("patch_is_packed_sequence", False),
            offload_policy=args.get("offload_policy", None),
            defer_fsdp_grad_sync=args.get("defer_fsdp_grad_sync", True),
            enable_async_tensor_parallel=args.get("enable_async_tensor_parallel", False),
            enable_compile=args.get("enable_compile", False),
            enable_fsdp2_prefetch=args.get("enable_fsdp2_prefetch", False),
            fsdp2_backward_prefetch_depth=args.get("fsdp2_backward_prefetch_depth", 2),
            fsdp2_forward_prefetch_depth=args.get("fsdp2_forward_prefetch_depth", 1),
        )

        world_size = args.get("world_size") or torch.distributed.get_world_size()
        device_mesh, moe_mesh = create_device_mesh(
            config,
            dp_size=args.get("dp_size"),
            dp_replicate_size=args.get("dp_replicate_size"),
            tp_size=args.get("tp_size", 1),
            pp_size=args.get("pp_size", 1),
            cp_size=args.get("cp_size", 1),
            ep_size=args.get("ep_size", 1),
            world_size=world_size,
        )

        logger.info("[Parallel] Creating FSDP2Manager with config: %s", config)
        return FSDP2Manager(config, device_mesh=device_mesh, moe_mesh=moe_mesh)

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
        if hasattr(manager, "maybe_compile"):
            manager.maybe_compile(parallel_module)
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
        peft_cfg=None,
        model_type=None,
        transformer_engine_linear: bool = False,
        transformer_engine_fp8_safe_only: bool = False,
        fuse_qkv_projections: bool = False,
        compact_fused_qkv_projections: bool = False,
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
            peft_cfg: PeftConfig instance or None. When provided, LoRA is injected
                before _apply_parallelization() (FSDP2 wrapping). Base weights
                are frozen after FSDP2; LoRA params are collected pre-FSDP2 and stored on pipe.
            model_type: "flux" | "wan" | "hunyuan". Required when peft_cfg is provided.
            transformer_engine_linear: Whether to replace torch.nn.Linear modules in the transformer with TE Linear.
            transformer_engine_fp8_safe_only: Whether to skip TE Linear conversion for known FP8-incompatible modules.
            fuse_qkv_projections: Whether to call Diffusers QKV projection fusion on the transformer.
            compact_fused_qkv_projections: Whether to remove original projection modules after QKV fusion.
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

        if compact_fused_qkv_projections and not fuse_qkv_projections:
            raise ValueError("model.compact_fused_qkv_projections=true requires model.fuse_qkv_projections=true")

        if fuse_qkv_projections:
            for name, module in _iter_pipeline_modules(pipe):
                if name == "transformer" and (not components_to_load or name in components_to_load):
                    _fuse_transformer_qkv_projections(module, name, compact=compact_fused_qkv_projections)

        if transformer_engine_linear:
            for name, module in _iter_pipeline_modules(pipe):
                if name == "transformer" and (not components_to_load or name in components_to_load):
                    _replace_linear_with_transformer_engine(
                        module,
                        name,
                        fp8_safe_only=transformer_engine_fp8_safe_only,
                    )

        if peft_cfg is not None:
            # ── LoRA path ─────────────────────────────────────────────────────
            # LoRA injection MUST run before _apply_parallelization (FSDP2).
            # FSDP2 must see the final module structure (with LoRA-patched linears)
            # to correctly shard both base weights and LoRA weights.
            # Pre-FSDP2 lora_params refs are stored on pipe and remain valid
            # after wrapping (FSDP2 preserves original Parameter objects).
            if model_type is None:
                raise ValueError("model_type must be set when peft_cfg is provided. Options: 'flux', 'wan', 'hunyuan'")
            import dataclasses

            from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules

            pipe._lora_params = {}
            pipe._peft_config = None

            for name, module in _iter_pipeline_modules(pipe):
                if name == "transformer":
                    # Pre-inject hook: Wan fuses to_q/to_k/to_v for inference
                    # efficiency — must unfuse before injection so individual
                    # Linear modules are visible to the module matcher.
                    if model_type == "wan":
                        for block in module.blocks:
                            block.attn1.unfuse_projections()
                            block.attn2.unfuse_projections()
                        logger.info("[LoRA] Wan: unfused attention projection groups")

                    # Ensure LoRA weights are bf16 to match base weights in the
                    # FSDP2 unit — mixed dtypes cause reduce-scatter to malfunction.
                    cfg = peft_cfg
                    if cfg.lora_dtype is None:
                        cfg = dataclasses.replace(cfg, lora_dtype=torch.bfloat16)

                    # skip_freeze=True: global base-weight freeze happens after
                    # FSDP2 wrapping (see below). FSDP2 must see all params as
                    # trainable during fully_shard() so gradient reduction is set
                    # up correctly for LoRA params.
                    apply_lora_to_linear_modules(module, cfg, skip_freeze=True)

                    lora_params = [p for n, p in module.named_parameters() if "lora_" in n and p.requires_grad]
                    pipe._lora_params[name] = lora_params
                    pipe._peft_config = cfg
                    logger.info(
                        "[LoRA] Stored %d lora_param tensors before FSDP2",
                        len(lora_params),
                    )
        else:
            # If loading for training, ensure the target module parameters are trainable
            if load_for_training:
                for name, module in _iter_pipeline_modules(pipe):
                    if not components_to_load or name in components_to_load:
                        logger.info("[INFO] Ensuring params trainable: %s", name)
                        _ensure_params_trainable(module, module_name=name)

        if peft_cfg is not None:
            transformer_parallel_args = (parallel_scheme or {}).get("transformer", {})
            manager_type = str(transformer_parallel_args.get("_manager_type", "fsdp2")).lower()
            if manager_type == "ddp":
                for param_name, param in pipe.transformer.named_parameters():
                    if "lora_" not in param_name and param.requires_grad:
                        param.requires_grad_(False)
                logger.info("[LoRA] Froze base weights before DDP wrapping")

        # Apply parallelization (FSDP2 or DDP)
        # FSDP2 LoRA: all params are trainable when fully_shard() runs so FSDP2
        # sets up gradient reduction for lora_A/lora_B correctly. Freeze happens below.
        # DDP LoRA: base weights are frozen before wrapping so DDP only reduces LoRA gradients.
        created_managers = _apply_parallelization(pipe, parallel_scheme)

        # Freeze base weights after FSDP2 wrapping — mirrors the LLM pattern in
        # nemo_automodel/_transformers/infrastructure.py lines 513-518.
        # FSDP2 must see all params as trainable during fully_shard(); freezing
        # before wrapping causes gradient reduction to malfunction for LoRA params.
        # For DDP this is harmless because the base was already frozen before wrapping.
        if peft_cfg is not None:
            for name, param in pipe.transformer.named_parameters():
                if "lora_" not in name and param.requires_grad:
                    param.requires_grad_(False)
            logger.info("[LoRA] Froze base weights after parallelization")

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
        transformer_engine_linear: bool = False,
        transformer_engine_fp8_safe_only: bool = False,
        fuse_qkv_projections: bool = False,
        compact_fused_qkv_projections: bool = False,
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
            transformer_engine_linear: Whether to replace torch.nn.Linear modules in the transformer with TE Linear.
            transformer_engine_fp8_safe_only: Whether to skip TE Linear conversion for known FP8-incompatible modules.
            fuse_qkv_projections: Whether to call Diffusers QKV projection fusion on the transformer.
            compact_fused_qkv_projections: Whether to remove original projection modules after QKV fusion.
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

        if compact_fused_qkv_projections and not fuse_qkv_projections:
            raise ValueError("model.compact_fused_qkv_projections=true requires model.fuse_qkv_projections=true")

        if fuse_qkv_projections:
            for name, module in _iter_pipeline_modules(pipe):
                if name == "transformer" and (not components_to_load or name in components_to_load):
                    _fuse_transformer_qkv_projections(module, name, compact=compact_fused_qkv_projections)

        # Make parameters trainable (always true for from_config / pretraining)
        for name, module in _iter_pipeline_modules(pipe):
            if not components_to_load or name in components_to_load:
                if transformer_engine_linear and name == "transformer":
                    _replace_linear_with_transformer_engine(
                        module,
                        name,
                        fp8_safe_only=transformer_engine_fp8_safe_only,
                    )
                _ensure_params_trainable(module, module_name=name)

        # Apply parallelization (FSDP2 or DDP)
        created_managers = _apply_parallelization(pipe, parallel_scheme)

        return pipe, created_managers
