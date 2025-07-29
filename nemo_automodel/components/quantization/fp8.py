# Copyright (c) NVIDIA CORPORATION and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from functools import partial
from typing import Optional, List

from nemo_automodel.shared.import_utils import MISSING_TORCHAO_MSG

logger = logging.getLogger(__name__)

# Safe imports for torchao components
try:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training, precompute_float8_dynamic_scale_for_fsdp
    HAVE_TORCHAO = True
except ImportError:
    HAVE_TORCHAO = False


def _has_cuda_capability(major: int, minor: int) -> bool:
    """Check if CUDA device has required compute capability."""
    if not torch.cuda.is_available():
        return False
    
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability >= (major, minor)


def _module_filter_fn(module, name, filter_fqns: List[str] = None):
    """
    Filter function to exclude certain modules from FP8 conversion.

    Args:
        module: The module to check
        name: Fully qualified name of the module
        filter_fqns: List of FQNs to filter out

    Returns:
        True if module should be converted to FP8, False otherwise
    """
    if filter_fqns is None:
        filter_fqns = []
    
    # Skip modules in filter list
    for fqn in filter_fqns:
        if fqn in name:
            return False
    
    # Always skip non-linear layers
    if not isinstance(module, nn.Linear):
        return False
    
    # Skip layers with dimensions not divisible by 16
    if hasattr(module, 'weight'):
        weight = module.weight
        if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
            logger.info(f"Skipping fp8 for layer {name} with weight shape {weight.shape}")
            return False
    
    return True


def apply_fp8_to_model(
    model: nn.Module, 
    filter_fqns: Optional[List[str]] = None,
    recipe_name: Optional[str] = None,
    force_recompute_fp8_weight_in_bwd: bool = False,
    enable_fsdp_float8_all_gather: bool = False,
    emulate: bool = False
) -> nn.Module:
    """
    Apply FP8 quantization to a PyTorch model using torchao.
    
    Args:
        model: The model to convert
        config: Float8LinearConfig from torchao (if None, will be created)
        filter_fqns: List of module names to exclude from FP8 conversion
        recipe_name: Recipe name for FP8 configuration ("tensorwise", "rowwise", etc.)
        dp_shard_enabled: Whether data parallel sharding is enabled
        force_recompute_fp8_weight_in_bwd: Whether to force recompute FP8 weight in backward pass
        enable_fsdp_float8_all_gather: Whether to enable FSDP FP8 all-gather
        emulate: Use emulation instead of hardware acceleration (for testing on older GPUs)
        
    Returns:
        The model with FP8 linear layers (modified in-place)
        
    Raises:
        ImportError: If torchao is not installed
        ValueError: If hardware doesn't support FP8 and emulation is disabled
    """
    # Check if torchao is available
    if not HAVE_TORCHAO:
        raise ImportError(MISSING_TORCHAO_MSG)
    
    # Handle config creation or recipe-based configuration
    if recipe_name is not None and recipe_name != "tensorwise":
        config = Float8LinearConfig.from_recipe_name(recipe_name)
        logger.info(f"Using FP8 recipe: {recipe_name}")
        
        # Enable inductor precision cast emulation for rowwise recipe
        if recipe_name == "rowwise":
            torch._inductor.config.emulate_precision_casts = True
            logger.debug("Enabled torch._inductor.config.emulate_precision_casts for rowwise recipe")
    else:
        # Manual configuration for tensorwise scaling        
        config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            force_recompute_fp8_weight_in_bwd=force_recompute_fp8_weight_in_bwd,
            emulate=emulate,
        )
        logger.info("Using FP8 tensorwise scaling")
    
    # Check hardware capability if not using emulation
    config_emulate = getattr(config, 'emulate', emulate)
    if not _has_cuda_capability(8, 9) and not config_emulate:
        raise ValueError(
            "FP8 is only supported on SM89 or later GPUs (H100+). "
            "To enable testing on older hardware, set emulate=True in Float8LinearConfig or pass emulate=True."
        )
    
    # Create module filter function
    if filter_fqns is None:
        filter_fqns = []
    filter_fn = partial(_module_filter_fn, filter_fqns=filter_fqns)
    
    # Convert model to use FP8 linear layers
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=filter_fn,
    )

    logger.info(
        f"**Successfully converted model to FP8. \n"
        f"**Recipe: {recipe_name or 'tensorwise'}\n"
        f"**FSDP all-gather enabled: {config.enable_fsdp_float8_all_gather}\n"
        f"**Force recompute FP8 weight in backward pass: {config.force_recompute_fp8_weight_in_bwd}\n"
    )
    verify_fp8_conversion(model)
    
    return model


def verify_fp8_conversion(model: nn.Module) -> dict:
    """
    Verify that FP8 conversion was successful by counting converted modules.
    
    Args:
        model: The model to verify
        
    Returns:
        Dict with conversion statistics
    """
    from torchao.float8.float8_linear import Float8Linear
    
    total_linear = 0
    fp8_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Count both nn.Linear and Float8Linear as linear layers
        if isinstance(module, nn.Linear):
            total_linear += 1
            logger.debug(f"Found nn.Linear: {name} ({module_type})")
            # Check if it's a Float8Linear by comparing class names or checking attributes
            if isinstance(module, Float8Linear):
                fp8_modules.append({
                    'name': name,
                    'type': module_type,
                    'weight_shape': list(module.weight.shape) if hasattr(module, 'weight') else None
                })
                logger.debug(f"Found Float8Linear: {name} ({module_type})")
            elif module_type == 'Float8Linear':
                # Fallback: check by class name in case isinstance fails
                fp8_modules.append({
                    'name': name,
                    'type': module_type,
                    'weight_shape': list(module.weight.shape) if hasattr(module, 'weight') else None
                })
                logger.debug(f"Found Float8Linear by name: {name} ({module_type})")
    
    logger.info(f"-"*50)
    logger.info(f"FP8 verification: {len(fp8_modules)} Float8Linear modules, {total_linear} total linear modules")
    logger.info(f"-"*50+"\n")
    return {
        'linear_count': total_linear,
        'fp8_count': len(fp8_modules),
        'conversion_rate': (len(fp8_modules) / total_linear * 100) if total_linear > 0 else 0,
        'fp8_modules': fp8_modules,
        'success': len(fp8_modules) > 0
    }


def precompute_fp8_scales_for_fsdp(model: nn.Module) -> None:
    """
    Precompute FP8 scales for FSDP optimization.
    
    This function should be called after FSDP setup to optimize
    scale computation for distributed training.
    
    Args:
        model: Model with FP8 linear layers
    """
    precompute_float8_dynamic_scale_for_fsdp(model)