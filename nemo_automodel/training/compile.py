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

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class CompileConfig:
    """Configuration for torch.compile settings."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize compile configuration.

        Args:
            enabled: Whether to enable torch.compile.
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
            fullgraph: Whether to compile the entire graph.
            dynamic: Whether to enable dynamic shapes.
            backend: Backend to use for compilation. If None, uses TORCH_COMPILE_BACKEND env var.
            options: Additional options to pass to torch.compile.
        """
        self.enabled = enabled
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        default_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        self.backend = backend if backend is not None else default_backend
        self.options = options or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "backend": self.backend,
            "options": self.options,
        }


def disable_flash_attention_compilation(model: nn.Module) -> nn.Module:
    """Selectively disable torch.compile for Flash Attention functions while keeping FA enabled.
    
    This approach keeps Flash Attention v2 active for performance but excludes the problematic
    Flash Attention functions from compilation to avoid FakeTensor/varlen compatibility issues.
    
    Args:
        model: The model to modify.
        
    Returns:
        The model with Flash Attention functions excluded from compilation.
    """

    
    modified_modules = 0
    
    # First, try to disable Flash Attention functions at the package level
    try:
        import flash_attn
        if hasattr(flash_attn, '_flash_attn_varlen_forward'):
            flash_attn._flash_attn_varlen_forward = torch._dynamo.disable(flash_attn._flash_attn_varlen_forward)
            logger.debug("Disabled compilation for flash_attn._flash_attn_varlen_forward")
        if hasattr(flash_attn, '_flash_attn_forward'):
            flash_attn._flash_attn_forward = torch._dynamo.disable(flash_attn._flash_attn_forward)
            logger.debug("Disabled compilation for flash_attn._flash_attn_forward")
    except ImportError:
        logger.debug("flash_attn package not available for function-level disabling")
    
    # Disable compilation for Flash Attention modules and functions
    for name, module in model.named_modules():
        should_disable = False
        
        # Check for Flash Attention related modules/functions
        if any(pattern in name.lower() for pattern in ['flash', 'attention']):
            if hasattr(module, '_attn_implementation') and module._attn_implementation == 'flash_attention_2':
                should_disable = True
            elif 'flash' in name.lower():
                should_disable = True
        
        # Also disable for modules that might call Flash Attention functions
        if hasattr(module, '__class__'):
            class_name = module.__class__.__name__.lower()
            if 'attention' in class_name and hasattr(module, '_attn_implementation'):
                if getattr(module, '_attn_implementation', None) == 'flash_attention_2':
                    should_disable = True
        
        # Also check for specific Flash Attention modules by looking for key methods
        if hasattr(module, 'forward'):
            # Look for modules that might call flash attention functions
            try:
                import inspect
                source = inspect.getsource(module.forward)
                if 'flash_attn' in source or '_flash_attn' in source:
                    should_disable = True
                    logger.debug(f"Found flash_attn usage in {name}")
            except (OSError, TypeError):
                # Can't get source code, continue with other checks
                pass
        
        if should_disable:
            # Wrap forward method to disable compilation
            if hasattr(module, 'forward') and not hasattr(module, '_compile_disabled'):
                original_forward = module.forward
                
                # Create a wrapper function that preserves the original function in closure
                def make_disabled_forward(orig_func):
                    @torch._dynamo.disable
                    def disabled_forward(*args, **kwargs):
                        return orig_func(*args, **kwargs)
                    return disabled_forward
                
                # Replace forward method with disabled version
                module.forward = make_disabled_forward(original_forward)
                module._compile_disabled = True  # Mark as already processed
                module._original_forward = original_forward  # Keep reference to original
                
                logger.debug(f"Disabled compilation for Flash Attention module: {name}")
                modified_modules += 1
    
    if modified_modules > 0:
        logger.info(
            f"Disabled torch.compile for {modified_modules} Flash Attention modules - "
            f"keeping FA v2 enabled but excluding from compilation"
        )
    else:
        logger.info("No Flash Attention modules found to exclude from compilation")
    
    return model


def compile_model(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile the model following torchtune's approach: try full model first, then selective fallback.
    
    Compilation strategy (following torchtune pattern):
    1. Try full model compilation first (most efficient when it works)
    2. If full model compilation fails, fall back to selective layer compilation
    3. If selective compilation also fails, gracefully return the original model
    
    Args:
        model: The model to compile.
        config: Compile configuration.
        
    Returns:
        The compiled model, partially compiled model, or original model if compilation fails.
    """
    if not config.enabled:
        logger.info("torch.compile is disabled")
        return model

    # Selectively disable compilation for Flash Attention functions while keeping FA enabled
    # TODO: This is a temporary solution to disable compilation for Flash Attention functions while keeping FA enabled
    model = disable_flash_attention_compilation(model)

    # Prepare torch.compile arguments
    options_dict = config.options.to_dict() if hasattr(config.options, 'to_dict') else dict(config.options)
    compile_kwargs = {
        "mode": config.mode,
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if config.backend is not None:
        compile_kwargs["backend"] = config.backend
    compile_kwargs.update(options_dict)

    logger.info(f"Starting compilation with backend={config.backend}, mode={config.mode}, dynamic={config.dynamic}")

    # Strategy 1: Try full model compilation
    logger.info("Attempting full model compilation...")
    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        #TODO: Test compilation with a simple forward pass to ensure it works
        logger.info("Full model compilation successful - using compiled model")
        return compiled_model
    except Exception as e:
        logger.warning(f"Full model compilation failed: {type(e).__name__}: {e}")
        logger.info("Falling back to selective layer compilation...")

    # Strategy 2: Fall back to selective layer compilation (torchtune fallback)
    try:
        return _compile_selective_layers(model, compile_kwargs)
    except Exception as e:
        logger.error(f"Selective layer compilation failed: {type(e).__name__}: {e}")
        logger.info("Compilation failed completely - returning original model")
        return model


def _is_transformer_block(module: nn.Module) -> bool:
    """Check if a module looks like a transformer block by examining its structure.
    
    This is a general approach that works with most HuggingFace LLMs by identifying
    modules that have the typical transformer block structure.
    """
    class_name = module.__class__.__name__
    
    # First check: Does the class name suggest it's a transformer block?
    name_patterns = [
        'Block', 'Layer', 'DecoderLayer', 'EncoderLayer', 'TransformerBlock',
        'TransformerLayer', 'DecoderBlock', 'EncoderBlock'
    ]
    
    has_block_name = any(pattern in class_name for pattern in name_patterns)
    
    if not has_block_name:
        return False
    
    # Second check: Does it have the typical transformer block submodules?
    child_names = [name for name, _ in module.named_children()]
    child_names_lower = [name.lower() for name in child_names]
    
    # Look for attention mechanism
    has_attention = any(
        'attn' in name or 'attention' in name 
        for name in child_names_lower
    )
    
    # Look for feedforward/MLP mechanism
    has_feedforward = any(
        'mlp' in name or 'ffn' in name or 'feed_forward' in name or 'feedforward' in name
        for name in child_names_lower
    )
    
    # A transformer block typically has both attention and feedforward
    return has_attention and has_feedforward


def _compile_selective_layers(model: nn.Module, compile_kwargs: dict) -> nn.Module:
    """Compile selective layers using torchtitan approach - target transformer blocks generally."""
    
    compiled_layers = 0
    failed_layers = 0
    backend = compile_kwargs.get('backend', 'inductor')
    
    logger.info("Selective compilation targeting transformer blocks (name-based replacement)")
    
    # Use named_modules instead of just modules - this gives us the module paths
    # which makes replacement much cleaner and more reliable
    for name, m in model.named_modules():
        should_compile = False
        
        # Use general transformer block detection
        if _is_transformer_block(m):
            should_compile = True
            logger.debug(f"Identified transformer block: {name} ({m.__class__.__name__})")
        
        if should_compile:
            try:
                # Apply torch.compile directly to the transformer block (torchtitan style)
                filtered_kwargs = {k: v for k, v in compile_kwargs.items() if k != 'backend'}
                compiled_module = torch.compile(m, backend=backend, **filtered_kwargs)
                # Replace using name-based approach (cleaner and more reliable)
                _replace_module_in_parent(model, name, compiled_module)
                compiled_layers += 1
                logger.debug(f"Compiled transformer block: {name}")
                
            except Exception as e:
                failed_layers += 1
                logger.debug(f"Failed to compile {name}: {type(e).__name__}: {e}")
                continue
    
    # Report results
    if compiled_layers > 0:
        logger.info(
            f"Selective compilation successful! Compiled {compiled_layers} transformer block modules "
            f"({failed_layers} failed)"
        )
        return model
    else:
        logger.warning("No transformer block modules found for compilation")
        logger.warning(
            "This approach targets modules with 'Block'/'Layer' names that contain "
            "attention and feedforward components"
        )
        raise RuntimeError("No transformer block modules could be compiled")


def _replace_module_in_parent(model: nn.Module, module_name: str, new_module: nn.Module):
    """Replace a module in its parent using name-based lookup (torchtitan style).
    
    This is cleaner, faster, and more reliable than instance-based replacement.
    """
    if '.' in module_name:
        parent_name = '.'.join(module_name.split('.')[:-1])
        parent = model.get_submodule(parent_name)
        child_name = module_name.split('.')[-1]
        setattr(parent, child_name, new_module)
        logger.debug(f"Replaced {module_name} in parent {parent_name}")
    else:
        # Root level module
        setattr(model, module_name, new_module)
        logger.debug(f"Replaced root level module {module_name}")


def create_compile_config_from_dict(config_dict: Dict[str, Any]) -> CompileConfig:
    """Create a CompileConfig from a dictionary.

    Args:
        config_dict: Dictionary containing compile configuration.

    Returns:
        CompileConfig instance.
    """
    return CompileConfig(
        enabled=config_dict.get("enabled", False),
        mode=config_dict.get("mode", "default"),
        fullgraph=config_dict.get("fullgraph", False),
        dynamic=config_dict.get("dynamic", False),
        backend=config_dict.get("backend", None),
        options=config_dict.get("options", {}),
    )


def build_compile_config(cfg: Optional[Dict[str, Any]]) -> CompileConfig:
    """Build a compile config from configuration.

    Args:
        cfg: Configuration dictionary for compilation.

    Returns:
        CompileConfig instance.
    """
    if cfg is None:
        return CompileConfig(enabled=False)
    else:
        return create_compile_config_from_dict(cfg) 