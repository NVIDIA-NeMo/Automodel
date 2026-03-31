# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoRA injection via PEFT.

inject_lora() MUST be called before FSDP2 wrapping (_apply_parallelization).

Why order matters:
  FSDP2 preserves original parameter objects (use_orig_params=True default).
  Collecting lora_params refs before FSDP2 gives us handles that remain
  valid after wrapping — FSDP2 only modifies .data to be a sharded DTensor,
  the Python Parameter object itself is unchanged. These refs are passed
  directly to AdamW.
"""

import logging
from typing import Callable

import torch
import torch.nn as nn
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft import PeftAdapterMixin

from .config import MODEL_DEFAULT_TARGET_MODULES

logger = logging.getLogger(__name__)


# ── Pre-inject hook registry ──────────────────────────────────────────────────
# Some models need setup before PEFT can find their target modules.
# Register a hook per model_type. inject_lora() calls it automatically.
# Adding a new model = one @register_pre_inject_hook decorator only.

_PRE_INJECT_HOOKS: dict[str, Callable[[nn.Module], None]] = {}


def register_pre_inject_hook(model_type: str):
    """Decorator to register a model-specific pre-injection setup function."""
    def decorator(fn: Callable[[nn.Module], None]):
        _PRE_INJECT_HOOKS[model_type] = fn
        return fn
    return decorator


@register_pre_inject_hook("wan")
def _wan_pre_inject(transformer: nn.Module) -> None:
    """
    Wan fuses to_q/to_k/to_v into a single to_qkv projection after loading
    for inference efficiency. PEFT walks the module tree looking for modules
    named "to_q", "to_k", "to_v" — these don't exist when fused.

    Must call unfuse_projections() on every attention block before add_adapter()
    so PEFT can find and replace the individual Linear modules.

    Verified from WanAttention.fuse_projections() / unfuse_projections()
    in diffusers WanTransformer3DModel.
    """
    unfused = 0
    for block in transformer.blocks:
        block.attn1.unfuse_projections()
        block.attn2.unfuse_projections()
        unfused += 2
    logger.info(f"[LoRA] Wan: unfused {unfused} attention projection groups")


# Flux:    no pre-inject hook — FluxTransformer2DModel has no fused projections
# Hunyuan: no pre-inject hook — uses standard diffusers Attention (no fusing)


def inject_lora(
    transformer: nn.Module,
    lora_cfg,        # LoRAConfig instance
    model_type: str, # "flux" | "wan" | "hunyuan"
) -> tuple[list[nn.Parameter], LoraConfig]:
    """
    Inject LoRA adapters into a transformer using PEFT.

    MUST be called BEFORE _apply_parallelization() / FSDP2 wrapping.

    After this call:
      - Base weights:  requires_grad=False, native dtype (bf16 as loaded)
      - LoRA A/B:      requires_grad=True,  fp32 (via cast_training_params)

    FSDP2 will then shard both base and LoRA weights. Because base weights
    have requires_grad=False, FSDP2 skips gradient allreduce for them —
    they act as inference-only sharded weights. LoRA weights get full
    allreduce treatment.

    Args:
        transformer: The transformer module (must have PeftAdapterMixin).
        lora_cfg:    LoRAConfig instance.
        model_type:  One of "flux", "wan", "hunyuan".

    Returns:
        lora_params:  List of LoRA Parameter tensors.
                      Collected BEFORE FSDP2 — refs remain valid after
                      wrapping. Pass directly to AdamW optimizer.
        peft_config:  The LoraConfig used for injection.
                      Pass to save_lora_weights() to write adapter_config.json.
                      Avoids fragile transformer.peft_config dict access.
    """
    # ── Guard ────────────────────────────────────────────────────────────────
    if not isinstance(transformer, PeftAdapterMixin):
        raise TypeError(
            f"{type(transformer).__name__} does not inherit PeftAdapterMixin. "
            f"LoRA injection requires PeftAdapterMixin. "
            f"Check your diffusers version for {model_type} support."
        )

    # ── Pre-inject hook ───────────────────────────────────────────────────────
    hook = _PRE_INJECT_HOOKS.get(model_type)
    if hook is not None:
        hook(transformer)

    # ── Resolve target modules ────────────────────────────────────────────────
    target_modules = (
        lora_cfg.target_modules
        or MODEL_DEFAULT_TARGET_MODULES.get(model_type)
    )
    if not target_modules:
        raise ValueError(
            f"No target_modules defined for model_type='{model_type}'. "
            f"Set lora.target_modules explicitly in your yaml config."
        )

    # ── Build LoraConfig ──────────────────────────────────────────────────────
    # Keep reference — returned for use in save_lora_weights().
    # init_lora_weights="gaussian" matches diffusers training scripts.
    peft_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    # ── Inject via PeftAdapterMixin.add_adapter() ─────────────────────────────
    # Internally calls inject_adapter_in_model() which walks the module tree
    # and replaces each target nn.Linear with peft.tuners.lora.Linear:
    #   peft.tuners.lora.Linear:
    #     .base_layer            → original nn.Linear
    #     .lora_A["default"]     → nn.Linear(in, rank, bias=False)
    #     .lora_B["default"]     → nn.Linear(rank, out, bias=False)
    transformer.add_adapter(peft_config)

    # ── Freeze base weights ───────────────────────────────────────────────────
    # Base layer weights must be frozen so FSDP2 skips gradient allreduce.
    # Any param without "lora_" in its name is part of the original model.
    for name, param in transformer.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # ── Cast LoRA params to fp32 ──────────────────────────────────────────────
    # diffusers utility — only touches requires_grad=True params.
    # Ensures LoRA weights stay fp32 for training stability even when
    # FSDP2 MixedPrecisionPolicy has param_dtype=None (our config).
    # Verified from train_dreambooth_flux.py:
    #   cast_training_params([transformer], dtype=torch.float32)
    cast_training_params([transformer], dtype=torch.float32)

    # ── Collect lora_params BEFORE FSDP2 ─────────────────────────────────────
    # FSDP2 uses use_orig_params=True by default — it shards the .data of
    # each Parameter to a DTensor but keeps the Python object intact.
    # These refs remain valid and can be passed to AdamW after wrapping.
    lora_params = [
        p for n, p in transformer.named_parameters()
        if "lora_" in n and p.requires_grad
    ]

    # ── Logging ───────────────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in transformer.parameters())
    lora_param_count = sum(p.numel() for p in lora_params)
    logger.info(
        f"[LoRA] Injected into {model_type}: "
        f"{lora_param_count:,} trainable / {total_params:,} total "
        f"({100.0 * lora_param_count / total_params:.4f}%)"
    )
    logger.info(
        f"[LoRA] rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, "
        f"dropout={lora_cfg.dropout}"
    )
    logger.info(f"[LoRA] target_modules={target_modules}")

    return lora_params, peft_config