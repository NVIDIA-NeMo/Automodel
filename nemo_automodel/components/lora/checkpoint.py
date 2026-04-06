# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoRA checkpoint save and load using the internal _peft component.

Save produces two files:
    adapter_model.safetensors  — lora_A / lora_B weight tensors
    adapter_config.json        — PeftConfig metadata

Load applies LoRA to a base transformer then restores weights from
adapter_model.safetensors.

Identical for Flux, Wan, and Hunyuan — no model_type needed.
"""

import json
import logging
import os
import re

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,  # returns Dict[str, Tensor] — NOT the tuple from get_state_dict
)

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules

logger = logging.getLogger(__name__)


def gather_lora_state_dict(
    transformer: torch.nn.Module,
    is_fsdp: bool,
) -> dict:
    """
    Extract LoRA-only state dict, handling FSDP2 sharded DTensors.

    FSDP2 shards each parameter as a DTensor across GPUs.
    We must gather full tensors first, then filter to lora_ keys.

    All ranks MUST call this — get_model_state_dict is a distributed
    collective when is_fsdp=True.

    Args:
        transformer: FSDP2-wrapped (or plain) transformer module.
        is_fsdp:     Whether transformer is wrapped in FSDP2.

    Returns:
        Dict of LoRA-only weights (lora_A, lora_B for each target module).
    """
    if is_fsdp and dist.is_initialized():
        # Diagnose: check raw local shard norms before gathering
        _b_norms = []
        for n, p in transformer.named_parameters():
            if "lora_B" in n:
                try:
                    _b_norms.append(p.data.to_local().float().norm().item())
                except Exception:
                    _b_norms.append(p.data.float().norm().item())
        if _b_norms:
            logger.info(f"[LoRA] lora_B local shard norms (first 3): {_b_norms[:3]}")

        # get_model_state_dict — returns Dict[str, Tensor] only.
        # Do NOT use get_state_dict() which returns Tuple[model_sd, optim_sd].
        # full_state_dict=True: all-gathers DTensor shards → full tensors on each rank
        # cpu_offload=True:     moves gathered tensors to CPU to avoid GPU OOM
        full_sd = get_model_state_dict(
            transformer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        # Extract LoRA keys from the gathered state dict.
        # Internal _peft keys have the form "<path>.lora_A.weight" — no adapter
        # name component. The re.sub is kept as a no-op safety pass in case any
        # residual adapter-name segments exist from older checkpoints.
        lora_sd = {}
        for k, v in full_sd.items():
            if "lora_A" not in k and "lora_B" not in k:
                continue
            new_key = re.sub(r"(\.lora_[AB])\.[^.]+\.", r"\1.", k)
            lora_sd[new_key] = v

        # Diagnose: check gathered norms
        _gathered_b = [(k, v.float().norm().item()) for k, v in lora_sd.items() if "lora_B" in k]
        if _gathered_b:
            logger.info(f"[LoRA] lora_B gathered norms (first 3): {_gathered_b[:3]}")
    else:
        # Non-FSDP: params are plain tensors, extract lora keys directly
        lora_sd = {k: v.detach().cpu() for k, v in transformer.state_dict().items() if "lora_A" in k or "lora_B" in k}

    return lora_sd


def save_lora_weights(
    transformer: torch.nn.Module,
    peft_config: PeftConfig,
    output_dir: str,
    is_fsdp: bool = True,
    is_main_process: bool = True,
):
    """
    Save LoRA weights in standard format.

    All ranks participate in the gather (distributed collective).
    Only the main process writes files to disk.

    Args:
        transformer:      The transformer module (FSDP2-wrapped or plain).
        peft_config:      PeftConfig returned by inject_lora().
                          Used to write adapter_config.json.
        output_dir:       Directory to write checkpoint files into.
        is_fsdp:          Whether transformer is FSDP2-wrapped.
        is_main_process:  Whether this rank should write to disk.

    Produces:
        {output_dir}/adapter_model.safetensors
        {output_dir}/adapter_config.json
    """
    # All ranks participate (FSDP2 collective — must not be rank-gated)
    lora_sd = gather_lora_state_dict(transformer, is_fsdp)

    # Only main process writes
    if not is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save weight tensors via safetensors
    weights_path = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in lora_sd.items()},
        weights_path,
    )
    logger.info(f"[LoRA] Saved weights  → {weights_path}")

    # 2. Save PeftConfig → adapter_config.json
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(peft_config.to_dict(), f, indent=2)
    logger.info(f"[LoRA] Saved config   → {config_path}")


def load_lora_weights_for_inference(
    transformer: torch.nn.Module,
    lora_path: str,
    adapter_name: str = "default",
    device: str = "cuda",
):
    """
    Load LoRA weights onto a fresh (base) transformer for inference.

    Reads adapter_config.json to reconstruct PeftConfig, applies LoRA
    injection via the internal _peft component, then loads weights from
    adapter_model.safetensors.

    Args:
        transformer:  The raw (base) transformer module.
        lora_path:    Directory containing adapter_model.safetensors
                      and adapter_config.json.
        adapter_name: Unused — kept for API compatibility.
        device:       Device to load weights onto.
    """
    from safetensors.torch import load_file

    # Load config
    config_path = os.path.join(lora_path, "adapter_config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    peft_config = PeftConfig.from_dict(config_dict)

    # Inject LoRA structure into the base model
    apply_lora_to_linear_modules(transformer, peft_config, skip_freeze=True)

    # Load weights
    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    state_dict = load_file(weights_path, device=device)

    incompatible = transformer.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        logger.warning("[LoRA] Unexpected keys (first 5): %s", incompatible.unexpected_keys[:5])
    missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
    if missing_lora:
        logger.warning("[LoRA] Missing lora keys (first 5): %s", missing_lora[:5])

    logger.info(f"[LoRA] Loaded adapter from {lora_path}")
