# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoRA checkpoint save and load using pure PEFT APIs.

Save produces two files (standard PEFT format):
    adapter_model.safetensors  — lora_A / lora_B weight tensors
    adapter_config.json        — LoraConfig metadata

Load uses PeftAdapterMixin.load_adapter() which handles:
    - Reading adapter_config.json
    - Recreating lora_A / lora_B module structure
    - Loading weights from adapter_model.safetensors
    - Key prefix mapping ("base_model.model." handling) internally

This avoids the manual sequence (add_adapter → load_file →
set_peft_model_state_dict) which has a key prefix mismatch:
  saved keys:   "base_model.model.blocks.0.attn1.to_q.lora_A.default.weight"
  expected:     "blocks.0.attn1.to_q.lora_A.default.weight"
load_adapter() resolves this correctly.

Identical for Flux, Wan, and Hunyuan — no model_type needed.
"""

import logging
import os

import torch
import torch.distributed as dist
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,  # returns Dict[str, Tensor] — NOT the tuple from get_state_dict
)

logger = logging.getLogger(__name__)


def gather_lora_state_dict(
    transformer: torch.nn.Module,
    is_fsdp: bool,
) -> dict:
    """
    Extract LoRA-only state dict, handling FSDP2 sharded DTensors.

    FSDP2 shards each parameter as a DTensor across GPUs.
    We must gather full tensors first, then pass the gathered dict
    to get_peft_model_state_dict so PEFT can filter to lora_ keys.

    All ranks MUST call this — get_model_state_dict is a distributed
    collective when is_fsdp=True.

    Args:
        transformer: FSDP2-wrapped (or plain) transformer module.
        is_fsdp:     Whether transformer is wrapped in FSDP2.

    Returns:
        Dict of LoRA-only weights (lora_A, lora_B for each target module).
    """
    if is_fsdp and dist.is_initialized():
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
        # Pass gathered dict — PEFT filters to lora_A/lora_B keys only
        # Matches Flux2 trainer pattern:
        #   get_peft_model_state_dict(transformer, state_dict=state_dict)
        lora_sd = get_peft_model_state_dict(
            transformer,
            state_dict=full_sd,
        )
    else:
        # Non-FSDP: params are plain tensors, extract directly
        lora_sd = get_peft_model_state_dict(transformer)

    return lora_sd


def save_lora_weights(
    transformer: torch.nn.Module,
    peft_config: LoraConfig,
    output_dir: str,
    is_fsdp: bool = True,
    is_main_process: bool = True,
):
    """
    Save LoRA weights in standard PEFT format.

    All ranks participate in the gather (distributed collective).
    Only the main process writes files to disk.

    Args:
        transformer:      The transformer module (FSDP2-wrapped or plain).
        peft_config:      LoraConfig returned by inject_lora().
                          Used to write adapter_config.json.
                          We use this explicit ref rather than accessing
                          transformer.peft_config dict to avoid version
                          fragility across PEFT releases.
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
        {
            k: v.detach().cpu().contiguous()
            for k, v in lora_sd.items()
        },
        weights_path,
    )
    logger.info(f"[LoRA] Saved weights  → {weights_path}")

    # 2. Save LoraConfig → adapter_config.json
    # LoraConfig inherits PeftConfigMixin.save_pretrained()
    # Writes: rank, alpha, target_modules, dropout, init_lora_weights, peft_type
    peft_config.save_pretrained(output_dir)
    logger.info(
        f"[LoRA] Saved config   → {os.path.join(output_dir, 'adapter_config.json')}"
    )


def load_lora_weights_for_inference(
    transformer: torch.nn.Module,
    lora_path: str,
    adapter_name: str = "default",
    device: str = "cuda",
):
    """
    Load LoRA weights onto a fresh (base) transformer for inference.

    Uses PeftAdapterMixin.load_adapter() which handles the complete
    sequence internally:
        1. Reads adapter_config.json → reconstructs LoraConfig
        2. Calls add_adapter() to create lora_A/lora_B module structure
        3. Loads adapter_model.safetensors
        4. Handles "base_model.model." key prefix mapping
        5. Fills lora_A/lora_B weights correctly

    This is preferred over the manual sequence:
        add_adapter() + load_file() + set_peft_model_state_dict()
    because set_peft_model_state_dict() has a key prefix mismatch when
    called on a PeftAdapterMixin model (keys saved with "base_model.model."
    prefix but model parameters don't have that prefix on a fresh load).
    load_adapter() resolves this internally.

    After loading, LoRA scale is applied via attention_kwargs at call time:
        pipe(..., attention_kwargs={"scale": lora_alpha / rank})
    All three transformers have @apply_lora_scale("attention_kwargs") on
    their forward() — verified for Flux, Wan (document 5), Hunyuan (document 31).

    Args:
        transformer:  pipe.transformer — the raw transformer module.
        lora_path:    Directory containing adapter_model.safetensors
                      and adapter_config.json.
        adapter_name: Name to register the adapter under.
        device:       Device to load weights onto.
    """
    # PeftAdapterMixin.load_adapter() handles the full sequence correctly
    transformer.load_adapter(
        lora_path,
        adapter_name=adapter_name,
        torch_device=device,
    )

    # Activate the loaded adapter
    # set_adapter() ensures this adapter's weights are used in forward()
    transformer.set_adapter(adapter_name)

    logger.info(
        f"[LoRA] Loaded adapter '{adapter_name}' from {lora_path}"
    )