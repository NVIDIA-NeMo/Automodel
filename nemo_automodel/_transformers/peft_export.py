# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Serve-ready Hugging Face export for native AutoModel PEFT checkpoints."""

from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.config import CheckpointingConfig


def _is_lora_state_key(key: str) -> bool:
    """Return whether a state-dict key belongs to an AutoModel LoRA adapter."""
    return any(part.startswith("lora_") for part in key.split("."))


@torch.no_grad()
def _merge_and_get_hf_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Fold a model's LoRA updates and convert its state to canonical HF layout.

    Args:
        model: Complete, unsharded AutoModel model whose base and LoRA tensors
            have already been restored. Base weights are updated in place, one
            layer at a time. Tensor layouts remain model-native until conversion
            by ``model.state_dict_adapter``.

    Returns:
        Canonical Hugging Face state mapping. Every tensor retains the dtype of
        the corresponding model weight; model-family adapters may split grouped
        tensors into canonical per-expert tensors.
    """
    state_dict = dict(model.state_dict())
    dtensor_keys = [key for key, tensor in state_dict.items() if isinstance(tensor, DTensor)]
    if dtensor_keys:
        raise ValueError(
            "Merged PEFT export requires a complete, unsharded model; found DTensor weights "
            f"(examples={dtensor_keys[:5]})"
        )

    lora_modules = [module for module in model.modules() if isinstance(module, (GroupedExpertsLoRA, LinearLoRA))]
    if not lora_modules:
        raise ValueError("The model contains no supported AutoModel LoRA modules to merge")

    for module in lora_modules:
        if isinstance(module, GroupedExpertsLoRA):
            gate_and_up, down = module.materialize_effective_weights()
            module.gate_and_up_projs.copy_(gate_and_up)
            del gate_and_up
            module.down_projs.copy_(down)
            del down
        elif isinstance(module, LinearLoRA):
            module.weight.copy_(module.materialize_effective_weight())

    state_dict = {
        key: tensor
        for key, tensor in state_dict.items()
        if not _is_lora_state_key(key) and not key.endswith("_extra_state")
    }

    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is not None:
        state_dict = adapter.to_hf(
            state_dict,
            exclude_key_regex=r".*_extra_state.*",
            quantization=False,
        )

    return state_dict


@torch.no_grad()
def export_merged_peft_checkpoint(
    model: nn.Module,
    *,
    adapter_path: str | Path,
    output_dir: str | Path,
    max_shard_size: int | str = "5GB",
) -> Path:
    """Restore and merge an AutoModel PEFT checkpoint into sharded HF weights.

    The caller constructs ``model`` from the original base checkpoint with the
    same :class:`PeftConfig` used for training. This operation restores the
    adapter with AutoModel's checkpoint loader, which routes PEFT-v5 grouped
    expert tensors through the model-family state-dict adapter before loading.
    It then folds supported ``LinearLoRA`` and ``GroupedExpertsLoRA`` updates
    into their base weights and uses the same adapter to emit canonical Hugging
    Face tensor names and expert layouts.

    Args:
        model: Complete, unsharded AutoModel PEFT model. Its tensors may use any
            model-native layout but must not be DTensors. The operation folds
            the restored adapter into this model's base weights in place.
        adapter_path: AutoModel PEFT model directory containing
            ``adapter_model.safetensors`` and ``automodel_peft_config.json``.
        output_dir: Directory in which to write ``config.json`` and canonical
            Hugging Face safetensor shards.
        max_shard_size: Maximum shard size in bytes or a Hugging Face size string
            such as ``"5GB"``.

    Returns:
        Path to ``output_dir``.

    Raises:
        FileNotFoundError: If the adapter weights do not exist.
        ValueError: If the model is sharded or contains no supported LoRA modules.
    """
    adapter_dir = Path(adapter_path)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"AutoModel PEFT weights not found at {adapter_weights}")

    config = getattr(model, "config", None)
    if config is None or not callable(getattr(config, "save_pretrained", None)):
        raise ValueError("The model config must provide save_pretrained() for Hugging Face export")

    checkpointer = CheckpointingConfig(is_peft=True, save_consolidated=False).build(0, 0, 0)
    try:
        checkpointer.load_model(model, str(adapter_dir))
    finally:
        checkpointer.close()

    model.eval()
    state_dict = _merge_and_get_hf_state_dict(model)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    config.save_pretrained(destination)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and callable(getattr(generation_config, "save_pretrained", None)):
        generation_config.save_pretrained(destination)

    save_torch_state_dict(
        state_dict,
        destination,
        max_shard_size=max_shard_size,
        safe_serialization=True,
        shared_tensors_to_discard=getattr(model, "_tied_weights_keys", None),
    )
    return destination
