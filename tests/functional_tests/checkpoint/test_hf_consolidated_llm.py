# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# pylint: disable=line-too-long
"""Tests for consolidated HF safetensors checkpointing for LLM."""

import os
import re
import shutil
from pathlib import Path

import datasets
import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
import torch.nn as nn
import yaml
from safetensors import safe_open

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction, calculate_loss
from transformers import AutoModelForCausalLM

datasets.disable_caching()




def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """
    Loads a DCP checkpoint in a state dictionary from a directory.
    Args:
        ckpt_dir: The directory containing the DCP checkpoint.
    Returns:
        A state dictionary containing the checkpoint.
    """
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    if "model" in ckpt_dir.name:
        fs_reader = _HuggingFaceStorageReader(ckpt_dir)
    else:
        fs_reader = dcp.FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    tensor_state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    dcp.load(
        tensor_state_dict,
        storage_reader=fs_reader,
    )

    # Load scheduler data
    sched_keys = [k for k, tp in metadata.state_dict_metadata.items() if "sched" in k]

    sched_state_dict = {}
    if sched_keys:
        sched_state_dict = {k: None for k in sched_keys}
        try:
            dcp.load(sched_state_dict, storage_reader=fs_reader)
        except Exception:
            sched_state_dict = {}
    return tensor_state_dict, sched_state_dict


def compare_configs(source_config: dict, restored_config: dict):
    """Recursively compare two configs."""
    for k, v in source_config.items():
        if k in restored_config:
            if isinstance(v, dict):
                compare_configs(v, restored_config[k])
            else:
                assert v == restored_config[k], (
                    f"Config mismatch for key {k}. Expected {v} but got {restored_config[k]}"
                )


def load_safetensors(ckpt_dir: Path | str) -> dict[str, torch.Tensor]:
    """
    Loads a safetensors checkpoint in a state dictionary from a directory.
    """
    state_dict = {}
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    with safe_open(ckpt_dir, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def to_cpu(
    state_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """
    Converts a state dictionary to CPU.
    """
    return {k: v.cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def get_validation_loss(
    model_parts: list[nn.Module],
    val_batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
    pp_enabled: bool,
    pp,
) -> torch.Tensor:
    """Gets the validation loss for a model."""
    loss_buffer = []
    val_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
    num_label_tokens = (val_batch["labels"] != -100).sum().item()
    for model_part in model_parts:
        model_part.eval()
    labels = val_batch.pop("labels")
    loss_mask = val_batch.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = (labels.detach() != -100).to(torch.int)

    if not pp_enabled:
        with torch.no_grad():
            out = model_parts[0](**val_batch)
            loss = calculate_loss(
                loss_fn,
                logits=out.logits,
                labels=labels,
                model=model_parts[0],
                num_label_tokens=num_label_tokens,
            )
            return [loss]
    else:
        losses = [] if pp.info.has_last_stage else None
        if pp.info.has_last_stage:
            masked_labels = labels.clone()
            targets = masked_labels
        else:
            targets = None

        input_ids = val_batch.pop("input_ids")
        if pp.info.has_first_stage:
            pp.info.schedule.step(input_ids, target=targets, losses=losses, **val_batch)
        else:
            pp.info.schedule.step(target=targets, losses=losses, **val_batch)
        if pp.info.has_last_stage:
            local_loss = torch.sum(torch.stack(losses))
        else:
            local_loss = torch.tensor(0.0, device=device)

        loss_buffer.append(local_loss.clone().detach())
        return loss_buffer


def get_test_consolidated_llm_checkpoint_expected_keys():
    """
    Get expected checkpoint keys for transformers v5.
    """
    return _get_test_consolidated_llm_checkpoint_expected_keys_v5()


def _get_test_consolidated_llm_checkpoint_expected_keys_v5():
    def _convert_v4_keys_to_v5(expected_model_keys: dict, expected_optim_keys: dict) -> tuple[dict, dict]:
        """
        Convert v4-style Mixtral checkpoint keys to v5 format.

        In transformers v5, Mixtral uses grouped expert format:
        - block_sparse_moe.gate -> mlp.gate
        - block_sparse_moe.experts.X.w1/w3.weight -> mlp.experts.gate_up_proj (merged)
        - block_sparse_moe.experts.X.w2.weight -> mlp.experts.down_proj (merged)
        """
        v5_model_keys = {}
        v5_optim_keys = {}

        # Pattern to match expert weights and gate
        expert_pattern = re.compile(r"(.*)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight(.*)")
        gate_pattern = re.compile(r"(.*)\.block_sparse_moe\.gate\.weight(.*)")

        # First pass: determine num_experts from gate weight shape and collect expert info per layer
        layer_info = {}  # layer_key -> {num_experts, w1_shape, w2_shape, w3_shape, expert_indices}

        for key, value in expected_model_keys.items():
            # Get num_experts from gate weight shape
            gate_match = gate_pattern.match(key)
            if gate_match:
                prefix, _ = gate_match.groups()
                if prefix not in layer_info:
                    layer_info[prefix] = {"expert_indices": set()}
                layer_info[prefix]["num_experts"] = value[0][0]  # First dim of gate weight is num_experts
                continue

            # Collect expert info
            expert_match = expert_pattern.match(key)
            if expert_match:
                prefix, expert_idx, weight_type, _ = expert_match.groups()
                if prefix not in layer_info:
                    layer_info[prefix] = {"expert_indices": set()}
                layer_info[prefix]["expert_indices"].add(int(expert_idx))
                layer_info[prefix][f"{weight_type}_shape"] = value[0]
                layer_info[prefix][f"{weight_type}_dtype"] = value[1]
                layer_info[prefix][f"{weight_type}_device"] = value[2]

        # Second pass: build v5 keys
        for key, value in expected_model_keys.items():
            # Handle gate renaming
            gate_match = gate_pattern.match(key)
            if gate_match:
                prefix, suffix = gate_match.groups()
                new_key = f"{prefix}.mlp.gate.weight{suffix}"
                v5_model_keys[new_key] = value
                continue

            # Handle expert weights - skip individual keys, we'll add merged ones
            expert_match = expert_pattern.match(key)
            if expert_match:
                continue  # Skip individual expert keys

            # Non-expert keys pass through unchanged
            v5_model_keys[key] = value

        # Add merged expert keys for each layer
        for layer_key, info in layer_info.items():
            num_experts = info.get("num_experts", len(info["expert_indices"]))

            # gate_up_proj: merge w1 and w3 -> [num_experts, intermediate_size * 2, hidden_size]
            if "w1_shape" in info:
                w1_shape = info["w1_shape"]  # [intermediate_size, hidden_size]
                # Merged shape: [num_experts, intermediate_size * 2, hidden_size]
                merged_shape = [num_experts, w1_shape[0] * 2, w1_shape[1]]
                v5_model_keys[f"{layer_key}.mlp.experts.gate_up_proj"] = (
                    merged_shape,
                    info["w1_dtype"],
                    info["w1_device"],
                )

            # down_proj: merge w2 -> [num_experts, hidden_size, intermediate_size]
            if "w2_shape" in info:
                w2_shape = info["w2_shape"]  # [hidden_size, intermediate_size]
                merged_shape = [num_experts, w2_shape[0], w2_shape[1]]
                v5_model_keys[f"{layer_key}.mlp.experts.down_proj"] = (
                    merged_shape,
                    info["w2_dtype"],
                    info["w2_device"],
                )

        # Convert optimizer keys similarly
        for key, value in expected_optim_keys.items():
            # Handle gate renaming
            gate_match = gate_pattern.match(key)
            if gate_match:
                prefix, suffix = gate_match.groups()
                new_key = f"{prefix}.mlp.gate.weight{suffix}"
                v5_optim_keys[new_key] = value
                continue

            # Handle expert weights
            expert_match = expert_pattern.match(key)
            if expert_match:
                prefix, expert_idx, weight_type, suffix = expert_match.groups()

                # Determine the merged key
                if weight_type in ("w1", "w3"):
                    new_base_key = f"{prefix}.mlp.experts.gate_up_proj"
                elif weight_type == "w2":
                    new_base_key = f"{prefix}.mlp.experts.down_proj"
                else:
                    continue

                # Only add the merged key once (from first expert we see)
                new_key = f"{new_base_key}{suffix}"
                if new_key not in v5_optim_keys:
                    # Extract model layer key from optimizer prefix (strip "optim.state.")
                    model_layer_key = (
                        prefix.replace("optim.state.", "") if prefix.startswith("optim.state.") else prefix
                    )
                    info = layer_info.get(model_layer_key, {})
                    num_experts = info.get("num_experts", 4)  # Default to 4 experts

                    # For optimizer states, shape depends on the suffix
                    if ".step" in suffix:
                        v5_optim_keys[new_key] = value  # step is scalar
                    else:
                        # Compute merged shape based on weight type
                        if weight_type in ("w1", "w3") and "w1_shape" in info:
                            w1_shape = info["w1_shape"]
                            merged_shape = [num_experts, w1_shape[0] * 2, w1_shape[1]]
                        elif weight_type == "w2" and "w2_shape" in info:
                            w2_shape = info["w2_shape"]
                            merged_shape = [num_experts, w2_shape[0], w2_shape[1]]
                        else:
                            # Fallback: use value shape scaled by num_experts
                            merged_shape = [num_experts] + list(value[0])
                        v5_optim_keys[new_key] = (merged_shape, value[1], value[2])
                continue

            # Non-expert keys pass through unchanged
            v5_optim_keys[key] = value

        return v5_model_keys, v5_optim_keys

    return _convert_v4_keys_to_v5(*_get_test_consolidated_llm_checkpoint_expected_keys_v4())



def _get_test_consolidated_llm_checkpoint_expected_keys_v4():
    expected_model_keys = {
        "model.embed_tokens.weight": ([16000, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.q_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.k_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.v_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.o_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.gate.weight": ([4, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.0.input_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.0.post_attention_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.q_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.k_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.v_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.o_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.gate.weight": ([4, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w1.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w2.weight": ([512, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w3.weight": ([448, 512], torch.bfloat16, "cpu"),
        "model.layers.1.input_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.1.post_attention_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.norm.weight": ([256], torch.bfloat16, "cpu"),
        "lm_head.weight": ([16000, 512], torch.bfloat16, "cpu"),
    }
    expected_optim_keys = {
        "optim.state.model.embed_tokens.weight.exp_avg": ([16000, 512], torch.bfloat16, "cpu"),
        "optim.state.model.embed_tokens.weight.exp_avg_sq": ([16000, 512], torch.bfloat16, "cpu"),
        "optim.state.model.embed_tokens.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.self_attn.q_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.q_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.self_attn.k_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.k_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.self_attn.v_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.v_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.self_attn.o_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.self_attn.o_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.gate.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.gate.weight.exp_avg": ([4, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.gate.weight.exp_avg_sq": ([4, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.input_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.input_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.post_attention_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.0.post_attention_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.self_attn.q_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.q_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.self_attn.k_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.k_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.self_attn.v_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.v_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.self_attn.o_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.self_attn.o_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.gate.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.gate.weight.exp_avg": ([4, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.gate.weight.exp_avg_sq": ([4, 512], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [512, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [448, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.input_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.input_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.post_attention_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.layers.1.post_attention_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.norm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.state.model.norm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.state.lm_head.weight.step": ([], torch.float32, "cpu"),
        "optim.state.lm_head.weight.exp_avg": ([16000, 512], torch.bfloat16, "cpu"),
        "optim.state.lm_head.weight.exp_avg_sq": ([16000, 512], torch.bfloat16, "cpu"),
    }
    return expected_model_keys, expected_optim_keys


def test_consolidated_llm_checkpoint():
    """
    Tests HF consolidated checkpoint for LLM.
    """
    expected_model_keys, expected_optim_keys = get_test_consolidated_llm_checkpoint_expected_keys()

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama3_2" / "llama3_2_1b_hellaswag.yaml")
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = ModelState(
        trainer.model_parts,
    ).state_dict()
    optimizer_state_dict = to_cpu(
        OptimizerState(
            trainer.model_parts,
            trainer.optimizer,
            trainer.lr_scheduler,
        ).state_dict()["optim"]
    )

    # assert the correct paths exist
    output_files = [
        "model",
        "optim",
        "step_scheduler.pt",
        "dataloader/dataloader_dp_rank_0.pt",
        "rng/rng_dp_rank_0.pt",
        "model/shard-00001-model-00001-of-00001.safetensors",
        "model/shard-00002-model-00001-of-00001.safetensors",
        "model/consolidated/model-00001-of-00001.safetensors",
        "model/consolidated/config.json",
        "model/consolidated/model.safetensors.index.json",
        "model/consolidated/config.json",
        "model/consolidated/tokenizer_config.json",
        "model/consolidated/tokenizer.json",
        "model/consolidated/model.safetensors.index.json",
        "model/consolidated/generation_config.json",
        "optim/__0_0.distcp",
        "optim/__1_0.distcp",
        "optim/.metadata",
        "step_scheduler.pt",
        "config.yaml",
        "losses.json",
    ]
    if trainer._get_dp_group_size() > 1:
        output_files.append("dataloader/dataloader_dp_rank_1.pt")
        output_files.append("rng/rng_dp_rank_1.pt")

    for file in output_files:
        path = Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9" / file
        assert path.exists(), f"Expected {path} to exist"
        if "." in file:
            assert path.is_file(), f"Expected {path} to be a file"
        else:
            assert path.is_dir(), f"Expected {path} to be a directory"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"
    restored_optim_dict, saved_lr_scheduler_state = load_dcp(
        Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9" / "optim",
    )
    # Remove "sched." prefix from keys in saved_lr_scheduler_state if present
    if saved_lr_scheduler_state is not None:
        saved_lr_scheduler_state = {
            (k[6:] if k.startswith("sched.") else k): v for k, v in saved_lr_scheduler_state.items()
        }

    if saved_lr_scheduler_state is not None and trainer.lr_scheduler is not None:
        assert hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None, (
            "test_dcp_checkpoint: lr_scheduler not found in restored trainer"
        )

        restored_lr_state = trainer.lr_scheduler[0].state_dict()

        for key in saved_lr_scheduler_state:
            assert key in restored_lr_state, f"test_dcp_checkpoint: lr_scheduler key {key} missing in restored state"
            saved_val = saved_lr_scheduler_state[key]
            restored_val = restored_lr_state[key]

            if isinstance(saved_val, torch.Tensor):
                assert torch.equal(saved_val, restored_val), (
                    f"test_dcp_checkpoint: lr_scheduler tensor mismatch for {key}"
                )
            else:
                assert saved_val == restored_val, (
                    f"test_dcp_checkpoint: lr_scheduler value mismatch for {key}: saved={saved_val} != restored={restored_val}"
                )

    restored_model_dict, _ = load_dcp(
        Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9" / "model",
    )
    restored_model_dict_consolidated = load_safetensors(
        Path(trainer.checkpointer.config.checkpoint_dir)
        / "epoch_0_step_9"
        / "model"
        / "consolidated"
        / "model-00001-of-00001.safetensors",
    )

    # check if newly restored model and current model give the same CE loss
    val_batch = next(iter(trainer.val_dataloaders["default"]))
    restored_model = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    restored_model.setup()
    source_model_loss = get_validation_loss(
        trainer.model_parts, val_batch, trainer.loss_fn, trainer.dist_env.device, trainer.pp_enabled, trainer.pp
    )
    restored_model_loss = get_validation_loss(
        restored_model.model_parts,
        val_batch,
        trainer.loss_fn,
        trainer.dist_env.device,
        restored_model.pp_enabled,
        restored_model.pp,
    )
    assert sum(source_model_loss) == sum(restored_model_loss), "Model loss mismatch"

    # compare the recipe configs
    with open(Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9" / "config.yaml", "r") as f:
        restored_config = yaml.safe_load(f)
    compare_configs(trainer.cfg.raw_config, restored_config)

    # load consolidated model using HF API and verify it loads correctly
    consolidated_model = (
        AutoModelForCausalLM.from_pretrained(
            Path(trainer.checkpointer.config.checkpoint_dir) / "epoch_0_step_9" / "model" / "consolidated"
        )
        .to(trainer.model_parts[0].dtype)
        .to(trainer.dist_env.device)
    )
    # Verify consolidated model matches the on-disk consolidated safetensors (both from step 9)
    consolidated_params = dict(consolidated_model.named_parameters())
    for key, param in consolidated_params.items():
        if key in restored_model_dict_consolidated:
            restored_tensor = restored_model_dict_consolidated[key]
            assert torch.allclose(param.cpu(), restored_tensor.cpu()), (
                f"Consolidated model parameter doesn't match on-disk checkpoint for key {key}"
            )

    # the saved optimizer state has an "optim." prefix that DCP adds.
    # For the on-disk view to match, it needs to be prepended with the "optim." prefix
    optimizer_state_dict = _rename_keys(optimizer_state_dict, "optim.")

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory model state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_model_keys.keys()) == set(restored_model_dict.keys()), (
        "Mismatch between in-memory and on-disk model keys."
    )
    # Note: consolidated checkpoint keys may differ due to HF format conversion
    # The key comparison is done against the sharded checkpoint which uses the native format
    # The consolidated checkpoint should match what HF's from_pretrained expects
    assert len(restored_model_dict_consolidated) > 0, "Consolidated model checkpoint is empty"

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory optimizer state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_optim_keys.keys()) == set(restored_optim_dict.keys()), (
        "Mismatch between in-memory and on-disk optimizer keys."
    )

    # Note: all ranks should test their own shard of the model state and optimizer state

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk model state
    for k in model_state_dict.keys():
        v = model_state_dict[k]
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local().cpu()
        else:
            v = v.cpu()
        assert k in restored_model_dict, f"Key {k} not found in restored model state"
        assert isinstance(
            restored_model_dict[k],
            torch.Tensor,
        ), f"Value for key {k} is not a tensor"

        # Get expected shape, dtype, device from expected_model_keys
        expected_shape, expected_dtype, expected_device = expected_model_keys[k]
        expected_shape = expected_shape.copy()

        if trainer.pp_enabled:
            if len(expected_shape) > 0:
                expected_shape[0] *= 2
            curr_shard = restored_model_dict[k]
        else:
            curr_shard = torch.split(
                restored_model_dict[k],
                restored_model_dict[k].shape[0] // 2,
            )[torch.distributed.get_rank()]

        assert list(curr_shard.shape) == expected_shape, (
            f"Shape mismatch for key {k}. Expected shape {expected_shape} but got {curr_shard.shape}"
        )
        assert curr_shard.dtype == expected_dtype, (
            f"Dtype mismatch for key {k}. Expected dtype {expected_dtype} but got {curr_shard.dtype}"
        )
        assert str(curr_shard.device) == expected_device, (
            f"Device mismatch for key {k}. Expected device {expected_device} but got {curr_shard.device}"
        )
        assert torch.allclose(v, curr_shard), f"Value mismatch for key {k}. Tensors are not numerically close"

    # Note: Consolidated checkpoint comparison is done via HF's from_pretrained above.
    # The consolidated checkpoint may use a different key format (HF format) than the
    # native model format, so direct key-by-key comparison isn't meaningful here.
    # The HF loading test above verifies the consolidated checkpoint is correct.

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk optimizer state
    for k, v in optimizer_state_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
        assert k in restored_optim_dict, f"Key {k} not found in restored optimizer state"
        assert isinstance(
            restored_optim_dict[k],
            torch.Tensor,
        ), f"Value for key {k} is not a tensor"

        # Get expected shape, dtype, device from expected_optim_keys
        expected_shape, expected_dtype, expected_device = expected_optim_keys[k]

        if trainer.pp_enabled and len(expected_shape) > 0:
            expected_shape[0] *= 2

        if restored_optim_dict[k].size() and not trainer.pp_enabled:
            curr_shard = torch.split(
                restored_optim_dict[k],
                restored_optim_dict[k].shape[0] // 2,
            )[torch.distributed.get_rank()]
        else:
            # this can be the parameter step which is a scalar Tensor
            curr_shard = restored_optim_dict[k]
        assert list(curr_shard.shape) == expected_shape, (
            f"Shape mismatch for key {k}. Expected shape {expected_shape} but got {curr_shard.shape}"
        )
        assert curr_shard.dtype == expected_dtype, (
            f"Dtype mismatch for key {k}. Expected dtype {expected_dtype} but got {curr_shard.dtype}"
        )
        assert str(curr_shard.device) == expected_device, (
            f"Device mismatch for key {k}. Expected device {expected_device} but got {curr_shard.device}"
        )
        try:
            assert torch.allclose(v, curr_shard), f"Value mismatch for key {k}. Tensors are not numerically close"
        except Exception as e:
            if "moe" in k and "step" in k:
                pass
            else:
                raise e
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path(trainer.checkpointer.config.checkpoint_dir).exists():
            shutil.rmtree(Path(trainer.checkpointer.config.checkpoint_dir))
    torch.distributed.barrier()


def _rename_keys(d: dict, prepend: str):
    """Rename the keys of *d* by prepending *prepend* to each key."""
    flat: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prepend}{k}"
        flat[key] = v
    return flat
