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
"""Tests for consolidated HF safetensors checkpointing for VLM."""

import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForImageTextToText
import yaml

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM, calculate_loss


def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """Loads a DCP checkpoint in a state dictionary from a directory."""
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    if "model" in ckpt_dir.name:
        fs_reader = _HuggingFaceStorageReader(ckpt_dir)
    else:
        fs_reader = dcp.FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    # Load tensor data
    tensor_state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    if tensor_state_dict:
        dcp.load(tensor_state_dict, storage_reader=fs_reader)

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
    """ Recursively compare two configs."""
    for k, v in source_config.items():
        if k in restored_config:
            if isinstance(v, dict):
                compare_configs(v, restored_config[k])
            else:
                assert v == restored_config[k], f"Config mismatch for key {k}. Expected {v} but got {restored_config[k]}"


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
    return {k: v.cpu() if isinstance(v, torch.Tensor) else to_cpu(v) for k, v in state_dict.items()}


def get_validation_loss(
    model: nn.Module, val_batch: dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device
) -> torch.Tensor:
    """Gets the validation loss for a model."""
    val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
    model.eval()
    labels = val_batch.pop("labels")
    loss_mask = val_batch.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = (labels.detach() != -100).to(torch.int)

    with torch.no_grad():
        out = model(**val_batch)
        loss = calculate_loss(
                loss_fn,
                logits=out.logits,
                labels=labels,
                mask=loss_mask,
            )
        return loss


def test_consolidated_vlm_checkpoint():
    """
    Tests HF consolidated checkpoint for VLM.
    """
    expected_model_keys = {
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": ([576, 3, 14, 14], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.embeddings.patch_embedding.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.embeddings.position_embedding.weight": ([2048, 1152], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm2.weight": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm2.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": ([2152, 1152], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias": ([2152], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight": ([576, 4304], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.layer_norm1.weight": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.layer_norm1.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.weight": (
            [576, 1152],
            torch.bfloat16,
            "cpu",
        ),
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.layer_norm2.weight": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.layer_norm2.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.mlp.fc1.weight": ([2152, 1152], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.mlp.fc1.bias": ([2152], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.mlp.fc2.weight": ([576, 4304], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.encoder.layers.1.mlp.fc2.bias": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.post_layernorm.weight": ([576], torch.bfloat16, "cpu"),
        "model.vision_tower.vision_model.post_layernorm.bias": ([576], torch.bfloat16, "cpu"),
        "model.multi_modal_projector.mm_input_projection_weight": ([576, 128], torch.bfloat16, "cpu"),
        "model.multi_modal_projector.mm_soft_emb_norm.weight": ([576], torch.bfloat16, "cpu"),
        "model.language_model.embed_tokens.weight": ([131104, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.q_proj.weight": ([64, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.k_proj.weight": ([32, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.v_proj.weight": ([32, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.o_proj.weight": ([64, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.q_norm.weight": ([32], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.self_attn.k_norm.weight": ([32], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.mlp.gate_proj.weight": ([128, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.mlp.up_proj.weight": ([128, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.mlp.down_proj.weight": ([64, 256], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.input_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.post_attention_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.q_proj.weight": ([64, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.k_proj.weight": ([32, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.v_proj.weight": ([32, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.o_proj.weight": ([64, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.q_norm.weight": ([32], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.self_attn.k_norm.weight": ([32], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.mlp.gate_proj.weight": ([128, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.mlp.up_proj.weight": ([128, 128], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.mlp.down_proj.weight": ([64, 256], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.input_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.post_attention_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.pre_feedforward_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.layers.1.post_feedforward_layernorm.weight": ([64], torch.bfloat16, "cpu"),
        "model.language_model.norm.weight": ([64], torch.bfloat16, "cpu"),
    }
    expected_optim_keys = {
        "optim.state.model.multi_modal_projector.mm_input_projection_weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.multi_modal_projector.mm_input_projection_weight.exp_avg": (
            [576, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.multi_modal_projector.mm_input_projection_weight.exp_avg_sq": (
            [576, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.multi_modal_projector.mm_soft_emb_norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.multi_modal_projector.mm_soft_emb_norm.weight.exp_avg": ([576], torch.bfloat16, "cpu"),
        "optim.state.model.multi_modal_projector.mm_soft_emb_norm.weight.exp_avg_sq": ([576], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.weight.exp_avg": ([64, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.weight.exp_avg_sq": (
            [64, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.weight.exp_avg": ([32, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.weight.exp_avg_sq": (
            [32, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.weight.exp_avg": ([32, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.weight.exp_avg_sq": (
            [32, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.weight.exp_avg": ([64, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.weight.exp_avg_sq": (
            [64, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.q_norm.weight.exp_avg": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.q_norm.weight.exp_avg_sq": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.k_norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.k_norm.weight.exp_avg": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.self_attn.k_norm.weight.exp_avg_sq": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.weight.exp_avg": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.weight.exp_avg_sq": (
            [128, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.up_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.up_proj.weight.exp_avg": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.up_proj.weight.exp_avg_sq": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.down_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.down_proj.weight.exp_avg": ([64, 256], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.down_proj.weight.exp_avg_sq": ([64, 256], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.input_layernorm.weight.exp_avg": ([64], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.input_layernorm.weight.exp_avg_sq": ([64], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.0.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.post_attention_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.post_attention_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.pre_feedforward_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.pre_feedforward_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.pre_feedforward_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.post_feedforward_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.post_feedforward_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.post_feedforward_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.weight.exp_avg": ([64, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.weight.exp_avg_sq": (
            [64, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.weight.exp_avg": ([32, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.weight.exp_avg_sq": (
            [32, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.weight.exp_avg": ([32, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.weight.exp_avg_sq": (
            [32, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.weight.exp_avg": ([64, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.weight.exp_avg_sq": (
            [64, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.q_norm.weight.exp_avg": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.q_norm.weight.exp_avg_sq": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.k_norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.k_norm.weight.exp_avg": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.self_attn.k_norm.weight.exp_avg_sq": ([32], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.weight.exp_avg": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.weight.exp_avg_sq": (
            [128, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.up_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.up_proj.weight.exp_avg": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.up_proj.weight.exp_avg_sq": ([128, 128], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.down_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.down_proj.weight.exp_avg": ([64, 256], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.down_proj.weight.exp_avg_sq": ([64, 256], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.input_layernorm.weight.exp_avg": ([64], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.input_layernorm.weight.exp_avg_sq": ([64], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.layers.1.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.post_attention_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.post_attention_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.pre_feedforward_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.pre_feedforward_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.pre_feedforward_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.post_feedforward_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.post_feedforward_layernorm.weight.exp_avg": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.post_feedforward_layernorm.weight.exp_avg_sq": (
            [64],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.norm.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.norm.weight.exp_avg": ([64], torch.bfloat16, "cpu"),
        "optim.state.model.language_model.norm.weight.exp_avg_sq": ([64], torch.bfloat16, "cpu"),
    }

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "gemma_3_vl_4b_cord_v2.yaml")
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = ModelState(
        trainer.model,
    ).state_dict()
    optimizer_state_dict = to_cpu(
        OptimizerState(
            trainer.model,
            trainer.optimizer,
            trainer.step_scheduler,
        ).state_dict()["optim"]["state"]
    )

    # assert the correct paths exist
    output_files = [
        "model",
        "optim",
        "step_scheduler.pt",
        "dataloader.pt",
        "model/shard-00001-model-00001-of-00001.safetensors",
        "model/shard-00002-model-00001-of-00001.safetensors",
        "model/consolidated/model-00001-of-00001.safetensors",
        "model/consolidated/config.json",
        "model/consolidated/model.safetensors.index.json",
        "model/consolidated/chat_template.jinja",
        "model/consolidated/preprocessor_config.json",
        "model/consolidated/processor_config.json",
        "model/consolidated/tokenizer_config.json",
        "model/consolidated/tokenizer.json",
        "model/consolidated/special_tokens_map.json",
        "model/consolidated/generation_config.json",
        "optim/__0_0.distcp",
        "optim/__1_0.distcp",
        "optim/.metadata",
        "step_scheduler.pt",
        "config.yaml",
    ]

    for file in output_files:
        path = Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_2_step_10" / file
        assert path.exists(), f"Expected {path} to exist"
        if "." in file:
            assert path.is_file(), f"Expected {path} to be a file"
        else:
            assert path.is_dir(), f"Expected {path} to be a directory"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"

    # Load checkpoint data
    restored_optim_dict, saved_lr_scheduler_state = load_dcp(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_2_step_10" / "optim",
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

        restored_lr_state = trainer.lr_scheduler.state_dict()

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
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_2_step_10" / "model",
    )
    restored_model_dict_consolidated = load_safetensors(
        Path(trainer.checkpoint_config.checkpoint_dir)
        / "epoch_2_step_10"
        / "model"
        / "consolidated"
        / "model-00001-of-00001.safetensors",
    )

    # check if newly restored model and current model give the same CE loss
    val_batch = next(iter(trainer.val_dataloader))
    restored_model = FinetuneRecipeForVLM(cfg)
    restored_model.setup()
    restored_model = restored_model.model
    source_model_loss = get_validation_loss(trainer.model, val_batch, trainer.loss_fn, trainer.dist_env.device)
    restored_model_loss = get_validation_loss(restored_model, val_batch, trainer.loss_fn, trainer.dist_env.device)
    assert torch.allclose(source_model_loss, restored_model_loss), "Model loss mismatch"

    # compare the recipe configs
    with open(Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_2_step_10" / "config.yaml", "r") as f:
        restored_config = yaml.safe_load(f)
    compare_configs(trainer.cfg.raw_config, restored_config)

    # load consolidated model using HF API and verify it's the same as the trained model
    consolidated_model = (
        AutoModelForImageTextToText.from_pretrained(
            Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_2_step_10" / "model" / "consolidated"
        )
        .to(trainer.model.dtype)
        .to(trainer.dist_env.device)
    )
    for source_key, source_param in model_state_dict.items():
        for consolidated_key, consolidated_param in consolidated_model.named_parameters():
            if source_key in consolidated_key:
                assert torch.allclose(source_param.full_tensor(), consolidated_param), (
                    "Parameter values are different when they should be the same"
                )

    # similarly, the optimizer states are saved in a dictionary formatted as:
    # {
    #     "optim": OptimizerState(...),
    #     "step_scheduler": StepSchedulerState(...)
    # }
    # and in addition, the optimizer state is saved in a dictionary formatted as:
    # {
    #     "optim": {
    #         "state": {
    #             "model.layers.0.self_attn.q_proj.weight":
    #                 "step": ...,
    #                 "exp_avg": ...
    #                 "exp_avg_sq": ...
    #         }
    #     }
    # }
    # because of this, DCP will flatten the optimizer state dictionary to:
    # {
    #     "optim.state.model.layers.0.self_attn.q_proj.weight.step": ...
    # }
    # so we flatten the in-memory optimizer state dictionary to match the on-disk view
    flattened_optim_dict = _flatten(optimizer_state_dict, parent_key="optim.state")

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory model state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_model_keys.keys()) == set(restored_model_dict.keys()), (
        "Mismatch between in-memory and on-disk model keys."
    )
    assert set(expected_model_keys.keys()) == set(restored_model_dict_consolidated.keys()), (
        "Mismatch between in-memory and on-disk consolidated model keys."
    )

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory optimizer state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_optim_keys.keys()) == set(restored_optim_dict.keys()), (
        "Mismatch between in-memory and on-disk optimizer keys."
    )

    # Note: all ranks should test their own shard of the model state and optimizer state

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk model state
    for k in expected_model_keys.keys():
        v = model_state_dict[k]
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local().cpu()
        assert k in restored_model_dict, f"Key {k} not found in restored model state"
        assert isinstance(
            restored_model_dict[k],
            torch.Tensor,
        ), f"Value for key {k} is not a tensor"

        # Get expected shape, dtype, device from expected_model_keys
        expected_shape, expected_dtype, expected_device = expected_model_keys[k]

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

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk consolidated model state
    for k in expected_model_keys.keys():
        v = model_state_dict[k]
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.full_tensor().cpu()
        assert k in restored_model_dict_consolidated, f"Key {k} not found in restored model state"
        assert isinstance(
            restored_model_dict_consolidated[k],
            torch.Tensor,
        ), f"Value for key {k} is not a tensor"

        # Get expected shape, dtype, device from expected_model_keys
        expected_shape, expected_dtype, expected_device = expected_model_keys[k]
        expected_shape[0] *= 2  # since the hardcoded shapes are for sharded Tensors

        full_shard = restored_model_dict_consolidated[k]

        assert list(full_shard.shape) == expected_shape, (
            f"Shape mismatch for key {k}. Expected shape {expected_shape} but got {full_shard.shape}"
        )
        assert full_shard.dtype == expected_dtype, (
            f"Dtype mismatch for key {k}. Expected dtype {expected_dtype} but got {full_shard.dtype}"
        )
        assert str(full_shard.device) == expected_device, (
            f"Device mismatch for key {k}. Expected device {expected_device} but got {full_shard.device}"
        )
        assert torch.allclose(v, full_shard), f"Value mismatch for key {k}. Tensors are not numerically close"

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk optimizer state
    for k, v in flattened_optim_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
        assert k in restored_optim_dict, f"Key {k} not found in restored optimizer state"
        assert isinstance(
            restored_optim_dict[k],
            torch.Tensor,
        ), f"Value for key {k} is not a tensor"

        # Get expected shape, dtype, device from expected_optim_keys
        expected_shape, expected_dtype, expected_device = expected_optim_keys[k]

        if restored_optim_dict[k].size():
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
        assert torch.allclose(v, curr_shard), f"Value mismatch for key {k}. Tensors are not numerically close"
    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path(trainer.checkpoint_config.checkpoint_dir).exists():
            shutil.rmtree(Path(trainer.checkpoint_config.checkpoint_dir))
    torch.distributed.barrier()


def _flatten(d: dict, parent_key: str | None = None):
    """Recursively flatten *d* using dot-separated keys (à la DCP).
    The first component in *parent_key* lets us prepend the outer-dict key
    ("optim" in our case) so that the resulting keys match the exact strings
    stored on disk by torch.distributed.checkpoint.
    """

    flat: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat
