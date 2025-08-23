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
"""Tests for PEFT checkpointing."""

import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
import torch.nn as nn
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForImageTextToText
import yaml

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM, calculate_loss


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


def test_hf_peft_checkpoint():
    """
    Tests HF PEFT checkpoint
    """
    expected_model_keys = {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_B.weight": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.v_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.v_proj.lora_B.weight": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_B.weight": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_A.weight": ([8, 128], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.0.mlp.gate_proj.lora_B.weight": ([256, 8], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight": ([8, 128], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.0.mlp.up_proj.lora_B.weight": ([256, 8], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight": ([8, 256], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight": ([128, 8], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_B.weight": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.k_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.k_proj.lora_B.weight": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.v_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.v_proj.lora_B.weight": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_A.weight": (
            [8, 128],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.self_attn.o_proj.lora_B.weight": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_A.weight": ([8, 128], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_B.weight": ([256, 8], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.mlp.up_proj.lora_A.weight": ([8, 128], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.mlp.up_proj.lora_B.weight": ([256, 8], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.mlp.down_proj.lora_A.weight": ([8, 256], torch.bfloat16, "cpu"),
        "base_model.model.model.language_model.layers.1.mlp.down_proj.lora_B.weight": ([128, 8], torch.bfloat16, "cpu"),
    }
    expected_optim_keys = {
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.q_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_B.weight.exp_avg": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.k_proj.lora_B.weight.exp_avg_sq": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_B.weight.exp_avg": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.v_proj.lora_B.weight.exp_avg_sq": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.self_attn.o_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_B.weight.exp_avg": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.gate_proj.lora_B.weight.exp_avg_sq": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_B.weight.exp_avg": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.up_proj.lora_B.weight.exp_avg_sq": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_A.weight.exp_avg": (
            [4, 256],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_A.weight.exp_avg_sq": (
            [4, 256],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.0.mlp.down_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.q_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_B.weight.exp_avg": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.k_proj.lora_B.weight.exp_avg_sq": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_B.weight.exp_avg": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.v_proj.lora_B.weight.exp_avg_sq": (
            [32, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_A.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_B.weight.step": (
            [],
            torch.float32,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.self_attn.o_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_B.weight.exp_avg": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.gate_proj.lora_B.weight.exp_avg_sq": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_A.weight.exp_avg": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_A.weight.exp_avg_sq": (
            [4, 128],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_B.weight.exp_avg": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.up_proj.lora_B.weight.exp_avg_sq": (
            [128, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_A.weight.exp_avg": (
            [4, 256],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_A.weight.exp_avg_sq": (
            [4, 256],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_B.weight.exp_avg": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.language_model.layers.1.mlp.down_proj.lora_B.weight.exp_avg_sq": (
            [64, 8],
            torch.bfloat16,
            "cpu",
        ),
    }
    expected_config = {
        "base_model_name_or_path": "/home/TestData/huiyingl/hf_gemma3_2l/",
        "bias": "none",
        "lora_alpha": 32,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": [
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.1.mlp.down_proj",
            "model.language_model.layers.1.mlp.gate_proj",
            "model.language_model.layers.1.mlp.up_proj",
            "model.language_model.layers.1.self_attn.k_proj",
            "model.language_model.layers.1.self_attn.o_proj",
            "model.language_model.layers.1.self_attn.q_proj",
            "model.language_model.layers.1.self_attn.v_proj",
        ],
        "task_type": "CAUSAL_LM",
    }
    expected_automodel_peft_config = {
        "dropout": 0.0,
        "dropout_position": "post",
        "exclude_modules": ["*vision_tower*", "*vision*", "*visual*", "*image_encoder*", "*lm_head*"],
        "lora_A_init": "xavier",
        "lora_dtype": None,
        "match_all_linear": False,
        "target_modules": [],
        "use_triton": True,
    }

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "gemma_3_vl_4b_cord_v2_peft.yaml")
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = ModelState(
        trainer.model,
        trainer.checkpoint_config.is_peft,
    ).state_dict()
    optimizer_state_dict = to_cpu(
        OptimizerState(
            trainer.model,
            trainer.optimizer,
            trainer.step_scheduler,
        ).state_dict()["optim"]["state"]
    )

    model_keys_fixture = {}
    for k, v in model_state_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
        # PEFT model state is consolidated - use FULL tensor shape (no splitting)
        curr_shard = v
        model_keys_fixture[k] = (list(curr_shard.shape), curr_shard.dtype, str(curr_shard.device))
    flattened_optim_dict = _flatten(optimizer_state_dict, parent_key="optim.state")
    optim_keys_fixture = {}
    for k, v in flattened_optim_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
        if v.size():
            curr_shard = v  # torch.split(v, v.shape[0] // 2)[torch.distributed.get_rank()]
        else:
            curr_shard = v
        optim_keys_fixture[k] = (list(curr_shard.shape), curr_shard.dtype, str(curr_shard.device))
    # assert the correct paths exist
    output_files = [
        "model",
        "optim",
        "step_scheduler.pt",
        "dataloader.pt",
        "model/adapter_model.safetensors",
        "model/adapter_config.json",
        "model/automodel_peft_config.json",
        "model/chat_template.jinja",
        "model/preprocessor_config.json",
        "model/processor_config.json",
        "model/tokenizer_config.json",
        "model/tokenizer.json",
        "model/special_tokens_map.json",
        "optim/__0_0.distcp",
        "optim/__1_0.distcp",
        "optim/.metadata",
        "step_scheduler.pt",
        "config.yaml",
    ]

    for file in output_files:
        path = Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / file
        assert path.exists(), f"Expected {path} to exist"
        if "." in file:
            assert path.is_file(), f"Expected {path} to be a file"
        else:
            assert path.is_dir(), f"Expected {path} to be a directory"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"

    # Load checkpoint data
    restored_optim_dict, saved_lr_scheduler_state = load_dcp(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "optim",
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

    restored_model_dict_consolidated = load_safetensors(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model" / "adapter_model.safetensors",
    )
    restored_config = json.load(
        open(Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model" / "adapter_config.json"),
    )
    restored_automodel_peft_config = json.load(
        open(
            Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model" / "automodel_peft_config.json"
        ),
    )
    _compare_dicts(expected_config, restored_config)
    _compare_dicts(expected_automodel_peft_config, restored_automodel_peft_config)

    # check if new model and current model give the same CE loss
    val_batch = next(iter(trainer.val_dataloader))
    restored_model = FinetuneRecipeForVLM(cfg)
    restored_model.setup()
    restored_model = restored_model.model
    source_model_loss = get_validation_loss(trainer.model, val_batch, trainer.loss_fn, trainer.dist_env.device)
    restored_model_loss = get_validation_loss(restored_model, val_batch, trainer.loss_fn, trainer.dist_env.device)
    assert torch.allclose(source_model_loss, restored_model_loss), "Model loss mismatch"

    # compare the recipe configs
    with open(Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "config.yaml", "r") as f:
        restored_config = yaml.safe_load(f)
    compare_configs(trainer.cfg.raw_config, restored_config)

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
    assert set(expected_model_keys.keys()) == set(restored_model_dict_consolidated.keys()), (
        "Mismatch between in-memory and on-disk consolidated model keys."
    )

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory optimizer state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_optim_keys.keys()) == set(restored_optim_dict.keys()), (
        "Mismatch between in-memory and on-disk optimizer keys."
    )

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk consolidated model state
    if torch.distributed.get_rank() != 0:
        assert len(model_state_dict) == 0, "Model state dict should be empty on non-rank-0 processes"

    if torch.distributed.get_rank() == 0:
        for k in expected_model_keys.keys():
            v = model_state_dict[k].cpu()
            assert k in restored_model_dict_consolidated, f"Key {k} not found in restored model state"
            assert isinstance(
                restored_model_dict_consolidated[k],
                torch.Tensor,
            ), f"Value for key {k} is not a tensor"

            # Get expected shape, dtype, device from expected_model_keys
            expected_shape, expected_dtype, expected_device = expected_model_keys[k]

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

    # finally check if the adapters loaded into the PEFT module are the same as the model we have trained
    if torch.distributed.get_rank() == 0:
        base = AutoModelForImageTextToText.from_pretrained(cfg.model.pretrained_model_name_or_path)
        peft_model = PeftModel.from_pretrained(
            base, Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model"
        ).to(trainer.model.dtype)

        for source_key, source_param in model_state_dict.items():
            # source key example: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
            for peft_model_key, peft_model_param in peft_model.named_parameters():
                if "lora" in peft_model_key and source_key.rsplit(".", 1)[0] in peft_model_key:
                    assert torch.allclose(source_param, peft_model_param), (
                        "Parameter values are different when they should be the same"
                    )
    torch.distributed.barrier()

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


def _compare_dicts(expected: dict, restored: dict):
    assert len(restored) == len(expected), (
        f"Mismatch between in-memory and on-disk config. Expected length {len(expected)} but got {len(restored)}"
    )

    for k, v in expected.items():
        assert k in restored, "Key {} not found in restored config".format(k)
        error_msg = "Mismatch between in-memory and on-disk config. Expected {{}} but got {{}}"
        if isinstance(v, list):
            assert sorted(restored[k]) == sorted(v), error_msg.format(sorted(v), sorted(restored[k]))
        else:
            assert restored[k] == v, error_msg.format(v, restored[k])
