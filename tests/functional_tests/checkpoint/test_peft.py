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

import os
from pathlib import Path
import json

from recipes.llm.finetune import FinetuneRecipeForNextTokenPrediction
from nemo_automodel.config.cli import parse_args_and_load_config
from nemo_automodel.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
import torch
import torch.distributed.tensor
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

def load_dcp(ckpt_dir: Path | str) -> dict[str, torch.Tensor]:
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

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == 'TensorStorageMetadata'
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict

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
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.2.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.3.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.4.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.5.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.6.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.7.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.8.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.9.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.10.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.11.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.12.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.13.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.14.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.q_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.q_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.k_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.k_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.v_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.v_proj.lora_B.weight": ([512, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.o_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.self_attn.o_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.gate_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.gate_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.up_proj.lora_A.weight": ([8, 2048], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.up_proj.lora_B.weight": ([8192, 8], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.down_proj.lora_A.weight": ([8, 8192], torch.float32, "cpu"),
        "base_model.model.model.layers.15.mlp.down_proj.lora_B.weight": ([2048, 8], torch.float32, "cpu"),
        "base_model.model.lm_head.weight": ([128256, 2048], torch.float32, "cpu"),
    }
    expected_optim_keys = {
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.2.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.3.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.4.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.5.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.6.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.7.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.8.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.9.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.10.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.11.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.12.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.13.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.14.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.q_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.k_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_B.weight.exp_avg": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.v_proj.lora_B.weight.exp_avg_sq": ([256, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.self_attn.o_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.gate_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_A.weight.exp_avg": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_A.weight.exp_avg_sq": ([4, 2048], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_B.weight.exp_avg": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.up_proj.lora_B.weight.exp_avg_sq": ([4096, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_A.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_A.weight.exp_avg": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_A.weight.exp_avg_sq": ([4, 8192], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_B.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_B.weight.exp_avg": ([1024, 8], torch.float32, "cpu"),
        "optim.optim.state.model.layers.15.mlp.down_proj.lora_B.weight.exp_avg_sq": ([1024, 8], torch.float32, "cpu"),
    }
    expected_config = {
        # "base_model_name_or_path": "mistralai/Mixtral-8x7B-v0.1",
        "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
        "bias": "none",
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": [
            "o_proj",
            "k_proj",
            "v_proj",
            "up_proj",
            "q_proj",
            "gate_proj",
            "down_proj"
        ],
        "task_type": "CAUSAL_LM"
    }

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama_3_2_1b_hellaswag_peft.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = ModelState(
        trainer.model,
        trainer.checkpoint_config.model_save_format,
        trainer.checkpoint_config.is_peft,
    ).state_dict()
    optimizer_state_dict = to_cpu(OptimizerState(
        trainer.model,
        trainer.optimizer,
        trainer.step_scheduler,
    ).state_dict()["optim"]["state"])

    # assert the correct paths exist
    output_files = [
        "model",
        "optim",
        "step_scheduler.pt",
        "dataloader.pt",
        "model/adapter_model.safetensors",
        "model/adapter_config.json",
        "optim/__0_0.distcp",
        "optim/__1_0.distcp",
        "optim/.metadata",
        "step_scheduler.pt",
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
    restored_optim_dict = load_dcp(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "optim",
    )
    restored_model_dict_consolidated = load_safetensors(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model" / "adapter_model.safetensors",
    )
    restored_config = json.load(
        open(Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model" / "adapter_config.json"),
    )
    assert restored_config == expected_config, f"Mismatch between in-memory and on-disk config. Expected {expected_config} but got {restored_config}"

    # at save time, the model is saved in a dictionary formatted as:
    # {
    #     "model": ModelState(...)
    # }
    # because of this, DCP will flatten the model state dictionary to:
    # {
    #     "model.model.embed_tokens.weight": ...
    # }
    # so we need to remove the first occurrence of "model." from the keys

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
    #     "optim.optim.state.model.layers.0.self_attn.q_proj.weight.step": ...
    # }
    # so we flatten the in-memory optimizer state dictionary to match the on-disk view
    flattened_optim_dict = _flatten(optimizer_state_dict, parent_key="optim.optim.state")

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
        assert len(model_state_dict) == 0, (
            "Model state dict should be empty on non-rank-0 processes"
        )

    if torch.distributed.get_rank() == 0:
        for k in expected_model_keys.keys():
            v = model_state_dict[k].cpu()
            assert k in restored_model_dict_consolidated, f"Key {k} not found in restored model state"
            assert isinstance(
                restored_model_dict_consolidated[k], torch.Tensor,
            ), f"Value for key {k} is not a tensor"

            # Get expected shape, dtype, device from expected_model_keys
            expected_shape, expected_dtype, expected_device = expected_model_keys[k]

            full_shard = restored_model_dict_consolidated[k]

            assert list(full_shard.shape) == expected_shape, (
                f"Shape mismatch for key {k}. "
                f"Expected shape {expected_shape} but got {full_shard.shape}"
            )
            assert full_shard.dtype == expected_dtype, (
                f"Dtype mismatch for key {k}. "
                f"Expected dtype {expected_dtype} but got {full_shard.dtype}"
            )
            assert str(full_shard.device) == expected_device, (
                f"Device mismatch for key {k}. "
                f"Expected device {expected_device} but got {full_shard.device}"
            )
            assert torch.allclose(v, full_shard), (
                f"Value mismatch for key {k}. "
                f"Tensors are not numerically close"
            )

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk optimizer state
    for k, v in flattened_optim_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
        assert k in restored_optim_dict, f"Key {k} not found in restored optimizer state"
        assert isinstance(
            restored_optim_dict[k], torch.Tensor,
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
            f"Shape mismatch for key {k}. "
            f"Expected shape {expected_shape} but got {curr_shard.shape}"
        )
        assert curr_shard.dtype == expected_dtype, (
            f"Dtype mismatch for key {k}. "
            f"Expected dtype {expected_dtype} but got {curr_shard.dtype}"
        )
        assert str(curr_shard.device) == expected_device, (
            f"Device mismatch for key {k}. "
            f"Expected device {expected_device} but got {curr_shard.device}"
        )
        assert torch.allclose(v, curr_shard), (
            f"Value mismatch for key {k}. "
            f"Tensors are not numerically close"
        )


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

test_hf_peft_checkpoint()