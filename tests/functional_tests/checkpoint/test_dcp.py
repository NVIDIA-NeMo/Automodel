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
"""Tests for DCP checkpointing."""

import os
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor

from nemo_automodel.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.config.cli import parse_args_and_load_config
from recipes.llm.finetune import FinetuneRecipeForNextTokenPrediction


def load_dcp(ckpt_dir: Path | str) -> dict[str, torch.Tensor]:
    """Loads a DCP checkpoint in a state dictionary from a directory.

    Args:
        ckpt_dir: The directory containing the DCP checkpoint.

    Returns:
        A state dictionary containing the checkpoint.
    """
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = dcp.FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    dcp.load(
        state_dict,
        storage_reader=fs_reader,
    )
    return state_dict


def to_cpu(
    state_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Converts a state dictionary to CPU."""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else to_cpu(v) for k, v in state_dict.items()}


def test_dcp_checkpoint():
    """Tests DCP checkpoint"""
    expected_model_keys = {
        "model.embed_tokens.weight": ([16000, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.q_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.k_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.v_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.0.self_attn.o_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.gate.weight": ([4, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.2.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.3.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.4.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.5.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.6.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.0.block_sparse_moe.experts.7.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.0.input_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.0.post_attention_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.q_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.k_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.v_proj.weight": ([64, 512], torch.bfloat16, "cpu"),
        "model.layers.1.self_attn.o_proj.weight": ([256, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.gate.weight": ([4, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.0.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.1.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.2.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.3.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.4.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.5.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.6.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w1.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w2.weight": ([256, 448], torch.bfloat16, "cpu"),
        "model.layers.1.block_sparse_moe.experts.7.w3.weight": ([224, 512], torch.bfloat16, "cpu"),
        "model.layers.1.input_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.layers.1.post_attention_layernorm.weight": ([256], torch.bfloat16, "cpu"),
        "model.norm.weight": ([256], torch.bfloat16, "cpu"),
        "lm_head.weight": ([16000, 512], torch.bfloat16, "cpu"),
    }
    expected_optim_keys = {
        "optim.optim.state.model.layers.0.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.q_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.k_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.v_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.self_attn.o_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.gate.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.gate.weight.exp_avg": ([4, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.gate.weight.exp_avg_sq": ([4, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.0.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.input_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.input_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.0.post_attention_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.0.post_attention_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.q_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.k_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.weight.exp_avg": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.v_proj.weight.exp_avg_sq": ([64, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.weight.exp_avg": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.self_attn.o_proj.weight.exp_avg_sq": ([256, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.gate.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.gate.weight.exp_avg": ([4, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.gate.weight.exp_avg_sq": ([4, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.optim.state.model.layers.1.input_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.input_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.input_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.post_attention_layernorm.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.layers.1.post_attention_layernorm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.layers.1.post_attention_layernorm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.norm.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.model.norm.weight.exp_avg": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.model.norm.weight.exp_avg_sq": ([256], torch.bfloat16, "cpu"),
        "optim.optim.state.lm_head.weight.step": ([], torch.float32, "cpu"),
        "optim.optim.state.lm_head.weight.exp_avg": ([16000, 512], torch.bfloat16, "cpu"),
        "optim.optim.state.lm_head.weight.exp_avg_sq": ([16000, 512], torch.bfloat16, "cpu"),
    }

    script_path = Path(__file__).parent.resolve()
    cfg = parse_args_and_load_config(script_path / "llama_3_2_1b_hellaswag.yaml")
    trainer = FinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = to_cpu(
        ModelState(
            trainer.model,
            trainer.checkpoint_config.model_save_format,
        ).state_dict()
    )
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
        "model/__0_0.distcp",
        "model/__1_0.distcp",
        "model/.metadata",
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
    restored_model_dict = load_dcp(
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model",
    )

    # at save time, the model is saved in a dictionary formatted as:
    # {
    #     "model": ModelState(...)
    # }
    # because of this, DCP will flatten the model state dictionary to:
    # {
    #     "model.model.embed_tokens.weight": ...
    # }
    # so we need to remove the first occurrence of "model." from the keys
    restored_model_dict = {k.replace("model.", "", 1): v for k, v in restored_model_dict.items()}

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
    assert set(expected_model_keys.keys()) == set(restored_model_dict.keys()), (
        "Mismatch between in-memory and on-disk model keys."
    )

    # ---------------------------------------------------------------------
    # Compare the flattened in-memory optimizer state with the on-disk view
    # ---------------------------------------------------------------------
    assert set(expected_optim_keys.keys()) == set(restored_optim_dict.keys()), (
        "Mismatch between in-memory and on-disk optimizer keys."
    )

    # Compare the values, shapes, dtype, and device of the in-memory and on-disk model state
    for k, v in model_state_dict.items():
        if isinstance(v, torch.distributed.tensor.DTensor):
            v = v.to_local()
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


test_dcp_checkpoint()
