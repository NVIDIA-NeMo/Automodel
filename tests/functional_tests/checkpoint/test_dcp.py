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
import torch.nn as nn
import yaml

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction, calculate_loss


def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """Loads a DCP checkpoint in a state dictionary from a directory."""
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
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


def to_cpu(
    state_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Converts a state dictionary to CPU."""
    return {k: v.cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def get_validation_loss(
    model_parts: list[nn.Module], val_batch: dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device, pp_enabled: bool, pp,
) -> torch.Tensor:
    """Gets the validation loss for a model."""
    loss_buffer = []
    val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
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
        "optim.state.model.embed_tokens.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.embed_tokens.weight.exp_avg": ([16000, 512], torch.bfloat16, "cpu"),
        "optim.state.model.embed_tokens.weight.exp_avg_sq": ([16000, 512], torch.bfloat16, "cpu"),
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
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.0.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [224, 512],
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
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.0.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.1.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.2.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.3.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.4.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.5.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.6.w3.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w1.weight.exp_avg_sq": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w2.weight.exp_avg_sq": (
            [256, 448],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.step": ([], torch.float32, "cpu"),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg": (
            [224, 512],
            torch.bfloat16,
            "cpu",
        ),
        "optim.state.model.layers.1.block_sparse_moe.experts.7.w3.weight.exp_avg_sq": (
            [224, 512],
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

    cfg_path = Path(__file__).parents[3] / "examples" / "llm_finetune" / "llama3_2" / "llama3_2_1b_hellaswag.yaml"
    cfg = parse_args_and_load_config(cfg_path)
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # checkpoint is saved at this point
    # first extract the in-memory checkpoint
    model_state_dict = to_cpu(
        ModelState(
            trainer.model_parts,
        ).state_dict()
    )
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
        "model/__0_0.distcp",
        "model/__1_0.distcp",
        "model/.metadata",
        "optim/__0_0.distcp",
        "optim/__1_0.distcp",
        "optim/.metadata",
        "step_scheduler.pt",
        "config.yaml",
    ]
    if trainer._get_dp_group_size() > 1:
        output_files.append("dataloader/dataloader_dp_rank_1.pt")

    for file in output_files:
        path = Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / file
        assert path.exists(), f"Expected {path} to exist"
        if "." in file:
            assert path.is_file(), f"Expected {path} to be a file"
        else:
            assert path.is_dir(), f"Expected {path} to be a directory"
        assert os.access(path, os.R_OK), f"Expected {path} to be readable"
        assert path.stat().st_size > 0, f"Expected {path} to be non-empty"
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
        Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "model",
    )

    # check if new model and current model give the same CE loss
    val_batch = next(iter(trainer.val_dataloader))
    restored_model = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    restored_model.setup()
    source_model_loss = get_validation_loss(trainer.model_parts, val_batch, trainer.loss_fn, trainer.dist_env.device, trainer.pp_enabled, trainer.pp)
    restored_model_loss = get_validation_loss(restored_model.model_parts, val_batch, trainer.loss_fn, trainer.dist_env.device, restored_model.pp_enabled, restored_model.pp)
    assert sum(source_model_loss) == sum(restored_model_loss), "Model loss mismatch"

    # compare the recipe configs
    with open(Path(trainer.checkpoint_config.checkpoint_dir) / "epoch_0_step_10" / "config.yaml", "r") as f:
        restored_config = yaml.safe_load(f)
    compare_configs(trainer.cfg.raw_config, restored_config)

    # the saved optimizer state has an "optim." prefix that DCP adds.
    # For the on-disk view to match, it needs to be prepended with the "optim." prefix
    optimizer_state_dict = _rename_keys(optimizer_state_dict, "optim.")

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
            if 'moe' in k and 'step' in k:
                pass
            else:
                raise e
    if torch.distributed.get_rank() == 0:
        # delete the checkpoint directory
        if Path(trainer.checkpoint_config.checkpoint_dir).exists():
            shutil.rmtree(Path(trainer.checkpoint_config.checkpoint_dir))
    torch.distributed.barrier()


def _rename_keys(d: dict, prepend: str):
    """Rename the keys of *d* by prepending *prepend* to each key.
    """
    flat: dict[str, torch.Tensor] = {}
    for k, v in d.items():
        key = f"{prepend}{k}"
        flat[key] = v
    return flat
