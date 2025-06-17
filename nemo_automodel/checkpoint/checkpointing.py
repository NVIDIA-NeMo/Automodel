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

"""Checkpoint management utilities for HF models."""

import os
from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed.checkpoint as dcp
from nemo_automodel.checkpoint._backports.hf_storage import (
    _HuggingFaceStorageWriter,
    _HuggingFaceStorageReader,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.checkpoint._backports.filesystem import SerializationFormat
import glob

@dataclass
class CheckpointingConfig:
    """
    Configuration for checkpointing.
    """
    enabled: bool
    checkpoint_dir: str | Path
    model_save_format: SerializationFormat | str
    model_cache_dir: str | Path
    model_repo_id: str
    save_consolidated: bool

    def __post_init__(self):
        # Convert a raw string such as "safetensors" into the right Enum
        if isinstance(self.model_save_format, str):
            self.model_save_format = SerializationFormat[
                self.model_save_format.upper()
            ]


def save_model(
        model: nn.Module,
        weights_path: str,
        checkpoint_config: CheckpointingConfig,
):
    """
    Save a model state dictionary to a weights path.

    This function can save a model in the following formats:
    - safetensors (in HF format)
    - torch_save (in DCP format)

    Args:
        model: Model to save
        weights_path: Path to save model weights
        checkpoint_config: Checkpointing configuration
    """
    # TODO(@adil-a): Need to add support for PEFT.
    # We also need to eventually add suport for HSDP, so we only save on non-duplicate ranks.
    # Add functionality to chunk different layers for different ranks to save.
    # The above functionality will also make it trivial to get a FQN -> rank mapping
    # which doesn't leave out any user modified layers.
    # This is because we need to create the mapping on the fly from the model state dict.
    model_path = os.path.join(weights_path, "model")
    consolidated_model_path = None
    if checkpoint_config.save_consolidated:
        consolidated_model_path = os.path.join(model_path, "consolidated")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(model_path, exist_ok=True)

        if (
            checkpoint_config.save_consolidated 
            and checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS
        ):
            os.makedirs(consolidated_model_path, exist_ok=True)
            # save the config.json file
            with open(os.path.join(consolidated_model_path, "config.json"), "w") as f:
                f.write(model.config.to_json_string())

    # Ensure all ranks wait for rank 0 to handle directories
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model_state = ModelState(model, checkpoint_config.model_save_format)
    
    if checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
        fqn_to_file_index_mapping = None
        if checkpoint_config.save_consolidated:
            # we first need to find the FQN -> .safetensors mapping
            index_path = _get_safetensors_index_path(
                checkpoint_config.model_cache_dir,
                checkpoint_config.model_repo_id,
            )
            if index_path:
                fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(index_path)

                # Add any missing keys from the model_state_dict
                # These will go to the same file as the last file (or file 1 for single-file models)
                default_index = max(fqn_to_file_index_mapping.values())

                # TODO:(@adil-a): This will need to change when we add PP. Maybe we can cache the keys in ModelState.
                for fqn in list(model.state_dict().keys()):
                    if fqn not in fqn_to_file_index_mapping:
                        if model_state.is_tied_lm_head and fqn == "lm_head.weight":
                            continue
                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                            print(f"Adding missing key to mapping: {fqn}")
                        fqn_to_file_index_mapping[fqn] = default_index

        storage_writer = _HuggingFaceStorageWriter(
            path=model_path,
            save_sharded=True,
            consolidated_output_path=consolidated_model_path,
            fqn_to_index_mapping=fqn_to_file_index_mapping,
        )

        dcp.save(
            {"model": model_state},
            checkpoint_id=model_path,
            storage_writer=storage_writer,
        )
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        dcp.save({"model": model_state}, checkpoint_id=model_path)
    else:
        raise ValueError(f"Unsupported model save format: {checkpoint_config.model_save_format}")


def load_model(
    model: torch.nn.Module,
    weights_path: str,
    checkpoint_config: CheckpointingConfig,
):
    """
    Load a model state dictionary from a weights path.

    Args:
        model: Model to load state into
        weights_path: Path to load model weights from
        checkpoint_config: Checkpointing configuration
    """
    model_path = os.path.join(weights_path, "model")

    # Validate checkpoint directory
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    if checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
        model_state = ModelState(model, checkpoint_config.model_save_format)
        storage_reader = _HuggingFaceStorageReader(path=model_path)

        dcp.load(
            state_dict={"model": model_state},
            checkpoint_id=model_path,
            storage_reader=storage_reader,
            planner=dcp.DefaultLoadPlanner(),
        )
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        model_state = ModelState(model, checkpoint_config.model_save_format)
        dcp.load(state_dict={"model": model_state}, checkpoint_id=model_path)
    else:
        raise ValueError(f"Unsupported model save format: {checkpoint_config.model_save_format}")


def save_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    weights_path: str,
    scheduler: Optional[Any] = None,
):
    """
    Save an optimizer state dictionary to a weights path.

    Args:
        optimizer: Optimizer to save
        model: Model to save optimizer state for
        weights_path: Path to save optimizer weights
        scheduler: Optional scheduler to save
    """
    optimizer_path = os.path.join(weights_path, "optim")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(optimizer_path, exist_ok=True)
    optimizer_state = OptimizerState(model, optimizer, scheduler)
    dcp.save({"optim": optimizer_state}, checkpoint_id=optimizer_path)


def load_optimizer(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    weights_path: str,
    scheduler: Optional[Any] = None,
):
    """
    Load an optimizer state dictionary from a weights path.

    Args:
        optimizer: Optimizer to load state into
        model: Model to load optimizer state for
        weights_path: Path to load optimizer weights from
        scheduler: Optional scheduler to load state into
    """
    optimizer_path = os.path.join(weights_path, "optim")
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"Optimizer path {optimizer_path} does not exist")

    optimizer_state = {"optim": OptimizerState(model, optimizer, scheduler)}
    dcp.load(state_dict=optimizer_state, checkpoint_id=optimizer_path)


def _get_safetensors_index_path(cache_dir: str, repo_id: str) -> str:
    """
    Return the directory containing the first `model.safetensors.index.json` found
    for a given model, or ``None`` if it does not exist in the cache yet.

    For example, if the file located is

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe.../model.safetensors.index.json

    this function will return the directory path

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe...

    This will error if the model hasn't been downloaded or if the cache directory is incorrect.

    Args:
        cache_dir: Path to cache directory
        repo_id: Hugging Face repository ID

    Returns:
        Path to the directory containing the index file.
    
    Raises:
        FileNotFoundError: If the index file is not found.
    """
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_root = Path(cache_dir) / repo_dir / "snapshots"

    # Look for an index file inside any snapshot directory.
    pattern = snapshots_root / "*" / "model.safetensors.index.json"
    matches = glob.glob(str(pattern))
    if matches:
        # Return the directory path that contains the index file.
        return str(Path(matches[0]).parent)

    # Fall back: if no index file, return the first available snapshot directory (if any).
    # This is the case for single-file models.
    snapshot_dirs = [p for p in glob.glob(str(snapshots_root / "*")) if Path(p).is_dir()]
    if snapshot_dirs:
        try:
            return snapshot_dirs[0]
        except IndexError:
            raise FileNotFoundError(f"No snapshot directories found in {snapshots_root}")