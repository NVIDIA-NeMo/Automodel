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
from typing import Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed.checkpoint as dcp
from nemo_automodel.checkpoint.hf_storage import (
    HuggingFaceStorageWriter,
    HuggingFaceStorageReader,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.checkpoint.stateful_wrappers import ModelState, OptimizerState
import glob
import json
from torch.distributed.checkpoint.filesystem import SerializationFormat

PathLike = Union[str, "os.PathLike[Any]"]
    

@dataclass
class CheckpointingConfig:
    enabled: bool
    checkpoint_dir: str | Path
    model_save_format: SerializationFormat | str
    model_cache_dir: str | Path
    model_repo_id: str

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
    # The above functionality will also make it trivial to get a FQN -> rank mapping which doesn't leave out any user modified layers.
        # This is because we need to create the mapping on the fly from the model state dict.
    model_path = os.path.join(weights_path, "model")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(model_path, exist_ok=True)

        # save the config.json file
        if checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
            with open(os.path.join(model_path, "config.json"), "w") as f:
                f.write(model.config.to_json_string())

    # Ensure all ranks wait for rank 0 to handle directories
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model_state_dict = ModelState(model, checkpoint_config.model_save_format).state_dict()
    
    if checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
        # we first need to find the FQN -> .safetensors mapping
        index_path = _get_safetensors_index_path(
            checkpoint_config.model_cache_dir,
            checkpoint_config.model_repo_id,
        )
        fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(index_path)

        # Add any missing keys from the model_state_dict
        # These will go to the same file as the last file (or file 1 for single-file models)
        default_index = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
        for fqn in model_state_dict.keys():
            if fqn not in fqn_to_file_index_mapping:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"Adding missing key to mapping: {fqn}")
                fqn_to_file_index_mapping[fqn] = default_index          

        storage_writer = HuggingFaceStorageWriter(
            path=model_path,
            fqn_to_index_mapping=fqn_to_file_index_mapping,
        )

        if torch.distributed.get_rank() == 0:
            dcp.save(
                model_state_dict,
                checkpoint_id=model_path,
                storage_writer=storage_writer,
                no_dist=True,
            )
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        dcp.save({"model": model_state_dict}, checkpoint_id=model_path)
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
        # For HF safetensors we rely on the custom HuggingFaceStorageReader which
        # understands both sharded and single-file checkpoints. Since we saved the
        # model with `no_dist=True`, we also load with `no_dist=True` so that only
        # rank-0 touches the checkpoint files and then broadcasts tensors to the
        # remaining ranks under the hood.

        # Destination state dict – FSDP aware via ModelState
        model_state_dict = {
            "model": ModelState(model, checkpoint_config.model_save_format).state_dict()
        }

        storage_reader = HuggingFaceStorageReader(path=model_path)

        # Allow the planner to resize tensors when the destination param has been
        # replaced/expanded (e.g., when users add new special tokens). For exact
        # restores this is a no-op.
        from nemo_automodel.checkpoint.hf_planner import HuggingFaceLoadPlanner

        load_planner = HuggingFaceLoadPlanner(allow_tensor_resize=True)

        dcp.load(
            state_dict=model_state_dict,
            checkpoint_id=model_path,
            storage_reader=storage_reader,
            planner=load_planner,
            no_dist=True,
        )
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        model_state_dict = {"model": ModelState(model, checkpoint_config.model_save_format).state_dict()}
        dcp.load(state_dict=model_state_dict, checkpoint_id=model_path)
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
    optimizer_state_dict = OptimizerState(model, optimizer, scheduler).state_dict()
    dcp.save({"optim": optimizer_state_dict}, checkpoint_id=optimizer_path)

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

    optimizer_state_dict = {"optim": OptimizerState(model, optimizer, scheduler).state_dict()}
    dcp.load(state_dict=optimizer_state_dict, checkpoint_id=optimizer_path)

# def save(
#     weights_path: str,
#     model_state_dict: dict[str, Any],
#     optimizer_state_dict: Optional[dict[str, Any]] = None,
#     scheduler_state_dict: Optional[dict[str, Any]] = None,
#     dataloader_state_dict: Optional[dict[str, Any]] = None,
#     tokenizer: Optional[Any] = None,
#     # reference_model_path: str = "/opt/models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/",
#     reference_model_path: str = "/opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/",
# ) -> None:
#     """Save a checkpoint of the model and optionally optimizer state.

#     Args:
#         weights_path: Path to save model weights
#         model_state_dict: Model state dictionary to save
#         optimizer_state_dict: Optional optimizer state dictionary to save
#         scheduler_state_dict: Optional scheduler state dictionary to save
#         dataloader_state_dict: Optional dataloader state dictionary to save
#         tokenizer: Optional tokenizer to save
#         reference_model_path: Path to reference model to copy file structure from.
#     """
#     # Load mapping from reference model
#     hf_storage_reader = HuggingFaceStorageReader(reference_model_path)
#     metadata = hf_storage_reader.read_metadata()
#     weight_map = metadata.storage_data
    
#     fqn_to_file_index_mapping = {}
#     for fqn, filename in weight_map.items():
#         if "-" in filename:
#             index = int(filename.split("-")[1])
#             fqn_to_file_index_mapping[fqn] = index
#         else:
#             # For single-file models, all tensors go to index 1
#             fqn_to_file_index_mapping[fqn] = 1
    
#     # Add any missing keys from the model_state_dict
#     # These will go to the same file as the last file (or file 1 for single-file models)
#     default_index = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
#     for fqn in model_state_dict.keys():
#         if fqn not in fqn_to_file_index_mapping:
#             if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#                 print(f"Adding missing key to mapping: {fqn}")
#             fqn_to_file_index_mapping[fqn] = default_index

#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         print(f"Saving checkpoint to {weights_path}")
    
#     model_path = os.path.join(weights_path, "model")
#     optimizer_path = os.path.join(weights_path, "optimizer")
#     tokenizer_path = os.path.join(weights_path, "tokenizer")

#     # Only rank 0 handles directory operations to avoid race conditions
#     if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#         # Create fresh directories
#         os.makedirs(model_path, exist_ok=True)
#         os.makedirs(optimizer_path, exist_ok=True)
#         os.makedirs(tokenizer_path, exist_ok=True)
    
#     # Ensure all ranks wait for rank 0 to handle directories
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()

#     storage_writer = HuggingFaceStorageWriter(
#         path=model_path,
#         fqn_to_index_mapping=fqn_to_file_index_mapping,
#     )

#     if torch.distributed.get_rank() == 0:
#         dcp.save(model_state_dict, checkpoint_id=model_path, storage_writer=storage_writer, no_dist=True, planner=DefaultSavePlanner())

#     # Save optimizer and scheduler together to avoid overwrite warnings
#     if optimizer_state_dict is not None or scheduler_state_dict is not None:
#         combined_state = {}
#         if optimizer_state_dict is not None:
#             combined_state["optimizer"] = optimizer_state_dict
#         if scheduler_state_dict is not None:
#             combined_state["scheduler"] = scheduler_state_dict
#         dcp.save(combined_state, checkpoint_id=optimizer_path)

#     if tokenizer is not None:
#         tokenizer.save_pretrained(tokenizer_path)
#     if dataloader_state_dict is not None:
#         torch.save(dataloader_state_dict, os.path.join(weights_path, "dataloader.pt"))


def load(
    model: torch.nn.Module,
    weights_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    optimizer_path: Optional[str] = None,
) -> None:
    """Load a model weights and optionally optimizer state.

    Args:
        model: The PyTorch model whose weights to update
        weights_path: Path to load model weights from
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        optimizer_path: Path to load optimizer state from (required if optimizer provided)
    """
    print(f"Loading weights from {weights_path}")
    model_state_dict = {"model": ModelState(model)}
    dcp.load(state_dict=model_state_dict, checkpoint_id=weights_path)

    if optimizer is not None:
        if optimizer_path is None:
            raise ValueError(
                "optimizer_path must be provided when loading optimizer state"
            )
        print(f"Loading optimizer from {optimizer_path}")
        optimizer_state_dict = {"optim": OptimizerState(model, optimizer, scheduler)}
        dcp.load(state_dict=optimizer_state_dict, checkpoint_id=optimizer_path)


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