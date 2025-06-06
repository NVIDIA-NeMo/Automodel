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

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from transformers import AutoConfig, AutoTokenizer
from nemo_automodel.checkpoint.hf_storage import (
    HuggingFaceStorageWriter,
    HuggingFaceStorageReader,
)
from nemo_automodel.checkpoint.hf_planner import (
    HuggingFaceSavePlanner,
    HuggingFaceLoadPlanner,
)
from torch.distributed.checkpoint import DefaultSavePlanner


def save(
    weights_path: str,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: Optional[dict[str, Any]] = None,
    scheduler_state_dict: Optional[dict[str, Any]] = None,
    dataloader_state_dict: Optional[dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    # reference_model_path: str = "/opt/models/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/",
    reference_model_path: str = "/opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/",
) -> None:
    """Save a checkpoint of the model and optionally optimizer state.

    Args:
        weights_path: Path to save model weights
        model_state_dict: Model state dictionary to save
        optimizer_state_dict: Optional optimizer state dictionary to save
        scheduler_state_dict: Optional scheduler state dictionary to save
        dataloader_state_dict: Optional dataloader state dictionary to save
        tokenizer: Optional tokenizer to save
        reference_model_path: Path to reference model to copy file structure from.
    """
    # Load mapping from reference model
    hf_storage_reader = HuggingFaceStorageReader(reference_model_path)
    metadata = hf_storage_reader.read_metadata()
    weight_map = metadata.storage_data
    
    fqn_to_file_index_mapping = {}
    for fqn, filename in weight_map.items():
        if "-" in filename:
            index = int(filename.split("-")[1])
            fqn_to_file_index_mapping[fqn] = index
        else:
            # For single-file models, all tensors go to index 1
            fqn_to_file_index_mapping[fqn] = 1
    
    # Add any missing keys from the model_state_dict
    # These will go to the same file as the last file (or file 1 for single-file models)
    default_index = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
    for fqn in model_state_dict.keys():
        if fqn not in fqn_to_file_index_mapping:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"Adding missing key to mapping: {fqn}")
            fqn_to_file_index_mapping[fqn] = default_index

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Saving checkpoint to {weights_path}")
    
    model_path = os.path.join(weights_path, "model")
    optimizer_path = os.path.join(weights_path, "optimizer")
    tokenizer_path = os.path.join(weights_path, "tokenizer")

    # Only rank 0 handles directory operations to avoid race conditions
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Create fresh directories
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(optimizer_path, exist_ok=True)
        os.makedirs(tokenizer_path, exist_ok=True)
    
    # Ensure all ranks wait for rank 0 to handle directories
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    storage_writer = HuggingFaceStorageWriter(
        path=model_path,
        fqn_to_index_mapping=fqn_to_file_index_mapping,
    )

    if torch.distributed.get_rank() == 0:
        dcp.save(model_state_dict, checkpoint_id=model_path, storage_writer=storage_writer, no_dist=True, planner=DefaultSavePlanner())

    # Save optimizer and scheduler together to avoid overwrite warnings
    if optimizer_state_dict is not None or scheduler_state_dict is not None:
        combined_state = {}
        if optimizer_state_dict is not None:
            combined_state["optimizer"] = optimizer_state_dict
        if scheduler_state_dict is not None:
            combined_state["scheduler"] = scheduler_state_dict
        dcp.save(combined_state, checkpoint_id=optimizer_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(tokenizer_path)
    if dataloader_state_dict is not None:
        torch.save(dataloader_state_dict, os.path.join(weights_path, "dataloader.pt"))


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
