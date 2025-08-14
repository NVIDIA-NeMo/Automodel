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

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _HuggingFaceStorageReader,
    _HuggingFaceStorageWriter,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import (
    ModelState,
    OptimizerState,
)

if TYPE_CHECKING:
    from peft import PeftConfig
    from transformers.tokenization_utils import PreTrainedTokenizerBase


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
    is_peft: bool

    def __post_init__(self):
        """
        Convert a raw string such as "safetensors" into the right Enum.
        """
        if isinstance(self.model_save_format, str):
            self.model_save_format = SerializationFormat[self.model_save_format.upper()]


def save_model(
    model: nn.Module,
    weights_path: str,
    checkpoint_config: CheckpointingConfig,
    peft_config: Optional["PeftConfig"] = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
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
        peft_config: PEFT config
        tokenizer: Tokenizer. Only saved if checkpoint_config.save_consolidated is True.
    """
    # We also need to eventually add suport for HSDP, so we only save on non-duplicate ranks.
    model_path = os.path.join(weights_path, "model")
    consolidated_model_path = None
    if checkpoint_config.save_consolidated:
        consolidated_model_path = os.path.join(model_path, "consolidated")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(model_path, exist_ok=True)

        if (
            checkpoint_config.save_consolidated
            and checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS
            and not checkpoint_config.is_peft
        ):
            os.makedirs(consolidated_model_path, exist_ok=True)
            # save the config.json file
            if hasattr(model, "config"):
                with open(os.path.join(consolidated_model_path, "config.json"), "w") as f:
                    f.write(model.config.to_json_string())
            # save the generation_config.json file
            if hasattr(model, "generation_config"):
                with open(os.path.join(consolidated_model_path, "generation_config.json"), "w") as f:
                    f.write(model.generation_config.to_json_string())

            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(consolidated_model_path)

    # Ensure all ranks wait for rank 0 to handle directories
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model_state = ModelState(model, checkpoint_config.is_peft)

    if checkpoint_config.is_peft:
        assert peft_config is not None, "PEFT config needs to be provided when checkpointing PEFT models."
        _save_peft_adapters(model_state, peft_config, model_path)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)

    elif checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
        model_state_dict = model_state.state_dict()
        fqn_to_file_index_mapping = None
        if checkpoint_config.save_consolidated:
            # we first need to find the FQN -> .safetensors mapping
            index_path = _get_safetensors_index_path(
                checkpoint_config.model_cache_dir,
                checkpoint_config.model_repo_id,
            )
            if index_path:
                # HF VLM models may contain a special checkpoint mapping attribute
                fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                    index_path, getattr(model, "_checkpoint_conversion_mapping", None)
                )
            else:
                fqn_to_file_index_mapping = {k: 1 for k in model_state_dict.keys()}

            # Add any missing keys from the model_state_dict
            # These will go to the same file as the last file (or file 1 for single-file models)
            default_index = max(fqn_to_file_index_mapping.values())

            # TODO:(@adil-a): This will need to change when we add PP. Maybe we can cache the keys in ModelState.
            for fqn in list(model_state_dict.keys()):
                fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)

        storage_writer = _HuggingFaceStorageWriter(
            path=model_path,
            save_sharded=True,
            consolidated_output_path=consolidated_model_path,
            fqn_to_index_mapping=fqn_to_file_index_mapping,
        )
        dcp.save(
            model_state_dict,
            checkpoint_id=model_path,
            storage_writer=storage_writer,
        )
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        dcp.save(model_state.state_dict(), checkpoint_id=model_path)
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
    model_state = ModelState(model, checkpoint_config.is_peft)

    if checkpoint_config.is_peft:
        state_dict = model.state_dict()
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            with safe_open(os.path.join(model_path, "adapter_model.safetensors"), framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
        # since we're loading the PEFT adapters on rank0, we don't need to call dcp.load
        # the call below will broadcast from rank0 to all other ranks
        model_state.load_state_dict(state_dict)

    elif checkpoint_config.model_save_format == SerializationFormat.SAFETENSORS:
        storage_reader = _HuggingFaceStorageReader(path=model_path)

        reinstated_state_dict = model_state.state_dict()
        dcp.load(
            reinstated_state_dict,
            checkpoint_id=model_path,
            storage_reader=storage_reader,
        )
        model_state.load_state_dict(reinstated_state_dict)
    elif checkpoint_config.model_save_format == SerializationFormat.TORCH_SAVE:
        reinstated_state_dict = model_state.state_dict()
        dcp.load(reinstated_state_dict, checkpoint_id=model_path)
        model_state.load_state_dict(reinstated_state_dict)
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
    dcp.save(optimizer_state.state_dict(), checkpoint_id=optimizer_path)


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

    optimizer_state = OptimizerState(model, optimizer, scheduler)
    reinstated_state_dict = optimizer_state.state_dict()
    dcp.load(reinstated_state_dict, checkpoint_id=optimizer_path)
    optimizer_state.load_state_dict(reinstated_state_dict)


def save_config(config: dict[str, Any], weights_path: str):
    """
    Save a config to a weights path.

    Args:
        config: Config to save
        weights_path: Path to save config
    """
    with open(os.path.join(weights_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def _get_safetensors_index_path(cache_dir: str, repo_id: str) -> str:
    """
    Return the directory containing the first `model.safetensors.index.json` found for given model.

    If no `model.safetensors.index.json` is found then it returns None.

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


def _save_peft_adapters(
    model_state: ModelState,
    peft_config: "PeftConfig",
    model_path: str,
):
    """
    Save PEFT adapters to a weights path.
    """
    hf_peft_config = _get_hf_peft_config(peft_config, model_state)
    automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
    state_dict = model_state.state_dict()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # save in HF format. Only keys that are needed for PEFT module loading will be saved here.
        with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
            json.dump(hf_peft_config, f, indent=2, sort_keys=True)
        # save the full PEFT config for inference loading inside Automodel.
        with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
            json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)
        save_file(state_dict, os.path.join(model_path, "adapter_model.safetensors"))


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState) -> dict:
    """
    Get the PEFT config in the format expected by Hugging Face.
    """
    MODEL_TYPE_TO_PEFT_TASK_TYPE = {
        "SequenceClassification": "SEQ_CLS",
        "Seq2SeqLM": "SEQ_2_SEQ_LM",
        "CausalLM": "CAUSAL_LM",
        "TokenClassification": "TOKEN_CLS",
        "QuestionAnswering": "QUESTION_ANS",
        "FeatureExtraction": "FEATURE_EXTRACTION",
    }
    target_modules = _extract_target_modules(model_state.model)
    try:
        model_task = model_state.model.config.architectures[0].split("For")[-1]
    except (AttributeError, IndexError, TypeError):
        model_task = "N/A"

    try:
        name_or_path = model_state.model.config.name_or_path
    except (AttributeError, TypeError):
        name_or_path = "N/A"

    try:
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except KeyError:
        task_type = "CAUSAL_LM"

    return {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": name_or_path,
    }


def _get_automodel_peft_metadata(peft_config: "PeftConfig") -> dict:
    """
    Get the PEFT metadata in the format expected by Automodel.
    """
    PEFT_KEYS = {"dim", "alpha"}
    return {k: v for k, v in peft_config.to_dict().items() if k not in PEFT_KEYS}


def _extract_target_modules(model: nn.Module) -> list[str]:
    """
    Extract the target modules from the model.
    """
    final_target_modules = set()
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            final_target_modules.add(name.rsplit(".", 1)[0])
    return sorted(list(final_target_modules))
