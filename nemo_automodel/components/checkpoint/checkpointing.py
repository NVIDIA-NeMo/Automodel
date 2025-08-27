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
import logging
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
from torch.distributed.device_mesh import DeviceMesh

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
            model_part = model[0] if isinstance(model, list) else model

            if hasattr(model_part, "config"):
                with open(os.path.join(consolidated_model_path, "config.json"), "w") as f:
                    f.write(model_part.config.to_json_string())
            # save the generation_config.json file
            if hasattr(model_part, "generation_config"):
                with open(os.path.join(consolidated_model_path, "generation_config.json"), "w") as f:
                    f.write(model_part.generation_config.to_json_string())

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
            index_path = get_safetensors_index_path(
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


def load_model_from_base_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    is_peft: bool,
    root_dir: str,
    model_name: str,
    peft_init_method: str,
    device_mesh: Optional[DeviceMesh] = None,
):
    """
    Load a model from the base Hugging Face checkpoint in parallel.

    Args:
        model: Model to load state into
        device: Device to load model onto
        is_peft: Whether the model is PEFT
        root_dir: Root directory of the model
        model_name: Name of the model
    """
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    to_empty_parameters_only(model, device=device)

    # HF models set _is_hf_initialized to True after initialization.
    # But because we initialize on meta device, these are erroneously set to True.
    # We need to set them to False and call initialize_weights to re-initialize the weights.

    # Gemma3ForConditionalGeneration cannot be pretrained currently. The pinned torch version
    # doesn't support initialize_weights when the model is sharded. This is because Gemma's
    # initialize_weights method requires setting a row to zeros in the embedding matrix.
    # This index selection op is not supported for DTensors in the pinned torch version.
    if not isinstance(model, Gemma3ForConditionalGeneration):
        for _, module in model.named_modules():
            if hasattr(module, "_is_hf_initialized"):
                module._is_hf_initialized = False

        # init model weights
        if hasattr(model, "initialize_weights"):
            model.initialize_weights()
        else:
            logging.warning(
                "Warning: Model does not have initialize_weights method. Requires custom initialization to be implemented."
            )

    # init peft adapters with the scaled weights
    _init_peft_adapters(model, peft_init_method)

    model_state = ModelState(model, is_peft=is_peft, is_init_step=True)
    model_state_dict = model_state.state_dict()
    if os.path.exists(model_name):
        # offline models will pass in the model path directly
        model_path = model_name
    else:
        model_path = get_safetensors_index_path(root_dir, model_name)
    dcp.load(
        model_state_dict,
        storage_reader=_HuggingFaceStorageReader(
            model_path, key_mapping=getattr(model, "_checkpoint_conversion_mapping", None)
        ),
        process_group=device_mesh["pp"].get_group() if device_mesh else None,
    )
    model_state.load_state_dict(model_state_dict)
    if hasattr(model, "tie_weights") and model_state.is_tied_lm_head:
        model.tie_weights()


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
        # no PP support for PEFT models
        model = model[0] if isinstance(model, list) else model
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


def get_safetensors_index_path(cache_dir: str, repo_id: str) -> str:
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


def to_empty_parameters_only(model: nn.Module, *, device: torch.device, recurse: bool = True) -> nn.Module:
    """
    Move parameters to the specified device without copying storage, skipping buffers.

    Mirrors torch.nn.Module.to_empty but applies only to parameters, not buffers.

    Args:
        model: The module to transform
        device: Target device
        recurse: Whether to recurse into child modules

    Returns:
        The same module instance
    """
    return _apply(model, lambda t: torch.empty_like(t, device=device), recurse=recurse)


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
    target_modules = _extract_target_modules(model_state.model[0])
    try:
        model_task = model_state.model[0].config.architectures[0].split("For")[-1]
    except (AttributeError, IndexError, TypeError):
        model_task = "N/A"

    try:
        name_or_path = model_state.model[0].config.name_or_path
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

    Note: When torch.compile is used, module names get prefixed with '_orig_mod.'.
    This function strips those prefixes to get the original module names.
    """
    final_target_modules = set()
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            # Remove the torch.compile _orig_mod prefix if present
            target_name = name.rsplit(".", 1)[0]
            if target_name.startswith("_orig_mod."):
                target_name = target_name[len("_orig_mod.") :]
            final_target_modules.add(target_name)
    return sorted(list(final_target_modules))


def _init_peft_adapters(model: nn.Module, peft_init_method: str):
    """
    Initialize the PEFT adapters with the scaled weights.

    Args:
        model: Model to initialize PEFT adapters for
        peft_init_method: Method to initialize PEFT adapters e.g. "xavier". See `LinearLoRA` for more details.
    """
    for module in model.modules():
        if hasattr(module, "init_lora_weights"):
            try:
                module.init_lora_weights(peft_init_method)
            except Exception as e:
                logging.warning(f"Failed to initialize weights for PEFT adapter `{module.__class__.__name__}`: {e}")


def _apply(module, fn, recurse=True):
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if recurse:
        for child in module.children():
            _apply(child, fn, recurse=recurse)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            # If the new tensor has compatible tensor type as the existing tensor,
            # the current behavior is to change the tensor in-place using `.data =`,
            # and the future behavior is to overwrite the existing tensor. However,
            # changing the current behavior is a BC-breaking change, and we want it
            # to happen in future releases. So for now we introduce the
            # `torch.__future__.get_overwrite_module_params_on_conversion()`
            # global flag to let the user control whether they want the future
            # behavior of overwriting the existing tensor or not.
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    should_use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()
    for key, param in module._parameters.items():
        if param is None:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            param_applied = fn(param)
        p_should_use_set_data = compute_should_use_set_data(param, param_applied)

        # subclasses may have multiple child tensors so we need to use swap_tensors
        p_should_use_swap_tensors = should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)

        param_grad = param.grad
        if p_should_use_swap_tensors:
            try:
                if param_grad is not None:
                    # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                    # Decrement use count of the gradient by setting to None
                    param.grad = None
                param_applied = torch.nn.Parameter(param_applied, requires_grad=param.requires_grad)
                torch.utils.swap_tensors(param, param_applied)
            except Exception as e:
                if param_grad is not None:
                    param.grad = param_grad
                raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}") from e
            out_param = param
        elif p_should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, torch.nn.Parameter)
            assert param.is_leaf
            out_param = torch.nn.Parameter(param_applied, param.requires_grad)
            module._parameters[key] = out_param

        if param_grad is not None:
            with torch.no_grad():
                grad_applied = fn(param_grad)
            g_should_use_set_data = compute_should_use_set_data(param_grad, grad_applied)
            if p_should_use_swap_tensors:
                grad_applied.requires_grad_(param_grad.requires_grad)
                try:
                    torch.utils.swap_tensors(param_grad, grad_applied)
                except Exception as e:
                    raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}.grad") from e
                out_param.grad = param_grad
            elif g_should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param_grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)

    return module
