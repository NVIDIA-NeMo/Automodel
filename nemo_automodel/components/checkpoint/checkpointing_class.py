
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

from typing import Optional, TYPE_CHECKING, Any

from dataclasses import dataclass
from pathlib import Path
from torch.futures import Future
import glob
from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
from torch.distributed.device_mesh import DeviceMesh
from nemo_automodel.components.checkpoint.addons import PeftAddon, ConsolidatedHFAddon
from torch import nn
from safetensors.torch import save_file
import torch

import torch.distributed.checkpoint as dcp
import os
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.checkpoint._backports.hf_storage import get_fqn_to_file_index_mapping, _HuggingFaceStorageWriter, _HuggingFaceStorageReader
from packaging.version import parse
import logging
import yaml
from safetensors.torch import load_file


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
    model_save_format: str
    model_cache_dir: str | Path
    model_repo_id: str
    save_consolidated: bool
    is_peft: bool
    model_state_dict_keys: list[str] = None  # copy of the model state dict keys before any parallelization
    is_async: bool = False

    def __post_init__(self):
        """
        Convert a raw string such as "safetensors" into the right Enum.
        """
        assert self.model_save_format in SerializationFormat, f"Unsupported model save format: {self.model_save_format}"
        self.model_save_format = SerializationFormat[self.model_save_format.upper()]
        
        # Async is only enabled for torch >= 2.9.0 currently because of large API changes in async DCP from 2.8.0 to 2.9.0
        if self.is_async and parse(torch.__version__).base_version < "2.9.0":
            logging.error("Async mode is only supported for torch >= 2.9.0, disabling async mode")
            self.is_async = False

class Checkpointer:
    """
    todo
    """
    def __init__(self, config: CheckpointingConfig, dp_rank: int, tp_rank: int, pp_rank: int, moe_mesh: Optional[DeviceMesh] = None):
        self.config = config
        self.moe_mesh = moe_mesh
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.__post_init__()
    
    def __post_init__(self):
        """
        Post-initialization hook.
        """
        self._addons = []
        if self._should_write_consolidated():
            self._addons.append(ConsolidatedHFAddon())
        if self.config.is_peft:
            self._addons.append(PeftAddon())
        self.inflight: Optional[Future] = None

    def save_model(
        self,
        model: nn.Module,
        weights_path: str,
        peft_config: Optional["PeftConfig"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ):
        """
        Save the checkpoint.
        """
        # Wait for any in-flight checkpoint (async case) to complete
        if self.inflight is not None:
            self.inflight.result()
            self.inflight = None
        
        # Create the model directories
        model_dir = os.path.join(weights_path, "model")
        consolidated_dir = os.path.join(model_dir, "consolidated") if self._should_write_consolidated() else None
        _ensure_dirs(model_dir, consolidated_dir)
        
        model_state = ModelState(model, self.config.is_peft)
        state_dict = model_state.state_dict()

        # Run pre-saves for addons e.g., PEFT or consolidated HF safetensors
        for addon in self._addons:
            addon.pre_save(
                model_state=model_state,
                model_path=model_dir,
                consolidated_path=consolidated_dir,
                tokenizer=tokenizer,
                peft_config=peft_config,
            )


        # Convert to HF format if using custom model implementations
        state_dict = _maybe_adapt_state_dict_to_hf(model_state.model[0], state_dict, quantization=False)
        # Build the consolidated model.safetensors.index.json if needed
        fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)
        
        storage_writer = self._get_storage_writer(consolidated_dir, fqn_to_file_index_mapping, model_dir)
        self._do_save(state_dict, model_dir, storage_writer)
    
    def save_optimizer(self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None):
        """
        Save the optimizer.
        """
        optimizer_path = os.path.join(weights_path, "optim")
        _ensure_dirs(optimizer_path)
        optimizer_state = OptimizerState(model, optimizer, scheduler)
        state_dict = optimizer_state.state_dict()
        self._do_save(state_dict, optimizer_path)
    
    def load_optimizer(self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None):
        """
        Load the optimizer.
        """
        optimizer_state = OptimizerState(model, optimizer, scheduler)
        state_dict = optimizer_state.state_dict()
        self._do_load(state_dict, os.path.join(weights_path, "optim"))

    
    def load_model(
        self,
        model: nn.Module,
        model_path: str,
        is_init_step: bool = False,
        use_checkpoint_id: bool = True,
        key_mapping: Optional[dict[str, str]] = None,
        quantization: bool = False,
    ) -> None:
        # Validate checkpoint directory
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        model_state = ModelState(model, is_peft=self.config.is_peft, is_init_step=is_init_step)
        state_dict = model_state.state_dict()
        storage_reader = self._get_storage_reader(model_path, key_mapping, is_init_step=is_init_step)

        state_dict = _maybe_adapt_state_dict_to_hf(model_state.model[0], state_dict, quantization=quantization)

        state_dict = self._do_load(state_dict, model_path, storage_reader, is_init_step=is_init_step)

        has_state_dict_adapter = hasattr(model_state.model[0], "state_dict_adapter")
        state_dict = _maybe_adapt_state_dict_from_hf(model_state.model[0], state_dict, moe_mesh=self.moe_mesh)
        model_state.load_state_dict(
            state_dict, strict=not (len(model_state.model) > 1 or has_state_dict_adapter)
        )
    
    def load_base_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        peft_init_method: str,
        load_base_model: bool = True,
        quantization: bool = False,
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

        if load_base_model:
            assert model_name is not None, "model_name is required when loading base model"
            self.load_model(
                model,
                model_path=model_name if os.path.exists(model_name) else get_safetensors_index_path(root_dir, model_name),
                is_init_step=True,
                key_mapping=getattr(model, "_checkpoint_conversion_mapping", None),
                quantization=quantization,
            )

        is_tied_lm_head = getattr(getattr(model, "config", {}), "tie_word_embeddings", False)
        if hasattr(model, "tie_weights") and is_tied_lm_head:
            model.tie_weights()

    def save_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """
        Save the stateful object.

        This function is a helper function currently used to save the dataloader and rng state.

        Args:
            state: Stateful object to save
            state_name: Name of the stateful object
            path: Path to save stateful object
        """
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir)
        if self.tp_rank == 0 and self.pp_rank == 0:
            torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"))
    
    def load_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """
        Load the stateful object.

        This function is a helper function currently used to load the dataloader and rng state.

        Args:
            state: Stateful object to load
            state_name: Name of the stateful object
            path: Path to load stateful object
        """
        state_dir = os.path.join(path, state_name)
        state.load_state_dict(torch.load(os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"), weights_only=False))
    
    def _do_load(self, state_dict: dict[str, torch.Tensor], path: str, storage_reader: Optional[_HuggingFaceStorageReader] = None, is_init_step: bool = False) -> dict[str, torch.Tensor]:
        # Both model and optimizer saving is done in this function
        is_model = True if "model" in path else False
        # PEFT loading is broadcasted from rank0 so it is a special case
        if self.config.is_peft and is_model and (not is_init_step):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
        else:
            dcp.load(state_dict, checkpoint_id=path, storage_reader=storage_reader)
        return state_dict
            
    
    def _do_save(self, state_dict: dict[str, torch.Tensor], path: str, storage_writer: Optional[_HuggingFaceStorageWriter] = None):
        # Both model and optimizer saving is done in this function
        is_model = True if "model" in path else False
        # PEFT saving is done on rank0 so it is a special case
        if self.config.is_peft and is_model:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
                return
        if self.config.is_async:
            # TODO: add process based async ckpt with pinned memory
            self.inflight = dcp.async_save(state_dict, checkpoint_id=path, storage_writer=storage_writer)
        else:
            dcp.save(state_dict, checkpoint_id=path, storage_writer=storage_writer)    

    def _should_write_consolidated(self) -> bool:
        return self.config.save_consolidated and self.config.model_save_format == SerializationFormat.SAFETENSORS and not self.config.is_peft
    
    def _maybe_build_consolidated_index(self, model_state: ModelState, state_dict: dict[str, torch.Tensor]) -> Optional[dict[str, int]]:
        if not self._should_write_consolidated():
            return None
        model = model_state.model[0]
        # we first need to find the FQN -> .safetensors mapping
        index_path = get_safetensors_index_path(
            self.config.model_cache_dir,
            self.config.model_repo_id,
        )
        if index_path:
            # HF VLM models may contain a special checkpoint mapping attribute
            fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                index_path, getattr(model, "_checkpoint_conversion_mapping", None)
            )
            # some HF models like Moonlight-16B have non-persistent buffers in the base checkpoint
            # however, HF initializes buffers with persistent=False, so we need to make sure these
            # buffer keys are not saved during checkpointing
            keys_to_remove = list(
                set(fqn_to_file_index_mapping.keys()) - set(self.config.model_state_dict_keys)
            )
            for key in keys_to_remove:
                fqn_to_file_index_mapping.pop(key)
        else:
            fqn_to_file_index_mapping = {k: 1 for k in state_dict.keys()}

        # Add any missing keys from the model_state_dict
        # These will go to the same file as the last file (or file 1 for single-file models)
        default_index = max(fqn_to_file_index_mapping.values())

        # add any additional keys that are not in the base checkpoint
        for fqn in list(state_dict.keys()):
            fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)
        return fqn_to_file_index_mapping
    
    def _get_storage_writer(
        self,
        consolidated_output_path: Optional[str],
        fqn_to_index_mapping: Optional[dict[str, int]],
        model_path: str,
    ) -> Optional[_HuggingFaceStorageWriter]:
        if self.config.model_save_format == SerializationFormat.SAFETENSORS:
            return _HuggingFaceStorageWriter(
                path=model_path,
                save_sharded=True,
                consolidated_output_path=consolidated_output_path,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )
    
    def _get_storage_reader(self, model_path: str, key_mapping: Optional[dict[str, str]], is_init_step: bool = False) -> Optional[_HuggingFaceStorageReader]:
        # If loading the model from the base checkpoint, we need to read the base model from the Hugging Face checkpoint
        if self.config.model_save_format == SerializationFormat.SAFETENSORS or is_init_step:
            return _HuggingFaceStorageReader(path=model_path, key_mapping=key_mapping)


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
    if os.path.exists(repo_id):
        return repo_id

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

def to_empty_parameters_only(
    model: nn.Module, *, device: torch.device, recurse: bool = True, dtype: torch.dtype | None = None
) -> nn.Module:
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
    return _apply(model, lambda t: torch.empty_like(t, device=device, dtype=dtype), recurse=recurse)

def save_config(config: dict[str, Any], weights_path: str):
    """
    Save a config to a weights path.

    Args:
        config: Config to save
        weights_path: Path to save config
    """
    with open(os.path.join(weights_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def _ensure_dirs(*dirs: Optional[str]) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for d in dirs:
            if d:
                os.makedirs(d, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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

def _maybe_adapt_state_dict_to_hf(model_part: nn.Module, state_dict: dict[str, torch.Tensor], quantization: bool = False) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to conver the state dict to the Hugging Face format.
    """
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        return adapter.to_hf(state_dict, exclude_key_regex=r".*_extra_state.*", quantization=quantization)
    return state_dict

def _maybe_adapt_state_dict_from_hf(model_part: nn.Module, state_dict: dict[str, torch.Tensor], moe_mesh: Optional[DeviceMesh] = None) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to convert the state dict from the Hugging Face format to the native format.
    """
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        ep_mesh_dims = [dim for dim in moe_mesh.mesh_dim_names if dim != "pp"] if moe_mesh is not None else []
        ep_mesh = moe_mesh[tuple(ep_mesh_dims)] if ep_mesh_dims else moe_mesh
        return adapter.from_hf(state_dict, device_mesh=ep_mesh)
    return state_dict