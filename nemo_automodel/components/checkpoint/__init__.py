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
import importlib as _importlib

from ._torch_backports import apply_async_checkpoint_patch as _nemo__apply_async_patch
from ._torch_backports import apply_patches as _nemo__apply_patches
from .config import CheckpointingConfig, _is_geq_torch_2_9, _is_leq_torch_2_7_1

__all__ = ["CheckpointingConfig"]

if _is_leq_torch_2_7_1():
    _nemo__apply_patches()

if _is_geq_torch_2_9():
    _nemo__apply_async_patch()

_LAZY_ATTRS = {
    "Checkpointer": (".checkpointing", "Checkpointer"),
    "StateDictAdapter": (".state_dict_adapter", "StateDictAdapter"),
    "find_latest_checkpoint": (".utils", "find_latest_checkpoint"),
    "get_checkpoint_tensor_dtypes": (".utils", "_get_checkpoint_tensor_dtypes"),
    "load_full_state_dict_into_model": (".checkpointing", "_load_full_state_dict_into_model"),
    "load_hf_checkpoint_preserving_dtype": (".checkpointing", "_load_hf_checkpoint_preserving_dtype"),
    "load_hf_safetensors_state_dict": (".checkpointing", "load_hf_safetensors_state_dict"),
    "load_torch_ckpt": (".checkpointing", "load_torch_ckpt"),
    "maybe_adapt_state_dict_to_hf": (".checkpointing", "_maybe_adapt_state_dict_to_hf"),
    "resolve_restore_from_to_checkpoint_dir": (".utils", "resolve_restore_from_to_checkpoint_dir"),
    "save_config": (".checkpointing", "save_config"),
    "save_generated_hf_assets": (".addons", "_save_generated_hf_assets"),
    "save_losses": (".checkpointing", "save_losses"),
}

__all__ += sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
