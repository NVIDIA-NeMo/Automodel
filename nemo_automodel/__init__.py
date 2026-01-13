# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import functools
import importlib
import inspect
import logging

from torch.utils.data import _utils as torch_data_utils

# Monkey patch pin_memory to optionally accept a device argument.
# The device argument was removed in some newer torch versions but we
# need it for compatibility with torchdata.
_original_pin_memory_loop = torch_data_utils.pin_memory._pin_memory_loop
_original_pin_memory = torch_data_utils.pin_memory.pin_memory
_original_pin_memory_sig = inspect.signature(_original_pin_memory)

if "device" not in _original_pin_memory_sig.parameters:

    @functools.wraps(_original_pin_memory)
    def _patched_pin_memory(data, device=None):
        """Patched pin_memory that accepts an optional device argument."""
        return _original_pin_memory(data)

    @functools.wraps(_original_pin_memory_loop)
    def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
        """Patched _pin_memory_loop to accept a device argument."""
        return _original_pin_memory_loop(in_queue, out_queue, device_id, done_event)

    torch_data_utils.pin_memory.pin_memory = _patched_pin_memory
    torch_data_utils.pin_memory._pin_memory_loop = _pin_memory_loop


# Monkey patch DeviceMesh to fix corner case in mesh slicing
# Fixes issue where _dim_group_names is accessed without checking if rank is in mesh
# Based on https://github.com/pytorch/pytorch/pull/169454/files
try:
    import torch as _torch

    # Only apply the patch for the specific PyTorch version with the regression
    # TODO: Remove this once bump up to a newer PyTorch version with the fix
    if "2.10.0" in _torch.__version__ and "nv25.11" in _torch.__version__:
        from torch.distributed._mesh_layout import _MeshLayout
        from torch.distributed.device_mesh import _MeshEnv

        _original_get_slice_mesh_layout = _MeshEnv._get_slice_mesh_layout

        def _patched_get_slice_mesh_layout(self, device_mesh, mesh_dim_names):
            """
            Patched _get_slice_mesh_layout based on PyTorch PR #169454.
            This fixes:
            1. _dim_group_names access (commit f6c8092)
            2. Regression in mesh slicing with size-1 dims (PR #169454 / Issue #169381)
            """
            # 1. First, build the layout manually to bypass the legacy 'stride < pre_stride' check
            slice_from_root = device_mesh == self.get_root_mesh(device_mesh)
            flatten_name_to_root_layout = (
                {key: mesh._layout for key, mesh in self.root_to_flatten_mapping.setdefault(device_mesh, {}).items()}
                if slice_from_root
                else {}
            )

            mesh_dim_names_list = getattr(device_mesh, "mesh_dim_names", [])
            valid_mesh_dim_names = [*mesh_dim_names_list, *flatten_name_to_root_layout]
            if not all(name in valid_mesh_dim_names for name in mesh_dim_names):
                raise KeyError(f"Invalid mesh_dim_names {mesh_dim_names}. Valid: {valid_mesh_dim_names}")

            layout_sliced = []
            for name in mesh_dim_names:
                if name in mesh_dim_names_list:
                    layout_sliced.append(device_mesh._layout[mesh_dim_names_list.index(name)])
                elif name in flatten_name_to_root_layout:
                    layout_sliced.append(flatten_name_to_root_layout[name])

            sliced_sizes = tuple(layout.sizes for layout in layout_sliced)
            sliced_strides = tuple(layout.strides for layout in layout_sliced)

            # Bypass the 'stride < pre_stride' check that exists in the original
            # and create the MeshLayout directly.
            slice_mesh_layout = _MeshLayout(sliced_sizes, sliced_strides)

            if not slice_mesh_layout.check_non_overlap():
                raise RuntimeError(f"Slicing overlapping dim_names {mesh_dim_names} is not allowed.")

            # 2. Replicate the _dim_group_names fix (commit f6c8092)
            # We need to return an object that HAS _dim_group_names if the rank is in the mesh
            if hasattr(device_mesh, "_dim_group_names") and len(device_mesh._dim_group_names) > 0:
                slice_dim_group_name = []
                submesh_dim_names = mesh_dim_names if isinstance(mesh_dim_names, tuple) else (mesh_dim_names,)
                for name in submesh_dim_names:
                    if name in mesh_dim_names_list:
                        slice_dim_group_name.append(device_mesh._dim_group_names[mesh_dim_names_list.index(name)])
                    elif hasattr(device_mesh, "_flatten_mapping") and name in device_mesh._flatten_mapping:
                        flatten_mesh = device_mesh._flatten_mapping[name]
                        slice_dim_group_name.append(
                            flatten_mesh._dim_group_names[flatten_mesh.mesh_dim_names.index(name)]
                        )

                # Attach the group names to the layout object so the caller can use them
                object.__setattr__(slice_mesh_layout, "_dim_group_names", slice_dim_group_name)

            return slice_mesh_layout

        # Apply the patch
        _MeshEnv._get_slice_mesh_layout = _patched_get_slice_mesh_layout
        logging.getLogger(__name__).debug(f"Applied DeviceMesh fix for PyTorch {_torch.__version__}")

except (ImportError, AttributeError) as e:
    logging.getLogger(__name__).debug(f"Could not apply DeviceMesh patch: {e}")
    pass


from .package_info import __package_name__, __version__

__all__ = [
    "recipes",
    "shared",
    "components",
    "__version__",
    "__package_name__",
]

# Promote NeMoAutoModelForCausalLM, AutoModelForImageTextToText into the top level
# to enable: `from nemo_automodel import NeMoAutoModelForCausalLM`
try:
    # adjust this import path if your class lives somewhere else
    from nemo_automodel._transformers.auto_model import (
        NeMoAutoModelForCausalLM,
        NeMoAutoModelForImageTextToText,
        NeMoAutoModelForSequenceClassification,
        NeMoAutoModelForTextToWaveform,
    )  # noqa: I001
    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

    globals()["NeMoAutoModelForCausalLM"] = NeMoAutoModelForCausalLM
    globals()["NeMoAutoModelForImageTextToText"] = NeMoAutoModelForImageTextToText
    globals()["NeMoAutoModelForSequenceClassification"] = NeMoAutoModelForSequenceClassification
    globals()["NeMoAutoModelForTextToWaveform"] = NeMoAutoModelForTextToWaveform
    globals()["NeMoAutoTokenizer"] = NeMoAutoTokenizer
    __all__.append("NeMoAutoModelForCausalLM")
    __all__.append("NeMoAutoModelForImageTextToText")
    __all__.append("NeMoAutoModelForSequenceClassification")
    __all__.append("NeMoAutoModelForTextToWaveform")
    __all__.append("NeMoAutoTokenizer")
except:
    # optional dependency might be missing,
    # leave the name off the module namespace so other imports still work
    pass


def __getattr__(name: str):
    """
    Lazily import and cache submodules listed in __all__ when accessed.

    Raises:
        AttributeError if the name isn’t in __all__.
    """
    if name in __all__:
        # import submodule on first access
        module = importlib.import_module(f"{__name__}.{name}")
        # cache it in globals() so future lookups do not re-import
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
