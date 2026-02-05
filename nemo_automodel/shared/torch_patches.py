# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
"""
Torch compatibility patches.

These patches are intentionally NOT applied at `import nemo_automodel` time to keep
tokenizer-only imports lightweight. Call `apply_torch_patches()` from code paths
that already depend on torch (training / distributed / dataloading).
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_TORCH_PATCHES_APPLIED = False


def apply_torch_patches() -> None:
    """
    Apply small, version/packaging-specific torch monkey patches.

    This function is idempotent and safe to call multiple times.
    """
    global _TORCH_PATCHES_APPLIED
    if _TORCH_PATCHES_APPLIED:
        return

    try:
        import torch as _torch
    except Exception:
        # torch not installed or failing to import: nothing to patch.
        return

    # -------------------------------------------------------------------------
    # Patch #1: torchdata compatibility
    # Monkey patch pin_memory to optionally accept a device argument.
    # The device argument was removed in some newer torch versions but torchdata
    # still passes it in some versions.
    # -------------------------------------------------------------------------
    try:
        import functools
        import inspect

        from torch.utils.data import _utils as torch_data_utils

        _original_pin_memory_loop = torch_data_utils.pin_memory._pin_memory_loop
        _original_pin_memory = torch_data_utils.pin_memory.pin_memory
        _original_pin_memory_sig = inspect.signature(_original_pin_memory)

        if "device" not in _original_pin_memory_sig.parameters:

            @functools.wraps(_original_pin_memory)
            def _patched_pin_memory(data, device=None):
                return _original_pin_memory(data)

            @functools.wraps(_original_pin_memory_loop)
            def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
                return _original_pin_memory_loop(in_queue, out_queue, device_id, done_event)

            torch_data_utils.pin_memory.pin_memory = _patched_pin_memory
            torch_data_utils.pin_memory._pin_memory_loop = _pin_memory_loop

    except Exception as e:
        _logger.debug(f"Could not apply torch pin_memory patch: {e}")

    # -------------------------------------------------------------------------
    # Patch #2: DeviceMesh slicing corner case (specific PyTorch regression)
    # Fixes issue where _dim_group_names is accessed without checking if rank is in mesh.
    # Based on https://github.com/pytorch/pytorch/pull/169454/files
    # -------------------------------------------------------------------------
    try:
        # Only apply the patch for the specific PyTorch version with the regression
        # TODO: Remove this once bump up to a newer PyTorch version with the fix
        if "2.10.0" in _torch.__version__ and "nv25.11" in _torch.__version__:
            from torch.distributed._mesh_layout import _MeshLayout
            from torch.distributed.device_mesh import _MeshEnv

            def _patched_get_slice_mesh_layout(self, device_mesh, mesh_dim_names):
                # 1. Build the layout manually to bypass the legacy 'stride < pre_stride' check
                slice_from_root = device_mesh == self.get_root_mesh(device_mesh)
                flatten_name_to_root_layout = (
                    {
                        key: mesh._layout
                        for key, mesh in self.root_to_flatten_mapping.setdefault(device_mesh, {}).items()
                    }
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

                # Bypass the 'stride < pre_stride' check that exists in the original and create MeshLayout directly.
                slice_mesh_layout = _MeshLayout(sliced_sizes, sliced_strides)

                if not slice_mesh_layout.check_non_overlap():
                    raise RuntimeError(f"Slicing overlapping dim_names {mesh_dim_names} is not allowed.")

                # 2. Replicate the _dim_group_names fix (commit f6c8092)
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

                    object.__setattr__(slice_mesh_layout, "_dim_group_names", slice_dim_group_name)

                return slice_mesh_layout

            _MeshEnv._get_slice_mesh_layout = _patched_get_slice_mesh_layout
            _logger.debug(f"Applied DeviceMesh fix for PyTorch {_torch.__version__}")

    except (ImportError, AttributeError) as e:
        _logger.debug(f"Could not apply DeviceMesh patch: {e}")

    _TORCH_PATCHES_APPLIED = True
