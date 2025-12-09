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


from threading import local

import torch

# inspired by https://github.com/zasdfgbnm/autonvtx

# Thread-local storage to track active NVTX ranges and prevent recursion
_thread_local = local()


def _get_active_ranges():
    """Get the set of currently active NVTX ranges for this thread."""
    if not hasattr(_thread_local, 'active_ranges'):
        _thread_local.active_ranges = set()
    return _thread_local.active_ranges


def _add_nvtx_hooks(model, name, add_backward_hooks=True):
    """Add NVTX range hooks to a model's forward and optionally backward passes."""
    if hasattr(model, '_nvtx_patched'):
        return
    
    def push_fwd(module, *args, **kwargs):
        if name in _get_active_ranges():
            module._nvtx_skipped = True
            return
        module._nvtx_skipped = False
        _get_active_ranges().add(name)
        torch.cuda.nvtx.range_push(name)
    
    def pop_fwd(module, *args, **kwargs):
        if getattr(module, '_nvtx_skipped', False):
            return
        torch.cuda.nvtx.range_pop()
        _get_active_ranges().discard(name)
    
    model.register_forward_pre_hook(push_fwd)
    model.register_forward_hook(pop_fwd)
    
    if add_backward_hooks:
        def push_bwd(module, grad_input):
            if name in _get_active_ranges():
                module._nvtx_skipped = True
                return
            module._nvtx_skipped = False
            _get_active_ranges().add(name)
            torch.cuda.nvtx.range_push(name)
        
        def pop_bwd(module, grad_input, grad_output):
            if getattr(module, '_nvtx_skipped', False):
                return
            torch.cuda.nvtx.range_pop()
            _get_active_ranges().discard(name)
        
        model.register_full_backward_pre_hook(push_bwd)
        model.register_full_backward_hook(pop_bwd)
    
    model._nvtx_patched = True


def patch(model, name=None, add_backward_hooks=True):
    """
    Recursively patch a model with NVTX profiling annotations.
    
    Prevents duplicate scopes when activation checkpointing reruns forward passes.
    """
    if hasattr(model, '_nvtx_patched'):
        return model
    
    name = type(model).__name__ if name is None else f"{name}: {type(model).__name__}"
    _add_nvtx_hooks(model, name, add_backward_hooks=add_backward_hooks)
    
    # Recursively patch all children
    for child_name, child in model.named_children():
        patch(child, child_name, add_backward_hooks)
    
    return model


# Export the functions properly
__all__ = ['patch']
