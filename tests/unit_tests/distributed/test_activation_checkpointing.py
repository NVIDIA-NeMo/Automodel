# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU coverage for the selective-AC save-set build (FFPA fold-in wiring)."""

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.distributed import activation_checkpointing as ac


def test_unwrap_checkpoint_wrapper_returns_input_module_when_unwrapped():
    module = nn.Linear(2, 2)

    assert ac.unwrap_checkpoint_wrapper(module) is module


def test_unwrap_checkpoint_wrapper_returns_inner_module_when_wrapped():
    module = nn.Linear(2, 2)
    wrapped = checkpoint_wrapper(module)

    assert ac.unwrap_checkpoint_wrapper(wrapped) is module


def test_ffpa_forward_ops_folded_into_save_set(monkeypatch):
    """Ops returned by _ffpa_forward_ops() land in the save set (kernel-free wiring)."""
    dense, varlen = object(), object()
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: (dense, varlen))

    save_ops = ac._build_selective_ac_save_ops()
    assert dense in save_ops
    assert varlen in save_ops


def test_build_save_set_ok_when_ffpa_absent(monkeypatch):
    """CPU degrade path: _ffpa_forward_ops() -> () must not break the build."""
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: ())

    save_ops = ac._build_selective_ac_save_ops()
    assert isinstance(save_ops, frozenset) and len(save_ops) > 0
