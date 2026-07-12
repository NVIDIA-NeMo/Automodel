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

"""CPU coverage for activation checkpointing helpers."""

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper

from nemo_automodel.components.distributed import activation_checkpointing as ac
from nemo_automodel.components.models.common.utils import cast_model_to_dtype


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


def test_submodule_checkpointing_wraps_registered_child_for_property_aliases():
    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("e_score_correction_bias", torch.tensor([1.001, -2.003], dtype=torch.float32))

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate()

    class AliasLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = Mixer()

        @property
        def mlp(self):
            return self.mixer

        @property
        def self_attn(self):
            return self.mixer

    class Model(nn.Module):
        _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

        def __init__(self):
            super().__init__()
            self.layer = AliasLayer()
            self.proj = nn.Linear(2, 2)

    model = Model()
    original_bias = model.layer.mixer.gate.e_score_correction_bias.clone()

    ac.apply_submodule_checkpointing([model.layer], has_kv_sharing=False)

    assert isinstance(model.layer._modules["mixer"], CheckpointWrapper)
    assert "mlp" not in model.layer._modules
    assert "self_attn" not in model.layer._modules
    for name, _ in model.named_buffers(remove_duplicate=False):
        module_name, _, _ = name.rpartition(".")
        model.get_submodule(module_name)

    cast_model_to_dtype(model, torch.bfloat16)

    assert model.proj.weight.dtype == torch.bfloat16
    restored_bias = model.layer.mixer.gate.e_score_correction_bias
    assert restored_bias.dtype == torch.float32
    torch.testing.assert_close(restored_bias, original_bias)
