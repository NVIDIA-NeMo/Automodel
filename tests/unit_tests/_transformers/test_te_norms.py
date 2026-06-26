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

"""Unit tests for Transformer Engine norm replacement helpers."""

from __future__ import annotations

import sys
import types

import torch

from nemo_automodel._transformers.te_norms import replace_norms_with_te


class FakeTELayerNorm(torch.nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-5, device=None, params_dtype=None):
        super().__init__(hidden_size, eps=eps, device=device, dtype=params_dtype)
        self.hidden_size_arg = hidden_size
        self.params_dtype_arg = params_dtype


class FakeTERMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, params_dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=params_dtype))
        self.eps = eps
        self.hidden_size_arg = hidden_size
        self.params_dtype_arg = params_dtype


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps


class CustomRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.eps = 1e-6


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(4, eps=1e-4, dtype=torch.float32)
        self.layer_norm.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        self.layer_norm.bias.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        self.rms_norm = LlamaRMSNorm(3, eps=1e-5, dtype=torch.float32)
        self.rms_norm.weight.data.copy_(torch.tensor([5.0, 6.0, 7.0]))

        self.nested = torch.nn.ModuleDict(
            {
                "multi_dim_layer_norm": torch.nn.LayerNorm((2, 2)),
                "affine_disabled_layer_norm": torch.nn.LayerNorm(4, elementwise_affine=False),
                "custom_rms_norm": CustomRMSNorm(),
            }
        )


def _install_fake_te(monkeypatch):
    te_pkg = types.ModuleType("transformer_engine")
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.LayerNorm = FakeTELayerNorm
    te_pytorch.RMSNorm = FakeTERMSNorm
    te_pkg.pytorch = te_pytorch
    monkeypatch.setitem(sys.modules, "transformer_engine", te_pkg)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch)


def test_replace_norms_with_te_replaces_supported_norms_and_copies_state(monkeypatch):
    _install_fake_te(monkeypatch)
    module = TinyModule()

    layer_norm_count, rms_norm_count = replace_norms_with_te(module)

    assert (layer_norm_count, rms_norm_count) == (1, 1)

    assert isinstance(module.layer_norm, FakeTELayerNorm)
    assert module.layer_norm.hidden_size_arg == 4
    assert module.layer_norm.eps == 1e-4
    assert module.layer_norm.params_dtype_arg == torch.float32
    assert torch.equal(module.layer_norm.weight, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(module.layer_norm.bias, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    assert isinstance(module.rms_norm, FakeTERMSNorm)
    assert module.rms_norm.hidden_size_arg == 3
    assert module.rms_norm.eps == 1e-5
    assert module.rms_norm.params_dtype_arg == torch.float32
    assert torch.equal(module.rms_norm.weight, torch.tensor([5.0, 6.0, 7.0]))

    assert isinstance(module.nested["multi_dim_layer_norm"], torch.nn.LayerNorm)
    assert isinstance(module.nested["affine_disabled_layer_norm"], torch.nn.LayerNorm)
    assert isinstance(module.nested["custom_rms_norm"], CustomRMSNorm)


def test_replace_norms_with_te_reports_zero_when_no_supported_norms(monkeypatch):
    _install_fake_te(monkeypatch)
    module = torch.nn.ModuleDict(
        {
            "multi_dim_layer_norm": torch.nn.LayerNorm((2, 2)),
            "custom_rms_norm": CustomRMSNorm(),
        }
    )

    assert replace_norms_with_te(module) == (0, 0)
