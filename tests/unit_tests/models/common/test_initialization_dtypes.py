# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Initialization owns temporary casts without persistent model metadata."""

import contextlib

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, yield_fp32_model


class MixedStorageModel(nn.Module):
    def __init__(self, device="cpu", reverse=False):
        super().__init__()
        layers = {
            "base": nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16).requires_grad_(False),
            "adapter": nn.Linear(4, 4, bias=False, device=device, dtype=torch.float32),
        }
        for name in reversed(layers) if reverse else layers:
            self.add_module(name, layers[name])
        self.register_buffer("scale", torch.ones(4, device=device, dtype=torch.float32))

    @torch.no_grad()
    def initialize_weights(self, dtype=torch.bfloat16):
        self.base.weight.fill_(0.25)
        self.adapter.weight.fill_(1.001)
        self.scale.fill_(1.001)
        cast_model_to_dtype(self, dtype)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("reverse", [False, True])
def test_checkpointer_preserves_storage_independently_of_parameter_order(device, reverse):
    model = MixedStorageModel(device=device, reverse=reverse)
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert model.base.weight.dtype == torch.bfloat16
    assert model.adapter.weight.dtype == torch.float32
    assert model.scale.dtype == torch.float32
    torch.testing.assert_close(model.base.weight, torch.full((4, 4), 0.25, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(model.adapter.weight, torch.full((4, 4), 1.001), rtol=0, atol=0)
    torch.testing.assert_close(model.scale, torch.full((4,), 1.001), rtol=0, atol=0)


@pytest.mark.parametrize("raise_in_body", [False, True])
def test_fp32_context_restores_mixed_parameters_buffers_aliases_and_gradients(raise_in_body):
    model = MixedStorageModel()
    model.register_parameter("tied", model.base.weight)
    model.register_buffer("low_scale", torch.ones(4, dtype=torch.bfloat16))
    model.register_buffer("low_scale_alias", model.low_scale)
    model.adapter.weight.grad = torch.full_like(model.adapter.weight, 1.001)
    original_parameters = dict(model.named_parameters())
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    expected_grad = model.adapter.weight.grad.clone()
    failure = pytest.raises(RuntimeError, match="initialization failed") if raise_in_body else contextlib.nullcontext()
    with failure:
        with yield_fp32_model(model):
            assert all(p.dtype == torch.float32 for p in model.parameters())
            assert all(b.dtype == torch.float32 for b in model.buffers())
            if raise_in_body:
                raise RuntimeError("initialization failed")
    assert model.tied is model.base.weight
    assert all(p is original_parameters[name] for name, p in model.named_parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    torch.testing.assert_close(model.adapter.weight.grad, expected_grad, rtol=0, atol=0)


def test_explicit_dtype_still_casts_after_fp32_initialization():
    model = MixedStorageModel()
    with yield_fp32_model(model, torch.bfloat16):
        assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.scale.dtype == torch.bfloat16
