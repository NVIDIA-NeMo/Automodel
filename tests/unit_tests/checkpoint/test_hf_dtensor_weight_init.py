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

"""Initialize real HF models after their meta parameters have been sharded."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers import FalconH1Config, FalconH1ForCausalLM

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer


@pytest.fixture
def cpu_mesh():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("placement", [Shard(0), Replicate()])
def test_falcon_h1_initializes_sharded_meta_weights(cpu_mesh, placement):
    config = FalconH1Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        mamba_d_ssm=16,
        mamba_n_heads=4,
        mamba_d_head=4,
        mamba_d_state=4,
        mamba_n_groups=1,
        mamba_chunk_size=4,
        pad_token_id=0,
        ssm_multipliers=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    reference = FalconH1ForCausalLM(config)
    with torch.device("meta"):
        model = FalconH1ForCausalLM(config)
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            module.register_parameter(
                name, torch.nn.Parameter(distribute_tensor(parameter.detach(), cpu_mesh, [placement]))
            )

    Checkpointer.initialize_model_weights(model, torch.device("cpu"))

    for parameter in model.parameters():
        assert isinstance(parameter, DTensor)
        assert parameter.placements == (placement,)
        assert torch.isfinite(parameter.to_local()).all()
    mixer = model.model.layers[0].mamba
    expected_mixer = reference.model.layers[0].mamba
    for name in ("A_log", "dt_bias", "D"):
        torch.testing.assert_close(getattr(mixer, name).full_tensor(), getattr(expected_mixer, name), rtol=0, atol=0)
    torch.testing.assert_close(mixer.mup_vector, expected_mixer.mup_vector, rtol=0, atol=0)
    assert model.model.embed_tokens.padding_idx == 0
    assert model.model.embed_tokens.weight.full_tensor()[0].count_nonzero() == 0
    assert model.model.embed_tokens.weight.full_tensor()[1:].count_nonzero() > 0


@pytest.mark.parametrize("placement", [Shard(0), Shard(1), Replicate()])
@pytest.mark.parametrize("keyword_source", [False, True])
def test_initialization_copy_preserves_broadcast_dtype_and_parameter(cpu_mesh, placement, keyword_source):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(architectures=["TestModel"])
            self.weight = torch.nn.Parameter(
                distribute_tensor(torch.zeros(5, 3, dtype=torch.bfloat16), cpu_mesh, [placement])
            )

        @torch.no_grad()
        def initialize_weights(self):
            source = torch.tensor([1.0, 2.0, 3.0])
            if keyword_source:
                result = self.weight.copy_(other=source)
            else:
                result = self.weight.copy_(source)
            assert result is self.weight
            torch.testing.assert_close(source, torch.tensor([1.0, 2.0, 3.0]))

    model = Model()
    parameter = model.weight
    local_storage = parameter.to_local().data_ptr()
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert model.weight is parameter
    assert model.weight.to_local().data_ptr() == local_storage
    torch.testing.assert_close(
        model.weight.full_tensor(), torch.tensor([[1.0, 2.0, 3.0]] * 5, dtype=torch.bfloat16), rtol=0, atol=0
    )
    # The adaptation is scoped to initialization, not a process-wide change to copy_.
    with torch.no_grad(), pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        model.weight.copy_(torch.zeros(5, 3))


def test_initialization_copy_error_restores_dispatch(cpu_mesh):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(distribute_tensor(torch.zeros(3), cpu_mesh, [Shard(0)]))

        @torch.no_grad()
        def initialize_weights(self):
            self.weight.copy_(torch.zeros(2))

    model = Model()
    with pytest.raises(RuntimeError, match="expanded size"):
        Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    with torch.no_grad(), pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        model.weight.copy_(torch.zeros(3))


@pytest.mark.parametrize("custom_code", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_fsdp_initialization_uses_original_code_classification(cpu_mesh, custom_code, fail):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("initialized", torch.zeros(1), persistent=False)

        @classmethod
        def is_custom_code(cls):
            # HF treats classes generated outside its package as custom code.
            return custom_code or cls.__module__ != __name__

        @torch.no_grad()
        def initialize_weights(self, dtype=None):
            self.initialized.fill_(2 if self.is_custom_code() else 1)
            if fail:
                raise ValueError("initializer failed")

    model = fully_shard(Model(), mesh=cpu_mesh)
    assert model.is_custom_code()
    if fail:
        with pytest.raises(ValueError, match="initializer failed"):
            Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    else:
        Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert model.initialized.item() == (2 if custom_code else 1)
    assert "is_custom_code" not in vars(model)
    assert model.is_custom_code()


def test_fsdp_initialization_preserves_instance_code_classification(cpu_mesh):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        @classmethod
        def is_custom_code(cls):
            return True

        @torch.no_grad()
        def initialize_weights(self, dtype=None):
            self.weight.fill_(2 if self.is_custom_code() else 1)

    model = fully_shard(Model(), mesh=cpu_mesh)
    override = lambda: False
    model.is_custom_code = override
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert model.is_custom_code is override
    torch.testing.assert_close(model.weight.full_tensor(), torch.ones(2), rtol=0, atol=0)
