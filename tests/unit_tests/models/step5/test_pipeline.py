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

"""Pipeline stage contracts for FP32 residuals and packed text."""

import copy

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.step5.model import Step5ForConditionalGeneration
from tests.unit_tests.models.step5.test_model import make_model


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the shared grouped-expert activation backward")
@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_pipeline_stage_forward_backward_parity(split: int, dtype: str, packed: bool) -> None:
    """Splitting stages must preserve logits, every gradient, and an SGD update."""
    reference = make_model(dtype=dtype, moe_layer_list=[1, 3])
    first, last = copy.deepcopy(reference), copy.deepcopy(reference)
    first.model.layers = nn.ModuleDict({k: v for k, v in first.model.layers.items() if int(k) < split})
    first.model.norm = None
    first.lm_head = None
    last.model.layers = nn.ModuleDict({k: v for k, v in last.model.layers.items() if int(k) >= split})
    last.model.embed_tokens = None

    tokens = torch.arange(24).reshape(2, 12)
    positions = torch.arange(12).expand(2, -1)
    documents = None
    if packed:
        positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]).expand(2, -1)
        documents = torch.tensor([[1] * 5 + [2] * 7]).expand(2, -1)
    kwargs = {"position_ids": positions, "_packed_seq_ids": documents}
    expected = reference(tokens, **kwargs).logits
    residual = first(tokens, **kwargs).logits
    assert residual.dtype == torch.float32
    actual = last(residual, **kwargs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    (expected * upstream).float().sum().backward()
    (actual * upstream).float().sum().backward()
    reference_parameters = dict(reference.named_parameters())
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            expected_parameter = reference_parameters[name]
            if expected_parameter.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0, msg=name)
    for model in (reference, first, last):
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0, msg=name)


def test_pipeline_metadata_and_fused_loss() -> None:
    """Intermediate buffers are FP32 while final fused-loss output stays BF16."""
    model = make_model(dtype="bfloat16")
    first, last = copy.deepcopy(model), copy.deepcopy(model)
    first.lm_head = None
    first.model.norm = None
    last.model.embed_tokens = None
    inputs, outputs = first.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    assert inputs[0].shape == (2, 12) and inputs[0].dtype == torch.long
    assert outputs[0].shape == (2, 12, 32) and outputs[0].dtype == torch.float32
    inputs, outputs = last.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    assert inputs[0].shape == (2, 12, 32) and inputs[0].dtype == torch.float32
    assert outputs[0].shape == (2, 12, 64) and outputs[0].dtype == torch.bfloat16
    last._pp_return_hidden_states = True
    _, outputs = last.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    hidden = last(torch.randn(2, 12, 32, dtype=torch.float32))
    assert outputs[0].shape == hidden.shape == (2, 12, 32)
    assert outputs[0].dtype == hidden.dtype == torch.bfloat16
    assert first._is_pipeline_parallel_stage()
    assert last._is_pipeline_parallel_stage()
    assert not model._is_pipeline_parallel_stage()
    assert model.ModelCapabilities.supports_pp
    assert not Step5ForConditionalGeneration.ModelCapabilities.supports_pp


def test_pipeline_rejects_rounded_residual() -> None:
    """A BF16 carry must fail instead of silently losing residual precision."""
    stage = make_model(dtype="bfloat16")
    stage.model.embed_tokens = None
    with pytest.raises(ValueError, match="require FP32 residuals"):
        stage(torch.randn(1, 12, 32, dtype=torch.bfloat16))
