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

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def _recipe(device="cpu", *, peft=True, explicit=None):
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.__dict__.update(
        cfg=SimpleNamespace(peft=SimpleNamespace() if peft else None, mtp=SimpleNamespace(scaling_factor=None)),
        distributed_config=SimpleNamespace(
            autocast_dtype=explicit, mp_policy=SimpleNamespace(param_dtype=torch.bfloat16)
        ),
        dist_env=SimpleNamespace(device=torch.device(device)),
        pp_enabled=False,
        device_mesh=None,
        tokenizer=None,
        domain_mixture=None,
        te_fp8=None,
        loss_fn=MaskedCrossEntropy(),
        _get_cp_group_size=lambda: 1,
        _get_dp_group_size=lambda **kw: 1,
    )
    return recipe


@pytest.mark.parametrize(
    "peft,explicit,expected",
    [
        (True, None, torch.bfloat16),
        (True, torch.float32, torch.float32),
        (True, torch.float16, torch.float16),
        (False, None, torch.float32),
        (False, torch.bfloat16, torch.bfloat16),
    ],
)
def test_autocast_policy(peft, explicit, expected):
    recipe = _recipe(peft=peft, explicit=explicit)
    linear = nn.Linear(4, 4)
    with recipe._autocast_context():
        output = linear(torch.ones(2, 4))
    assert output.dtype == expected
    assert linear.weight.dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")


class _ToyLM(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embed = nn.Embedding(16, 8, dtype=torch.bfloat16, device=device)
        self.embed.requires_grad_(False)
        self.proj = LinearLoRA(nn.Linear(8, 16, bias=False, dtype=torch.bfloat16, device=device), dim=4)
        nn.init.normal_(self.proj.lora_B.weight, std=0.02)

    def forward(self, input_ids):
        """Map tokens [batch, sequence] to logits [batch, sequence, vocab]."""
        assert torch.is_autocast_enabled(input_ids.device.type)
        output = self.proj(self.embed(input_ids))
        if output.requires_grad:

            def check_backward(grad):
                """Check autocast is disabled for logits gradients [batch, sequence, vocab]."""
                assert not torch.is_autocast_enabled(grad.device.type)

            output.register_hook(check_backward)
        assert output.dtype == torch.bfloat16
        return output


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.parametrize("is_train", [False, True])
def test_llm_recipe_mixed_storage_step(device, is_train):
    recipe = _recipe(device)
    model = _ToyLM(device)
    reference = copy.deepcopy(model)
    recipe.__dict__["model_parts"] = [model]
    inputs = torch.arange(8, device=device).reshape(2, 4)
    labels = inputs.roll(-1, -1)
    with torch.autocast(device, dtype=torch.bfloat16):
        expected = recipe.loss_fn(reference(inputs), labels, num_label_tokens=labels.numel())
    if is_train:
        expected.backward()
    losses = []
    recipe._forward_backward_step(
        0,
        {"input_ids": inputs, "labels": labels},
        loss_buffer=losses,
        num_label_tokens=labels.numel(),
        num_batches=1,
        is_train=is_train,
    )
    torch.testing.assert_close(losses[0], expected)
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        if is_train and parameter.requires_grad:
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            torch.testing.assert_close(parameter.grad, ref_parameter.grad)
        else:
            assert parameter.grad is None
    if is_train:
        optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
        optimizer.step()
        assert all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32 for s in optimizer.state.values())


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("module_loss", [False, True])
def test_pipeline_autocast_scopes_and_restoration(fail, module_loss):
    recipe = _recipe()
    model = _ToyLM("cpu")
    recipe.__dict__["model_parts"] = [model]

    class Loss(nn.Module):
        def forward(self, output, target):
            """Compare logits and targets, both [batch, sequence, vocab]."""
            assert torch.is_autocast_enabled("cpu")
            return (output.float() - target).square().mean()

    loss = Loss()
    original_loss = loss if module_loss else loss.forward
    schedule = SimpleNamespace(_loss_fn=original_loss)
    recipe.__dict__["pp"] = SimpleNamespace(parts=[model], info=SimpleNamespace(schedule=schedule))
    original_forward = model.forward
    try:
        with recipe._pipeline_autocast_context(recipe.pp):
            if module_loss:
                assert schedule._loss_fn is loss
            for _ in range(2):
                output = model(torch.arange(8).reshape(2, 4))
                schedule._loss_fn(output, torch.zeros_like(output)).backward()
                assert not torch.is_autocast_enabled("cpu")
            if fail:
                raise ValueError("schedule failure")
    except ValueError:
        assert fail
    assert model.forward == original_forward
    assert "forward" not in model.__dict__
    assert "forward" not in loss.__dict__
    assert schedule._loss_fn is original_loss
    assert not torch.is_autocast_enabled("cpu")
