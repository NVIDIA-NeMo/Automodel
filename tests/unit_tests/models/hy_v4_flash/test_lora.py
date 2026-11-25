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

"""LoRA on ``kv_b_proj``, whose weight the attention kernel absorbs into the queries and the values."""

import pytest
import torch
from torch.func import functional_call

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from tests.unit_tests.models.hy_v4_flash.conftest import TOKEN_RANGE, build_model

TARGET = "model.layers.*.self_attn.kv_b_proj"


def _lora_model(dropout: float) -> HYV4FlashForCausalLM:
    """The tiny model with an adapter on every ``kv_b_proj``; B is non-zero so the adapters change the output."""
    model = build_model()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=[TARGET], dim=4, alpha=8, dropout=dropout))
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LinearLoRA):
                module.lora_B.weight.copy_(0.05 * torch.randn(module.lora_B.weight.shape, generator=generator))
    return model


def _ids() -> torch.Tensor:
    return torch.randint(*TOKEN_RANGE, (1, 40), generator=torch.Generator().manual_seed(4))


def test_active_dropout_on_the_absorbed_projection_is_rejected() -> None:
    """One absorbed weight would share a single dropout mask between all tokens."""
    model = _lora_model(dropout=0.5)
    with pytest.raises(RuntimeError, match="cannot apply LoRA dropout"):
        model(_ids())
    model.eval()  # dropout is inactive in evaluation
    with torch.no_grad():
        assert torch.isfinite(model(_ids()).logits).all()


def test_adapter_without_dropout_trains_like_the_merged_weight() -> None:
    """Logits and adapter gradients equal those of the plain model run with ``W + scale * B @ A``."""
    model = _lora_model(dropout=0.0)
    ids = _ids()
    logits = model(ids).logits
    logits.square().sum().backward()

    plain = build_model()
    adapters, merged = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            a = module.lora_A.weight.detach().clone().requires_grad_(True)
            b = module.lora_B.weight.detach().clone().requires_grad_(True)
            adapters[name] = (a, b)
            merged[f"{name}.weight"] = module.weight.detach() + module.scale * (b @ a)
    assert len(adapters) == len(model.model.layers)
    expected = functional_call(plain, merged, (ids,)).logits
    expected.square().sum().backward()
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=2e-5)
    for name, (a, b) in adapters.items():
        module = model.get_submodule(name)
        assert b.grad.abs().sum() > 0 and a.grad.abs().sum() > 0
        torch.testing.assert_close(module.lora_A.weight.grad, a.grad, rtol=2e-3, atol=2e-4)
        torch.testing.assert_close(module.lora_B.weight.grad, b.grad, rtol=2e-3, atol=2e-4)
