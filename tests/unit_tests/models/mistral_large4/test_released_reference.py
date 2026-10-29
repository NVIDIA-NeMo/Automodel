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

"""Cross-check the bounded released-weight oracle against token/head equations."""

import pytest
import torch
from torch.nn import functional as F

from tests.unit_tests.models.mistral_large4.math_reference import reference_logits, rms
from tests.unit_tests.models.mistral_large4.released_reference import reference_block
from tests.unit_tests.models.mistral_large4.test_model import make_model


@pytest.mark.parametrize("layers", [4, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bounded_reference_matches_original_equations(layers: int, dtype: torch.dtype) -> None:
    overrides = (
        {}
        if layers == 4
        else dict(num_hidden_layers=40, nope_layer_flags=None, kv_source_layers=None, index_sharing=None)
    )
    model = make_model(dtype=dtype, seed=619, **overrides)
    original = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    a = {k: p.detach().clone().requires_grad_(p.requires_grad) for k, p in original.items()}
    b = {k: p.detach().clone().requires_grad_(p.requires_grad) for k, p in original.items()}
    ids = torch.tensor([1, 3, 7, 9, 4, 1, 12])
    positions = torch.arange(ids.numel()) * 157 + 2077
    expected = reference_logits(a, ids, model.config, positions, expert_batching="matrix")
    x = F.embedding(ids, b["tok_embeddings.weight"])
    cache, indices = {}, None
    for layer in range(layers):
        out = reference_block(
            b,
            x,
            positions,
            model.config,
            layer=layer,
            shared_kv=cache.get(model.config.kv_source_layers[layer]),
            indices=indices,
        )
        x, indices = out.hidden, out.indices
        cache[layer] = out.kv
    actual = 30 * torch.tanh(
        F.linear(rms(x, b["norm.weight"], model.config.rms_norm_eps), b["output.weight"]).float() / 30
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for name, parameter in a.items():
        if not parameter.requires_grad:
            assert parameter.grad is None and b[name].grad is None
        else:
            torch.testing.assert_close(b[name].grad, parameter.grad, atol=1.6e-5, rtol=4e-5, msg=name)


@pytest.mark.parametrize("offset", [0, 4093])
def test_bf16_rotary_uses_fp32_frequency_rounding(offset: int) -> None:
    """Exercise all 32 released rotary frequencies and learned-alpha gradients."""
    from nemo_automodel.components.models.mistral_large4.rope_utils import MistralLarge4Rotary, apply_interleaved_rope
    from tests.unit_tests.models.mistral_large4.math_reference import rotate, tiny_config
    from tests.unit_tests.models.mistral_large4.test_model import assert_bf16_error, tensor_error

    torch.manual_seed(1433)
    config = tiny_config(qk_nope_head_dim=64, qk_rope_head_dim=64, index_head_dim=128)
    rotary = MistralLarge4Rotary(config)
    with torch.no_grad():
        rotary.alpha.copy_(torch.linspace(-0.6, 0.6, 32))
    alpha = rotary.alpha.detach().clone().requires_grad_()
    x = torch.randn(5, 3, 64).bfloat16().requires_grad_()
    other = x.detach().clone().requires_grad_()
    positions = torch.arange(5) + offset
    actual = apply_interleaved_rope(x, rotary(positions))
    expected = rotate(other, positions, alpha, config, frequency_mode="float32")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    assert_bf16_error(tensor_error("alpha", rotary.alpha.grad, alpha.grad))
    assert_bf16_error(tensor_error("input", x.grad, other.grad))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_released_width_rms_rounding_and_backward(dtype: torch.dtype) -> None:
    """Pin normalization arithmetic before the BF16 boundary at released width."""
    from nemo_automodel.components.models.common.utils import Float32RMSNorm
    from tests.unit_tests.models.mistral_large4.test_model import assert_bf16_error, tensor_error

    torch.manual_seed(1511)
    norm = Float32RMSNorm(6144, eps=1e-5, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    weight = norm.weight.detach().clone().requires_grad_()
    x = torch.randn(5, 6144).to(dtype).requires_grad_()
    other = x.detach().clone().requires_grad_()
    actual, expected = norm(x), rms(other, weight, 1e-5)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    if dtype == torch.bfloat16:
        assert_bf16_error(tensor_error("input", x.grad, other.grad))
        assert_bf16_error(tensor_error("weight", norm.weight.grad, weight.grad))
    else:
        torch.testing.assert_close(x.grad, other.grad, atol=1.6e-5, rtol=4e-5)
        torch.testing.assert_close(norm.weight.grad, weight.grad, atol=1.6e-5, rtol=4e-5)
