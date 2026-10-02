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

"""Real-TE regression coverage for checkpoint-free nested Qwen3 construction."""

import copy

import pytest
import torch
from transformers import Qwen3Config

from nemo_automodel._transformers.model_init import _apply_backend_module_overrides
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM
from nemo_automodel.shared.import_utils import safe_import

HAVE_TE, te = safe_import("transformer_engine.pytorch")

# Real TE kernels require CUDA; MXFP8 additionally requires Blackwell or newer.
pytestmark = pytest.mark.skipif(not HAVE_TE or not torch.cuda.is_available(), reason="requires TE and CUDA")


@pytest.mark.parametrize("copy_before_init", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
def test_te_linear_meta_init_and_backward(tied: bool, quantized: bool, copy_before_init: bool) -> None:
    """Preserve parameters, initialize nested projections, and execute real kernels."""
    if quantized and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 kernels require Blackwell or newer")
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import autocast

    torch.manual_seed(42)
    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=tied,
        initializer_range=0.03,
        pad_token_id=None,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    backend = BackendConfig(attn="sdpa", linear="te", rms_norm="torch", rope_fusion=False)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config, backend=backend)
    original_parameters = dict(model.named_parameters())
    original_keys = set(model.state_dict())
    projection_count = sum(type(module) is torch.nn.Linear for module in model.modules())
    model.eval()

    _apply_backend_module_overrides(model, backend)

    linears = [module for module in model.modules() if isinstance(module, te.Linear)]
    assert len(linears) == projection_count == 15
    assert not any(type(module) is torch.nn.Linear for module in model.modules())
    assert all(not module.training for module in linears)
    assert all(dict(model.named_parameters())[name] is value for name, value in original_parameters.items())
    assert {key for key in model.state_dict() if not key.endswith("_extra_state")} == original_keys

    if copy_before_init:
        source = model
        model = copy.deepcopy(source)
        linears = [module for module in model.modules() if isinstance(module, te.Linear)]
        assert all(parameter.is_meta for parameter in source.parameters())

    # Do not load any checkpoint: exercise the production materialize/init path.
    Checkpointer.initialize_model_weights(model, device=torch.device("cuda"))
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied
    if copy_before_init:
        assert all(parameter.is_meta for parameter in source.parameters())
    for module in linears:
        assert torch.isfinite(module.weight).all()
        # Many independent samples per matrix; a 10% band catches skipped/default
        # initialization without requiring identical RNG draw order across backends.
        assert abs(module.weight.float().std().item() - config.initializer_range) < 0.003
        assert abs(module.weight.float().mean().item()) < 0.003

    model = model.to(dtype=torch.bfloat16).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
    labels = torch.randint(0, config.vocab_size, (2, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16), autocast(enabled=quantized, recipe=MXFP8BlockScaling()):
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, config.vocab_size), labels.reshape(-1))
    loss.backward()
    assert logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", ["cuda", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_te_compatible_linear_deepcopy_forward_backward_parity(
    device: str, dtype: torch.dtype, bias: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check the real subtype and TE kernels against an ordinary torch Linear."""
    from nemo_automodel.components.models.common.te_linear import TELinear

    # Compare full fp32 GEMMs rather than torch's optional TF32 approximation.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    model = torch.nn.Sequential(torch.nn.Linear(32, 64, bias=bias, device=device, dtype=dtype))
    _apply_backend_module_overrides(model, BackendConfig(linear="te"))
    layer = copy.deepcopy(model)[0]
    assert isinstance(layer, TELinear)
    assert isinstance(layer, torch.nn.Linear)
    assert isinstance(layer, te.Linear)
    layer.to_empty(device="cuda")
    reference = torch.nn.Linear(32, 64, bias=bias, device="cuda", dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(reference.weight)
        if bias:
            layer.bias.copy_(reference.bias)
    inputs = torch.randn(16, 32, device="cuda", dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = layer(inputs)
    expected = reference(reference_inputs)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    # TE's fp32 GEMMs use reduced-precision products independently of torch's
    # TF32 setting; bf16 also rounds bias/gradient reductions differently. Check
    # both against torch within product/reduction error, then native TE exactly.
    tolerance = dict(rtol=0.02, atol=0.02) if dtype == torch.bfloat16 else dict(rtol=0.002, atol=0.01)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, **tolerance)
    torch.testing.assert_close(layer.weight.grad, reference.weight.grad, **tolerance)
    if bias:
        torch.testing.assert_close(layer.bias.grad, reference.bias.grad, **tolerance)

    native = te.Linear(32, 64, bias=bias, device="cuda", params_dtype=dtype)
    with torch.no_grad():
        native.weight.copy_(layer.weight)
        if bias:
            native.bias.copy_(layer.bias)
    native_inputs = inputs.detach().clone().requires_grad_()
    native_output = native(native_inputs)
    native_output.backward(upstream)
    torch.testing.assert_close(actual, native_output, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, native_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, native.weight.grad, rtol=0, atol=0)
    if bias:
        torch.testing.assert_close(layer.bias.grad, native.bias.grad, rtol=0, atol=0)
