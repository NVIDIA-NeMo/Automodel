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

"""Independent mHC equations and RMSNorm backend construction/checkpoint contracts."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41HyperConnection,
    DeepseekV41Mix,
    DeepseekV41RMSNorm,
)
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.test_attention import _arithmetic_config
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config


def test_mhc_combination_orientation_and_explicit_predecessor_mix() -> None:
    streams = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    previous_pre = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    current_pre = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    mix = DeepseekV41Mix(
        current_pre,
        torch.tensor([[[2.0, 3.0]]]),
        torch.tensor([[[[0.1, 0.9], [0.6, 0.4]]]]),
    )
    collapsed = DeepseekV41HyperConnection.collapse(streams, previous_pre)
    torch.testing.assert_close(collapsed, torch.tensor([[[1.0, 2.0]]]))
    result = DeepseekV41HyperConnection.expand(collapsed, streams, mix)
    # Residual output0 = .1*stream0 + .6*stream1; output1 = .9*stream0 + .4*stream1.
    expected = torch.tensor([[[[3.9, 6.6], [5.1, 9.4]]]])
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert previous_pre.grad is not None
    assert current_pre.grad is None  # It belongs to the following sublayer's input.


def test_mhc_coefficients_and_gradients_are_finite() -> None:
    torch.manual_seed(42)
    module = DeepseekV41HyperConnection(_tiny_config().text_config)
    streams = torch.randn(2, 3, 4, 16, requires_grad=True)
    mix = module(streams)
    assert torch.all(mix.pre > 0)
    assert torch.all((mix.post > 0) & (mix.post < 2))
    torch.testing.assert_close(mix.comb.sum(-1), torch.ones(2, 3, 4), atol=1e-4, rtol=0)
    torch.testing.assert_close(mix.comb.sum(-2), torch.ones(2, 3, 4), atol=1e-4, rtol=0)
    (mix.pre.square().sum() + mix.post.square().sum() + mix.comb.square().sum()).backward()
    assert torch.isfinite(streams.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def _mhc_reference(x, parameters, streams, norm_eps, hc_eps, repeat):
    flat = x.flatten(2).float()
    projected = F.linear(flat, parameters["fn"]) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    scales = parameters["scale"].repeat_interleave(torch.tensor([streams, streams, streams * streams]))
    logits = torch.addcmul(parameters["base"], projected, scales)
    pre = logits[..., :streams].sigmoid() + hc_eps
    post = 2 * logits[..., streams : 2 * streams].sigmoid()
    comb = logits[..., 2 * streams :].reshape(*x.shape[:2], streams, streams).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mhc_coefficients_collapse_expand_and_gradients_match_released_equations(dtype):
    torch.manual_seed(43)
    module = DeepseekV41HyperConnection(_arithmetic_config(hc_eps=1e-6), sinkhorn_backend="torch")
    with torch.no_grad():
        module.fn.normal_(std=0.1)
        module.base.normal_(std=0.3)
        module.scale.copy_(torch.tensor([0.9, 1.1, 0.7]))
    reference_parameters = {name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    x = torch.randn(2, 3, 4, 16, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual_mix = module(x)
    expected_mix = _mhc_reference(reference_x, reference_parameters, 4, 1e-6, 1e-6, 4)
    for actual, expected in zip((actual_mix.pre, actual_mix.post, actual_mix.comb), expected_mix):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    previous = torch.rand(2, 3, 4, requires_grad=True)
    reference_previous = previous.detach().clone().requires_grad_()
    collapsed = module.collapse(x, previous)
    reference_collapsed = (reference_previous[..., None] * reference_x.float()).sum(2).to(dtype)
    torch.testing.assert_close(collapsed, reference_collapsed, rtol=0, atol=0)
    actual = module.expand(collapsed, x, actual_mix)
    # Share each FP32 cast across output streams. Repeating the cast inside
    # the loop would round separate BF16 gradient contributions prematurely.
    reference_collapsed_fp32 = reference_collapsed.float()
    reference_x_fp32 = reference_x.float()
    expected = torch.stack(
        [
            reference_collapsed_fp32 * expected_mix[1][..., out, None]
            + (reference_x_fp32 * expected_mix[2][..., :, out, None]).sum(2)
            for out in range(4)
        ],
        2,
    ).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    loss = (actual * upstream).sum() + actual_mix.pre.square().sum()
    reference_loss = (expected * upstream).sum() + expected_mix[0].square().sum()
    loss.backward()
    reference_loss.backward()
    tolerance = dict(atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x.grad, reference_x.grad, **tolerance)
    torch.testing.assert_close(previous.grad, reference_previous.grad, **tolerance)
    for name, parameter in module.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_te_norms_preserve_meta_storage_initialization_and_checkpoint_keys(dtype: torch.dtype) -> None:
    # This checks real optional TE modules on meta/CPU, without invoking CUDA
    # kernels. Environments without TE still run the eager mathematical tests.
    te = pytest.importorskip("transformer_engine.pytorch.module.rmsnorm")
    config = _tiny_config()
    config.text_config.dtype = dtype
    config.text_config.rms_norm_eps = 1e-20
    with torch.device("meta"):
        eager = DeepseekV41ForCausalLM(config, backend=_backend())
        fused = DeepseekV41ForCausalLM(config, backend=replace(_backend(), rms_norm="te"))
    assert all(parameter.is_meta for parameter in fused.parameters())
    norm_names = {"model.norm"}
    for index in range(config.text_config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        norm_names.update(f"{prefix}.{suffix}" for suffix in ("attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"))
        if index in config.text_config.kv_source_layer_ids:
            norm_names.update((f"{prefix}.attn.compressor.norm", f"{prefix}.attn.indexer.k_norm"))
    assert len(norm_names) == 29  # All seven norm sites across Full/Reuse/Reindex layers.
    for name in norm_names:
        baseline, actual = eager.get_submodule(name), fused.get_submodule(name)
        assert isinstance(baseline, DeepseekV41RMSNorm)
        assert isinstance(actual, te.RMSNorm)
        assert actual.weight.dtype == baseline.weight.dtype == dtype
        assert actual.weight.shape == baseline.weight.shape
        assert actual.eps == baseline.eps == 1e-20
        assert actual.zero_centered_gamma is False
        assert actual.weight.requires_grad == baseline.weight.requires_grad == (".indexer." not in name)
    for model in (eager, fused):
        model.to_empty(device="cpu")
        torch.manual_seed(913)
        model.initialize_weights(torch.device("cpu"), dtype=dtype)
        for name in norm_names:
            parameter = model.get_submodule(name).weight
            assert parameter.device.type == "cpu" and parameter.dtype == dtype
            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
        for index in config.text_config.kv_source_layer_ids:
            compressor = model.model.layers[str(index)].attn.compressor
            assert compressor.wkv.weight.dtype == torch.float32
            if compressor.ratio > 1:
                assert compressor.wgate.weight.dtype == torch.float32
            assert compressor.norm.weight.dtype == dtype
    expected = eager.state_dict()
    actual = fused.state_dict()
    extra_state = {name: value for name, value in actual.items() if name.endswith("._extra_state")}
    assert extra_state.keys() == {f"{name}._extra_state" for name in norm_names}
    assert all(value.dtype == torch.uint8 and value.numel() == 0 for value in extra_state.values())
    assert expected.keys() == actual.keys() - extra_state.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert eager.state_dict_adapter.get_hf_state_dict_keys(expected) == fused.state_dict_adapter.get_hf_state_dict_keys(
        actual
    )
    identities = {name: id(parameter) for name, parameter in fused.named_parameters()}
    with torch.no_grad():
        for name in norm_names:
            parameter = eager.get_submodule(name).weight
            parameter.copy_(torch.linspace(-0.75, 1.25, parameter.numel(), dtype=dtype).reshape_as(parameter))
        for parameter in fused.parameters():
            parameter.fill_(-13.5)
    exported = eager.state_dict_adapter.to_hf(eager.state_dict(), exclude_key_regex=r".*_extra_state.*")
    assert not any("_extra_state" in name for name in exported)
    restored = fused.state_dict_adapter.from_hf(exported)
    restored.update(extra_state)
    fused.load_state_dict(restored)
    assert identities == {name: id(parameter) for name, parameter in fused.named_parameters()}
    loaded = fused.state_dict()
    for name, expected in eager.state_dict().items():
        torch.testing.assert_close(loaded[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("rms_norm", ["torch", "quack"])
def test_unvalidated_norm_backends_are_rejected(rms_norm: str) -> None:
    with torch.device("meta"), pytest.raises(ValueError, match="torch_fp32 or te RMSNorm"):
        DeepseekV41ForCausalLM(_tiny_config(), backend=replace(_backend(), rms_norm=rms_norm))
