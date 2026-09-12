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

import copy

import pytest
import torch
from transformers.models.mistral4 import modeling_mistral4 as hf_module
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

from nemo_automodel.components.config.loader import ConfigNode
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _extract_custom_args,
    _hf_reference_context,
    _with_fp32_rotary_application,
)


def _reference_config(enabled: bool = True) -> ConfigNode:
    return ConfigNode({"ci": {"checkpoint_robustness": {"hf_reference_compute_fp32": enabled}}})


def _router(dtype=torch.bfloat16):
    torch.manual_seed(1234)
    config = Mistral4Config(hidden_size=16, num_local_experts=8, num_experts_per_tok=2, n_group=1, topk_group=1)
    router = Mistral4TopkRouter(config).to(dtype)
    with torch.no_grad():
        router.weight.normal_(std=0.1)
    return router


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_scores_preserve_projection_and_match_weights_and_gradients(dtype):
    router = _router(dtype)
    hidden = torch.randn(2, 5, 16, dtype=dtype, requires_grad=True)
    baseline_logits, baseline_weights, _ = router(hidden)
    original_weight = router.weight
    original_values = router.weight.detach().clone()

    with _hf_reference_context(_reference_config(), router):
        logits, weights, indices = router(hidden)
        assert logits.dtype == dtype
        assert weights.dtype == torch.float32
        torch.testing.assert_close(logits, baseline_logits, atol=0, rtol=0)
        expected = torch.nn.functional.linear(hidden.flatten(0, 1), router.weight).softmax(-1, dtype=torch.float32)
        expected_weights, expected_indices = expected.topk(2, dim=-1, sorted=False)
        expected_weights = expected_weights / (expected_weights.sum(-1, keepdim=True) + 1e-20)
        expected_weights = expected_weights * router.routed_scaling_factor
        torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
        torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
        upstream = torch.randn_like(weights)
        actual_grad = torch.autograd.grad(weights, (hidden, router.weight), upstream, retain_graph=True)
        expected_grad = torch.autograd.grad(expected_weights, (hidden, router.weight), upstream)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    assert router.weight is original_weight
    torch.testing.assert_close(router.weight, original_values, atol=0, rtol=0)
    assert "forward" not in router.__dict__
    torch.testing.assert_close(router(hidden)[1], baseline_weights, atol=0, rtol=0)


def test_reference_context_preserves_existing_instance_wrapper_and_cleans_up_after_failure():
    router = _router()
    native_forward = router.forward
    calls = []

    def device_dispatch_forward(hidden_states):
        """Emulate a device-map wrapper.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading axes.

        Returns:
            Logits [tokens, experts], weights [tokens, top_k], and indices [tokens, top_k]
            from the native router, with input leading axes flattened into tokens.
        """
        calls.append(hidden_states.dtype)
        return native_forward(hidden_states)

    router.forward = device_dispatch_forward
    unaffected = _router()
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="sentinel"):
        with _hf_reference_context(_reference_config(), router):
            assert router(hidden)[1].dtype == torch.float32
            assert unaffected(hidden)[1].dtype == torch.bfloat16
            raise RuntimeError("sentinel")
    assert router.forward is device_dispatch_forward
    assert calls == [torch.bfloat16]
    assert router(hidden)[1].dtype == torch.bfloat16


def test_reference_context_fails_if_hf_stops_using_the_expected_softmax():
    router = _router()

    def changed_forward(hidden_states):
        """Represent an incompatible router implementation.

        Args:
            hidden_states: Tensor of shape [tokens, hidden].

        Returns:
            The input tensor unchanged, with no softmax call.
        """
        return hidden_states

    router.forward = changed_forward
    original = router.forward
    with pytest.raises(RuntimeError, match="softmax contract changed"):
        with _hf_reference_context(_reference_config(), router):
            router(torch.ones(2, 16, dtype=torch.bfloat16))
    assert router.forward is original


def test_reference_context_rejects_other_model_families():
    with pytest.raises(ValueError, match="no HF Mistral4 routers"):
        with _hf_reference_context(_reference_config(), torch.nn.Linear(16, 8)):
            pass


def test_tracked_recipe_selects_precision_context_without_relaxing_source_or_automodel_reload():
    from pathlib import Path

    recipe = Path(__file__).resolve().parents[3] / "examples/vlm_finetune/mistral4/mistral4_medpix.yaml"
    import yaml

    cfg = ConfigNode(yaml.safe_load(recipe.read_text()))
    custom, remaining = _extract_custom_args(["--config", str(recipe)])
    assert "hf_reference_compute_fp32" not in custom
    assert not any("hf_reference_compute_fp32" in value for value in remaining)
    assert custom["parity_tolerance_profile_overrides"] == {"hf_reload": "relaxed"}
    assert "parity_threshold_overrides" not in custom
    router = _router()
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)
    with _hf_reference_context(cfg, router):
        assert router(hidden)[1].dtype == torch.float32
    assert router(hidden)[1].dtype == torch.bfloat16
    with _hf_reference_context(ConfigNode({}), router):
        assert router(hidden)[1].dtype == torch.bfloat16


def test_reference_precision_can_be_disabled_explicitly():
    router = _router()
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)
    with _hf_reference_context(_reference_config(enabled=False), router):
        assert router(hidden)[1].dtype == torch.bfloat16
    assert "forward" not in router.__dict__


def test_reference_norm_delays_rounding_and_preserves_weight_gradients():
    torch.manual_seed(12)
    norm = hf_module.Mistral4RMSNorm(32).bfloat16()
    with torch.no_grad():
        norm.weight.uniform_(0.1, 4.0)
    hidden = torch.randn(3, 17, 32, dtype=torch.bfloat16, requires_grad=True)
    model = torch.nn.ModuleList([_router(), norm])
    native = norm(hidden)
    original_weight = norm.weight
    x64 = hidden.detach().double()
    oracle = (
        x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.double()
    ).bfloat16()
    reference = copy.deepcopy(norm).float()
    reference_input = hidden.detach().float().requires_grad_()
    with _hf_reference_context(_reference_config(), model):
        actual = norm(hidden)
        assert actual.dtype == hidden.dtype
        assert (actual.float() - oracle.float()).square().sum() < (native.float() - oracle.float()).square().sum()
        torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        reference(reference_input).bfloat16().backward(gradient)
    assert norm.weight is original_weight and norm.weight.dtype == torch.bfloat16
    torch.testing.assert_close(hidden.grad, reference_input.grad.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(norm.weight.grad, reference.weight.grad.bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(norm(hidden), native, atol=0, rtol=0)


@pytest.mark.parametrize("interleave", [False, True])
def test_reference_rotation_matches_fp64_math_and_gradients(interleave):
    torch.manual_seed(12)
    q = torch.randn(2, 3, 11, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 1, 11, 8, dtype=torch.bfloat16, requires_grad=True)
    angles = torch.randn(2, 11, 4)
    cos = torch.cat([angles.cos()] * 2, -1)
    sin = torch.cat([angles.sin()] * 2, -1)
    function = hf_module.apply_rotary_pos_emb_interleave if interleave else hf_module.apply_rotary_pos_emb
    actual = _with_fp32_rotary_application(function)(q, k, cos, sin)
    for x, rotated in zip((q, k), actual):
        x64 = x.detach().double().requires_grad_()
        a, b = (x64[..., 0::2], x64[..., 1::2]) if interleave else x64.chunk(2, dim=-1)
        c, s = cos[..., :4].double().unsqueeze(1), sin[..., :4].double().unsqueeze(1)
        oracle = torch.cat((a * c - b * s, b * c + a * s), dim=-1).bfloat16()
        # FP32 subtraction can straddle a BF16 midpoint near cancellation.
        torch.testing.assert_close(rotated, oracle, atol=2e-7, rtol=0)
        gradient = torch.randn_like(rotated)
        rotated.backward(gradient)
        oracle.backward(gradient)
        # The same cancellation/midpoint rounding applies to the backward sum.
        torch.testing.assert_close(x.grad, x64.grad.bfloat16(), atol=2e-7, rtol=0)


@pytest.mark.parametrize("interleave", [False, True])
def test_reference_attention_uses_fp32_tables_and_restores_native_dispatch(interleave):
    torch.manual_seed(12)
    config = Mistral4Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        rope_interleave=interleave,
    )
    config._attn_implementation = "sdpa"
    attention = hf_module.Mistral4Attention(config, layer_idx=0).bfloat16()
    rotary = hf_module.Mistral4RotaryEmbedding(config)
    model = torch.nn.ModuleList([_router(), attention, rotary])
    unaffected = copy.deepcopy(attention)
    hidden = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    position_ids = torch.arange(8192, 8199).unsqueeze(0).expand(2, -1)
    native_tables = rotary(hidden, position_ids)
    expected_tables = rotary(hidden.float(), position_ids)
    native = unaffected(hidden, native_tables, None, position_ids)[0]
    original_functions = (hf_module.apply_rotary_pos_emb, hf_module.apply_rotary_pos_emb_interleave)
    with _hf_reference_context(_reference_config(), model):
        tables = rotary(hidden, position_ids)
        for actual, expected in zip(tables, expected_tables):
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual = attention(hidden, tables, None, position_ids)[0]
        assert actual.shape == hidden.shape and actual.dtype == hidden.dtype
        actual.square().mean().backward()
        assert torch.isfinite(hidden.grad).all() and torch.count_nonzero(hidden.grad) > 0
        assert (hf_module.apply_rotary_pos_emb, hf_module.apply_rotary_pos_emb_interleave) == original_functions
        torch.testing.assert_close(unaffected(hidden, native_tables, None, position_ids)[0], native, atol=0, rtol=0)
        with pytest.raises(AttributeError):
            attention(None, tables, None, position_ids)
        assert (hf_module.apply_rotary_pos_emb, hf_module.apply_rotary_pos_emb_interleave) == original_functions
    assert all("forward" not in module.__dict__ for module in model.modules())
    assert rotary(hidden, position_ids)[0].dtype == torch.bfloat16


def test_mistral4_gate_retains_fp32_selected_weights_and_reference_gradients():
    from nemo_automodel.components.models.mistral4.configuration import Mistral4Config as AMConfig
    from nemo_automodel.components.models.mistral4.model import _build_moe_config
    from nemo_automodel.components.moe.layers import Gate

    reference = _router()
    config = AMConfig(hidden_size=16, n_routed_experts=8, num_experts_per_tok=2, n_group=1, topk_group=1)
    assert _build_moe_config(config, {"router_weights_fp32": False}).router_weights_fp32 is False
    gate = Gate(_build_moe_config(config))
    with torch.no_grad():
        gate.weight.copy_(reference.weight)
        gate.e_score_correction_bias.zero_()
    x = torch.randn(10, 16, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights, indices, _ = gate(x, torch.ones(10, dtype=torch.bool), None)
    assert weights.dtype == torch.float32 and gate.weight.dtype == torch.bfloat16
    with _hf_reference_context(_reference_config(), reference):
        _, expected_weights, expected_indices = reference(reference_x)
    actual = torch.zeros(10, 8).scatter(1, indices, weights)
    expected = torch.zeros(10, 8).scatter(1, expected_indices, expected_weights)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-6, rtol=1e-3)
    torch.testing.assert_close(gate.weight.grad, reference.weight.grad, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("implementation", ["eager", "batched_mm"])
def test_reference_expert_sum_matches_fp64_accumulation_of_native_projections(implementation):
    torch.manual_seed(42)
    config = Mistral4Config(hidden_size=16, moe_intermediate_size=8, num_local_experts=4)
    config._experts_implementation = implementation
    experts = hf_module.Mistral4Experts(config).bfloat16()
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.normal_(std=0.5)
    hidden = torch.randn(13, 16, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.arange(4).expand(13, -1)
    weights = torch.randn(13, 4).softmax(-1).requires_grad_()
    native = experts(hidden, indices, weights)
    original_weights = tuple(experts.parameters())
    contributions = []
    for e in range(4):
        gate, up = torch.nn.functional.linear(hidden, experts.gate_up_proj[e]).chunk(2, -1)
        down = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, experts.down_proj[e])
        assert down.dtype == torch.bfloat16
        contributions.append((down * weights[:, e, None]).double())
    oracle = torch.stack(contributions).sum(0).bfloat16()
    model = torch.nn.ModuleList([_router(), experts])
    with _hf_reference_context(_reference_config(), model):
        actual = experts(hidden, indices, weights)
        assert actual.dtype == hidden.dtype
        torch.testing.assert_close(actual, oracle, atol=0, rtol=0)
        if implementation == "eager":
            # Batched/grouped experts already reduce in FP32; eager needs promotion.
            assert (actual.float() - oracle.float()).square().sum() < (native.float() - oracle.float()).square().sum()
        gradient = torch.randn_like(actual)
        parameters = (hidden, weights, *experts.parameters())
        actual_grad = torch.autograd.grad(actual, parameters, gradient, retain_graph=True)
        expected_grad = torch.autograd.grad(oracle, parameters, gradient)
        for actual_g, expected_g in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_g, expected_g, atol=1e-6, rtol=1e-5)
    assert config._experts_implementation == implementation
    assert all(
        actual is original and actual.dtype == torch.bfloat16
        for actual, original in zip(experts.parameters(), original_weights)
    )
    torch.testing.assert_close(experts(hidden, indices, weights), native, atol=0, rtol=0)


def test_reference_precision_survives_hf_save_reload_and_device_hooks(tmp_path):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    torch.manual_seed(42)
    config = Mistral4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    original = hf_module.Mistral4ForCausalLM(config).bfloat16().eval()
    original.save_pretrained(tmp_path)
    reloaded = hf_module.Mistral4ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, attn_implementation="sdpa", experts_implementation="batched_mm"
    ).eval()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, atol=0, rtol=0)
    experts = reloaded.model.layers[0].mlp.experts
    add_hook_to_module(experts, AlignDevicesHook(execution_device="cpu"))
    device_forward = experts.forward
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        native = reloaded(ids).logits
        with _hf_reference_context(_reference_config(), original):
            expected = original(ids).logits
        with _hf_reference_context(_reference_config(), reloaded):
            torch.testing.assert_close(reloaded(ids).logits, expected, atol=0, rtol=0)
        assert experts.config._experts_implementation == "batched_mm"
        assert experts.forward is device_forward
        with pytest.raises(RuntimeError, match="sentinel"):
            with _hf_reference_context(_reference_config(), reloaded):
                torch.testing.assert_close(reloaded(ids).logits, expected, atol=0, rtol=0)
                raise RuntimeError("sentinel")
        assert experts.config._experts_implementation == "batched_mm"
        assert experts.forward is device_forward
        torch.testing.assert_close(reloaded(ids).logits, native, atol=0, rtol=0)
