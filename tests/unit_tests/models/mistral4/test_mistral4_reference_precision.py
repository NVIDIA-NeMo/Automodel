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

import pytest
import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.mistral4.reference_precision import fp32_router_scores
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _extract_custom_args,
    _hf_reference_context,
)


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

    with fp32_router_scores(router):
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
        with fp32_router_scores(router):
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
        with fp32_router_scores(router):
            router(torch.ones(2, 16, dtype=torch.bfloat16))
    assert router.forward is original


def test_reference_context_rejects_other_model_families():
    with pytest.raises(ValueError, match="no HF Mistral4 routers"):
        with fp32_router_scores(torch.nn.Linear(16, 8)):
            pass


def test_tracked_recipe_selects_precision_context_without_relaxing_source_gates():
    from pathlib import Path

    recipe = Path(__file__).resolve().parents[4] / "examples/vlm_finetune/mistral4/mistral4_medpix.yaml"
    import yaml

    cfg = ConfigNode(yaml.safe_load(recipe.read_text()))
    custom, remaining = _extract_custom_args(["--config", str(recipe)])
    assert "hf_reference_context" not in custom
    assert not any("hf_reference_context" in value for value in remaining)
    assert "source_load" not in custom["parity_tolerance_profile_overrides"]
    assert "parity_threshold_overrides" not in custom
    router = _router()
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)
    with _hf_reference_context(cfg, router):
        assert router(hidden)[1].dtype == torch.float32
    assert router(hidden)[1].dtype == torch.bfloat16
    with _hf_reference_context(ConfigNode({}), router):
        assert router(hidden)[1].dtype == torch.bfloat16
