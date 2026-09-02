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

"""DeepSeek V3 and V3.2 must share one router precision policy.

V3.2 skips DeepseekV3ForCausalLM.__init__, so a default installed there does
not reach it. Parametrizing over V3, V3.2, and Kimi K2 keeps the construction
paths from drifting apart silently.
"""

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# Mock fast_hadamard_transform before importing deepseek_v32 modules
if "fast_hadamard_transform" not in sys.modules:
    mock_hadamard = types.ModuleType("fast_hadamard_transform")
    mock_hadamard.__spec__ = importlib.util.spec_from_loader("fast_hadamard_transform", loader=None)
    mock_hadamard.hadamard_transform = lambda x, scale: x
    sys.modules["fast_hadamard_transform"] = mock_hadamard

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM
from nemo_automodel.components.models.kimi_k2.config import KimiK2Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate

_DEEPSEEK_MODEL_CASES = (
    (
        DeepseekV3ForCausalLM,
        "nemo_automodel.components.models.deepseek_v3.model.DeepseekV3Model",
    ),
    (
        DeepseekV32ForCausalLM,
        "nemo_automodel.components.models.deepseek_v32.model.DeepseekV32Model",
    ),
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        tie_word_embeddings=False,
        hidden_size=16,
        vocab_size=32,
        torch_dtype="bfloat16",
    )


def _backend(*, gate_precision: torch.dtype | None = None) -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        gate_precision=gate_precision,
    )


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _DEEPSEEK_MODEL_CASES)
def test_deepseek_gate_precision_defaults_to_fp32_without_mutating_backend(model_cls, inner_model_path):
    backend = _backend()

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert backend.gate_precision is None
    assert model.backend.gate_precision is torch.float32
    assert inner_model_cls.call_args.kwargs["backend"] is model.backend


@pytest.mark.parametrize(("model_cls", "inner_model_path"), _DEEPSEEK_MODEL_CASES)
def test_deepseek_gate_precision_respects_explicit_override(model_cls, inner_model_path):
    backend = _backend(gate_precision=torch.bfloat16)

    with patch(inner_model_path) as inner_model_cls:
        model = model_cls(_config(), backend=backend)

    assert model.backend is backend
    assert model.backend.gate_precision is torch.bfloat16
    assert inner_model_cls.call_args.kwargs["backend"] is backend


@pytest.mark.parametrize("model_cls", [DeepseekV3ForCausalLM, DeepseekV32ForCausalLM])
def test_deepseek_preserves_score_correction_bias_in_fp32(model_cls):
    assert "e_score_correction_bias" in model_cls._keep_in_fp32_modules_strict


# These cases construct for real: layer 0 is dense (first_k_dense_replace=1) and
# layer 1 is MoE, so a Gate exists and every stage can be asserted on the router
# the model actually built. Expert dims are tiny; n_group=2 keeps grouped routing on.
_COMMON = dict(
    vocab_size=100,
    hidden_size=64,
    num_attention_heads=4,
    num_hidden_layers=2,
    first_k_dense_replace=1,
    intermediate_size=128,
    qk_rope_head_dim=16,
    v_head_dim=16,
    qk_nope_head_dim=16,
    torch_dtype="bfloat16",
)

_MOE = dict(
    moe_intermediate_size=64,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
)


def _v3_config() -> DeepseekV3Config:
    return DeepseekV3Config(**_COMMON, **_MOE)


def _v32_config() -> DeepseekV32Config:
    return DeepseekV32Config(
        **_COMMON,
        **_MOE,
        qk_head_dim=32,
        kv_lora_rank=32,
        q_lora_rank=64,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=16,
    )


def _kimi_k2_config() -> KimiK2Config:
    # Kimi K2 checkpoints are config only on the V3 path: KimiK2Config subclasses
    # DeepseekV3Config and the architecture resolves to DeepseekV3ForCausalLM.
    return KimiK2Config(**_COMMON, **_MOE)


def _router_config(**overrides) -> MoEConfig:
    """Tiny grouped router matching the DeepSeek policy (n_expert_groups > 1)."""
    base = dict(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=2,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=2.5,
        norm_topk_prob=True,
        router_weights_fp32=True,
        dtype=torch.bfloat16,
    )
    base.update(overrides)
    return MoEConfig(**base)


def _run_gate(moe_config, gate_precision):
    torch.manual_seed(0)
    gate = Gate(moe_config, gate_precision=gate_precision)
    gate.weight.data.normal_(std=0.02)
    x = torch.randn(16, moe_config.dim, dtype=torch.bfloat16)
    token_mask = torch.ones(16, dtype=torch.bool)
    return gate, gate(x, token_mask, None)


def test_gate_hands_fp32_weights_to_expert_compute():
    """Out stage: with the DeepSeek policy, bf16 in still yields fp32 weights."""
    gate, (weights, indices, _) = _run_gate(_router_config(), torch.float32)

    assert gate.score_dtype is torch.float32
    assert weights.dtype is torch.float32
    assert indices.shape == (16, 2)


def test_gate_without_policy_downcasts_weights():
    """The bug this PR fixes: type_as(x) hands bf16 weights to expert compute."""
    _, (weights, _, _) = _run_gate(_router_config(router_weights_fp32=False), None)

    assert weights.dtype is torch.bfloat16


def test_explicit_gate_precision_also_moves_scoring():
    """Score has no separate knob: overriding Proj to bf16 takes Score with it."""
    gate, _ = _run_gate(_router_config(), torch.bfloat16)

    assert gate.score_dtype is torch.bfloat16


_DEEPSEEK_MOE_CONFIG_CASES = (
    pytest.param(DeepseekV3ForCausalLM, _v3_config, id="v3"),
    pytest.param(DeepseekV32ForCausalLM, _v32_config, id="v32"),
    pytest.param(DeepseekV3ForCausalLM, _kimi_k2_config, id="kimi_k2"),
)


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_selected_router_weights_default_to_fp32(model_cls, config_fn):
    model = model_cls(config_fn(), backend=_backend())
    assert model.model.moe_config.router_weights_fp32 is True


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_selected_router_weights_are_overridable(model_cls, config_fn):
    model = model_cls(
        config_fn(),
        backend=_backend(),
        moe_overrides={"router_weights_fp32": False},
    )
    assert model.model.moe_config.router_weights_fp32 is False


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_router_param_stays_in_model_dtype(model_cls, config_fn):
    """Param stage: HF stores the router weight in model dtype and casts at use."""
    model = model_cls(config_fn(), backend=_backend())
    assert model.model.moe_config.gate_dtype is None


@pytest.mark.parametrize(("model_cls", "config_fn"), _DEEPSEEK_MOE_CONFIG_CASES)
def test_deepseek_router_stages_on_constructed_model(model_cls, config_fn):
    """Param / Proj / Score / Out asserted on the Gate the model actually built."""
    model = model_cls(config_fn(), backend=_backend())
    gate = model.model.layers["1"].mlp.gate
    assert isinstance(gate, Gate)

    assert gate.weight.dtype is torch.bfloat16  # Param: model dtype, cast at use
    assert gate.e_score_correction_bias.dtype is torch.float32
    assert gate.gate_precision is torch.float32  # Proj
    assert gate.score_dtype is torch.float32  # Score
    assert gate.router_weights_fp32 is True

    torch.manual_seed(0)
    gate.weight.data.normal_(std=0.02)
    x = torch.randn(16, model.model.moe_config.dim, dtype=torch.bfloat16)
    weights, indices, _ = gate(x, torch.ones(16, dtype=torch.bool), None)
    assert weights.dtype is torch.float32  # Out
    assert indices.shape == (16, model.model.moe_config.n_activated_experts)


# Released DeepSeek-V3 (rev e815299) stores gate.weight in BF16 and
# e_score_correction_bias in F32 in all 59 routers, with a large positive
# per-layer bias mean (2.23 to 8.04) and a small per-expert spread
# (std 0.0018 to 0.0236). Cases are (bias_mean, bias_std, router_overrides,
# weight_atol): the first two use the tiny 8-expert grouped router, the third
# matches the released V3 routing shape (256 experts, top-8, 8 groups limited
# to 4) at a reduced hidden dim so it stays a CPU test.
#
# weight_atol is None where the weights must match bitwise. At top-2 the norm
# denominator is a two-element sum, and a + b == b + a exactly, so the reference's
# unsorted top-k cannot reorder it. Only v3_shape sums 8 elements, where order
# does change the result.
_PARITY_CASES = (
    pytest.param(0.0, 0.1, {}, None, id="zero_centered"),
    pytest.param(4.95, 0.002, {}, None, id="v3_like_bias"),
    pytest.param(
        4.95,
        0.002,
        dict(n_routed_experts=256, n_activated_experts=8, n_expert_groups=8, n_limited_groups=4),
        1e-7,
        id="v3_shape",
    ),
)


@pytest.mark.parametrize(("bias_mean", "bias_std", "router_overrides", "weight_atol"), _PARITY_CASES)
def test_deepseek_gate_matches_hf_reference_router_grouped(bias_mean, bias_std, router_overrides, weight_atol):
    """Proj/Score/Out parity vs the pinned HF reference, grouped routing.

    The reference splits the router across two objects: DeepseekV3TopkRouter does
    the fp32 projection and returns logits only, while DeepseekV3MoE.route_tokens_to_experts
    does sigmoid/bias/group-mask/top-k/norm/scale. Automodel's Gate does all of it,
    so the comparison drives both.
    """
    # Imported here, not at module scope: modeling_deepseek_v3 pulls in the whole
    # transformers.generation stack, which would couple the other tests in this
    # file to dependencies none of them need.
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    cfg = _router_config(**router_overrides)
    gate = Gate(cfg, gate_precision=torch.float32).eval()
    torch.manual_seed(0)
    gate.weight.data.normal_(std=0.02)
    gate.e_score_correction_bias.normal_(mean=bias_mean, std=bias_std)

    hf_cfg = DeepseekV3Config(
        hidden_size=cfg.dim,
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.n_activated_experts,
        n_group=cfg.n_expert_groups,
        topk_group=cfg.n_limited_groups,
        routed_scaling_factor=cfg.route_scale,
        norm_topk_prob=cfg.norm_topk_prob,
        moe_intermediate_size=cfg.moe_inter_dim,
        n_shared_experts=cfg.n_shared_experts,
    )
    hf_moe = DeepseekV3MoE(hf_cfg).eval()  # experts are allocated but never run
    with torch.no_grad():
        hf_moe.gate.weight.copy_(gate.weight)  # bf16 -> fp32 is exact
        hf_moe.gate.e_score_correction_bias.copy_(gate.e_score_correction_bias)

    x = torch.randn(64, cfg.dim, dtype=torch.bfloat16)
    w_am, i_am, _ = gate(x, torch.ones(64, dtype=torch.bool), None)

    with torch.no_grad():
        router_logits = hf_moe.gate(x)
        i_hf, w_hf = hf_moe.route_tokens_to_experts(router_logits)

    assert router_logits.dtype is torch.float32  # Proj: reference projects in fp32
    assert w_hf.dtype is torch.float32  # Out: reference never casts back
    assert w_am.dtype is torch.float32  # Out: this PR's default matches it

    # Automodel's top-k is sorted, the reference's is sorted=False; compare as sets
    # by sorting each side's indices and applying the same permutation to the weights.
    o_am, o_hf = i_am.argsort(dim=1), i_hf.argsort(dim=1)
    assert torch.equal(i_am.gather(1, o_am), i_hf.gather(1, o_hf))
    w_am_sorted, w_hf_sorted = w_am.gather(1, o_am), w_hf.gather(1, o_hf)
    if weight_atol is None:
        assert torch.equal(w_am_sorted, w_hf_sorted)
    else:
        # Measured max delta 5.96e-08, two fp32 ulp at these magnitudes (~0.31).
        assert torch.allclose(w_am_sorted, w_hf_sorted, atol=weight_atol, rtol=0)


def test_kimi_k2_routes_to_the_deepseek_v3_construction_path():
    """Kimi K2 is config only: it inherits V3's router precision policy.

    If Kimi K2 ever gains its own ForCausalLM, it will also need its own copy of
    the fp32 gate_precision default and router_weights_fp32, the same way V3.2
    does. This test fails at that moment.
    """
    from nemo_automodel._transformers.registry import ModelRegistry, resolve_custom_config_cls

    assert resolve_custom_config_cls("kimi_k2") is KimiK2Config
    assert issubclass(KimiK2Config, DeepseekV3Config)
    assert ModelRegistry.get_model_cls_from_model_arch("DeepseekV3ForCausalLM") is DeepseekV3ForCausalLM
