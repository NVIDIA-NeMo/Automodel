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

"""The fp32 strict contract, checked the same way for every hybrid family.

Each family provides a tiny-model builder that returns the initialized model and the
exact set of parameter FQNs its ``_keep_in_fp32_modules_strict`` tokens are meant to
select. The tests then assert, per family, that (a) the tokens match exactly those
parameters, (b) a bf16 initialization and ``cast_model_to_dtype(bf16)`` leave exactly
those parameters fp32 with bitwise-unchanged values, and (c) no holder module survives.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.fp32_gates import GDN_FP32_PARAM_TOKENS, MAMBA_FP32_PARAM_TOKENS
from nemo_automodel.components.models.common.utils import cast_model_to_dtype

Builder = Callable[[torch.dtype], tuple[nn.Module, set[str]]]
GDN_PARAMS = ("A_log", "dt_bias")


def _backend(**overrides) -> BackendConfig:
    options = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    options.update(overrides)
    return BackendConfig(**options)


# --------------------------------------------------------------------------- families
def _qwen3_next(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "full_attention"],
    )
    model = Qwen3NextForCausalLM(config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.layers.0.linear_attn.{name}" for name in GDN_PARAMS}


def _qwen3_5(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["linear_attention", "full_attention"],
        attn_implementation="eager",
        torch_dtype="float32",
    )
    model = Qwen3_5ForCausalLM(config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.layers.0.linear_attn.{name}" for name in GDN_PARAMS}


def _qwen3_5_moe(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
    from nemo_automodel.components.moe.layers import MoEConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        tie_word_embeddings=False,
        layer_types=["linear_attention", "full_attention"],
    )
    moe_config = MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
    )
    model = Qwen3_5MoeForCausalLM.from_config(config, moe_config=moe_config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.layers.0.linear_attn.{name}" for name in GDN_PARAMS}


def _qwen3_8_flash_next(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    from nemo_automodel.components.models.qwen3_8_flash_next.config import (
        Qwen3_8_FlashNextConfig,
        Qwen3_8_FlashNextTextConfig,
    )
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from nemo_automodel.components.moe.layers import MoEConfig

    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention"],
        full_attention_interval=1,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    config = Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)
    moe_config = MoEConfig(
        dim=text.hidden_size,
        inter_dim=text.hidden_size,
        moe_inter_dim=text.moe_intermediate_size,
        n_routed_experts=text.num_experts,
        n_shared_experts=1,
        n_activated_experts=text.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=text.shared_expert_intermediate_size,
        dtype=torch.float32,
    )
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(config, moe_config=moe_config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.language_model.layers.0.linear_attn.{name}" for name in GDN_PARAMS}


def _kimi_k3(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    pytest.importorskip("fla")
    from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
    from tests.unit_tests.models.kimi_k3.test_pipeline_parallel import _tiny_config

    config = _tiny_config(num_hidden_layers=2)
    config.linear_attn_config = {**config.linear_attn_config, "kda_layers": [1], "full_attn_layers": [2]}
    # The fused decay gate is a Triton kernel; the torch gate keeps this on the CPU.
    config.kda_use_fused_gate = False
    model = KimiK3ForCausalLM(config, backend=_backend(attn="eager"))
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    names = ("A_log", "dt_bias", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight")
    return model, {f"model.layers.0.self_attn.{name}" for name in names}


def _kimi_linear(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    pytest.importorskip("fla.ops.kda.gate")
    from nemo_automodel.components.models.kimi_linear.model import KimiLinear48BForCausalLM
    from tests.unit_tests.models.kimi_linear.test_model import _tiny_kimi_config

    model = KimiLinear48BForCausalLM(
        _tiny_kimi_config(use_kda=True), backend=_backend(attn="eager", rms_norm="torch_fp32")
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.layers.0.self_attn.{name}" for name in GDN_PARAMS}


def _nemotron_v3(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_model import MockNemotronV3Config

    config = MockNemotronV3Config(layers_block_type=["mamba", "attention", "mlp", "moe"], num_hidden_layers=4)
    model = NemotronHForCausalLM(config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    return model, {f"model.layers.0.mixer.{name}" for name in ("A_log", "dt_bias", "D")}


def _glm5_next(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from tests.unit_tests.models.glm5_next.conftest import tiny_glm5_next_config

    config = tiny_glm5_next_config()
    model = Glm5NextForConditionalGeneration(config, backend=_backend())
    model.initialize_weights(torch.device("cpu"), dtype=dtype)
    expected = set()
    for index, block_type in enumerate(config.text_config.layer_types):
        prefix = f"model.language_model.layers.{index}"
        expected.update(f"{prefix}.{site}.{name}" for site in ("attn_hc", "ffn_hc") for name in ("base", "scale"))
        if block_type == "linear_attention":
            expected.update(f"{prefix}.self_attn.{name}" for name in GDN_PARAMS)
    return model, expected


def _inkling(dtype: torch.dtype) -> tuple[nn.Module, set[str]]:
    from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration
    from tests.unit_tests.models.inkling.parity_check_inkling import build_tiny_config

    config = build_tiny_config()
    config.torch_dtype = dtype
    config.text_config.torch_dtype = dtype
    model = InklingForConditionalGeneration.from_config(config, backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"))
    expected = set()
    for index, mlp_type in enumerate(config.text_config.mlp_layer_types):
        prefix = f"model.language_model.layers.{index}"
        sites = ("self_attn.k_sconv", "self_attn.v_sconv", "attn_sconv", "mlp_sconv")
        expected.update(f"{prefix}.{site}.conv1d.weight" for site in sites)
        if mlp_type == "sparse":
            expected.add(f"{prefix}.mlp.gate.e_score_correction_bias")
    return model, expected


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device"
)
FAMILIES = [
    pytest.param(_qwen3_next, id="qwen3_next", marks=needs_cuda),
    pytest.param(_qwen3_5, id="qwen3_5"),
    pytest.param(_qwen3_5_moe, id="qwen3_5_moe"),
    pytest.param(_qwen3_8_flash_next, id="qwen3_8_flash_next"),
    pytest.param(_kimi_k3, id="kimi_k3"),
    pytest.param(_kimi_linear, id="kimi_linear"),
    pytest.param(_nemotron_v3, id="nemotron_v3"),
    pytest.param(_glm5_next, id="glm5_next"),
    pytest.param(_inkling, id="inkling"),
]


# --------------------------------------------------------------------------- helpers
def _fp32_parameter_names(model: nn.Module) -> set[str]:
    return {name for name, param in model.named_parameters() if param.dtype is torch.float32}


def _assert_only_expected_fp32(model: nn.Module, expected: set[str]) -> None:
    assert _fp32_parameter_names(model) == expected
    for name, param in model.named_parameters():
        if name not in expected and param.is_floating_point():
            assert param.dtype is torch.bfloat16, name


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("build", FAMILIES)
def test_strict_tokens_match_exactly_the_fp32_parameters(build: Builder):
    model, expected = build(torch.float32)
    tokens = model._keep_in_fp32_modules_strict

    matched = {name for name, _ in model.named_parameters() if any(token in name for token in tokens)}

    assert matched == expected
    assert not any("_fp32_params" in name for name, _ in model.named_modules())
    assert not any("_fp32_params" in key for key in model.state_dict())


@pytest.mark.parametrize("build", FAMILIES)
def test_bf16_initialization_keeps_exactly_the_fp32_parameters_with_exact_values(build: Builder):
    torch.manual_seed(31)
    reference, expected = build(torch.float32)
    torch.manual_seed(31)
    model, _ = build(torch.bfloat16)

    _assert_only_expected_fp32(model, expected)
    reference_parameters = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        if name in expected:
            assert torch.equal(param, reference_parameters[name]), name


@pytest.mark.parametrize("build", FAMILIES)
def test_cast_model_to_dtype_restores_exactly_the_fp32_parameters_with_exact_values(build: Builder):
    model, expected = build(torch.float32)
    before = {name: param.detach().clone() for name, param in model.named_parameters() if name in expected}

    cast_model_to_dtype(model, torch.bfloat16)

    _assert_only_expected_fp32(model, expected)
    for name, param in model.named_parameters():
        if name in expected:
            assert torch.equal(param, before[name]), name


def test_gdn_and_mamba_families_share_the_token_constants():
    from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    gdn_models = (
        Qwen3NextForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_8_FlashNextForConditionalGeneration,
    )
    assert all(model._keep_in_fp32_modules_strict == list(GDN_FP32_PARAM_TOKENS) for model in gdn_models)
    # Omni reuses the NemotronH mixers, so its strict tokens are the V3 ones.
    assert NemotronHForCausalLM._keep_in_fp32_modules_strict == list(MAMBA_FP32_PARAM_TOKENS)
    assert NemotronOmniForConditionalGeneration._keep_in_fp32_modules_strict == list(MAMBA_FP32_PARAM_TOKENS)
