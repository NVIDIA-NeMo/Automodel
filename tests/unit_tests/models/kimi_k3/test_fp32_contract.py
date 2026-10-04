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
"""Kimi-K3 fp32 contract: KDA decay parameters, short convolutions and gated norm stay fp32 under HF names."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
from tests.unit_tests.models.kimi_k3.test_pipeline_parallel import _tiny_config, _torch_backend

pytest.importorskip("fla")

_KDA_FP32_NAMES = ("A_log", "dt_bias", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "o_norm.weight")


def _hybrid_config() -> KimiK3TextConfig:
    config = _tiny_config(num_hidden_layers=2)
    config.linear_attn_config = {**config.linear_attn_config, "kda_layers": [1], "full_attn_layers": [2]}
    # The fused decay gate is a Triton kernel; the torch gate keeps this test on the CPU.
    config.kda_use_fused_gate = False
    return config


def test_strict_fp32_tokens_select_exactly_the_kda_fp32_parameters():
    model = KimiK3ForCausalLM(_hybrid_config(), backend=_torch_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    tokens = model._keep_in_fp32_modules_strict
    expected = {f"model.layers.0.self_attn.{name}" for name in _KDA_FP32_NAMES}

    matched = {name for name, _ in model.named_parameters() if any(token in name for token in tokens)}
    assert matched == expected
    for name, parameter in model.named_parameters():
        assert parameter.dtype == (torch.float32 if name in expected else torch.bfloat16), name
    kda = model.model.layers["0"].self_attn
    assert torch.isfinite(kda.A_log).all() and torch.all(kda.dt_bias == 0)


def test_kda_fp32_state_round_trips_to_hf_keys():
    backend = replace(_torch_backend(), enable_hf_state_dict_adapter=True)
    model = KimiK3ForCausalLM(_hybrid_config(), backend=backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    native = model.state_dict()

    hf_state = model.state_dict_adapter.to_hf(native)

    assert all("_fp32_params" not in key for key in native)
    for name in _KDA_FP32_NAMES:
        assert f"language_model.model.layers.0.self_attn.{name}" in hf_state
    assert hf_state["language_model.model.layers.0.self_attn.A_log"].shape == (128,)
    assert set(model.state_dict_adapter.from_hf(hf_state)) == set(native)
