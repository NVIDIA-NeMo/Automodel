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

import nemo_automodel.components.models.kimi_k3.situ as kimi_k3_situ
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiK3MoE, KimiRMSNorm, _build_moe_config


def _tiny_config() -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        first_k_dense_replace=0,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=1,
    )


def _build_moe(backend: BackendConfig) -> KimiK3MoE:
    config = _tiny_config()
    moe_config = _build_moe_config(config, torch.float32, None)
    return KimiK3MoE(config, moe_config, backend)


def _torch_backend(**overrides) -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        **overrides,
    )


@pytest.fixture
def restore_norm_core():
    """Snapshot and restore the module-level RMSNorm core around a test."""
    orig_dispatch = kimi_k3_situ._rms_norm_dispatch
    orig_flag = kimi_k3_situ._RMS_NORM_COMPILED
    yield
    kimi_k3_situ._rms_norm_dispatch = orig_dispatch
    kimi_k3_situ._RMS_NORM_COMPILED = orig_flag


def test_compile_norm_defaults_to_false():
    assert BackendConfig().compile_norm is False


def test_default_backend_leaves_norm_eager(restore_norm_core):
    orig_dispatch = kimi_k3_situ._rms_norm_dispatch

    _build_moe(_torch_backend())

    assert kimi_k3_situ._rms_norm_dispatch is orig_dispatch
    assert kimi_k3_situ._RMS_NORM_COMPILED is False


def test_compile_norm_wraps_core_once(restore_norm_core):
    orig_dispatch = kimi_k3_situ._rms_norm_dispatch

    _build_moe(_torch_backend(compile_norm=True))

    assert kimi_k3_situ._RMS_NORM_COMPILED is True
    assert kimi_k3_situ._rms_norm_dispatch is not orig_dispatch

    wrapped = kimi_k3_situ._rms_norm_dispatch
    _build_moe(_torch_backend(compile_norm=True))
    assert kimi_k3_situ._rms_norm_dispatch is wrapped


def test_compiled_norm_matches_eager(restore_norm_core):
    torch.manual_seed(0)
    module = KimiRMSNorm(hidden_size=32, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        module.weight.mul_(0.0).add_(torch.rand_like(module.weight) + 0.5)
    x = torch.randn(4, 7, 32, dtype=torch.float32, requires_grad=True)

    eager_out = module(x)
    eager_out.sum().backward()
    eager_grad = x.grad.detach().clone()
    x.grad = None

    # Modules built before the hook runs must still pick up the compiled core.
    kimi_k3_situ._compile_norm_core()
    compiled_out = module(x)
    compiled_out.sum().backward()

    torch.testing.assert_close(compiled_out, eager_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(x.grad, eager_grad, rtol=1e-5, atol=1e-6)


def test_benchmark_static_routing_requires_forced_balance():
    with pytest.raises(ValueError, match="benchmark_static_routing"):
        BackendConfig(benchmark_static_routing=True)

    with pytest.raises(ValueError, match="benchmark_static_routing"):
        BackendConfig(benchmark_static_routing=True, fake_balanced_gate=True, fake_gate_noise=0.1)

    cfg = BackendConfig(benchmark_static_routing=True, fake_balanced_gate=True, fake_gate_noise=0.0)
    assert cfg.benchmark_static_routing is True
