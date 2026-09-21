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

import nemo_automodel.components.models.kimi_k3.model as kimi_k3_model
import nemo_automodel.components.models.kimi_k3.situ as kimi_k3_situ
import nemo_automodel.components.moe.optimized_ops as optimized_ops
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiDecoderLayer, KimiK3MoE, _build_moe_config


def _tiny_config(**overrides) -> KimiK3TextConfig:
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
        **overrides,
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


_SITU_STATE_ATTRS = (
    "_situ_fwd_core",
    "_situ_bwd_core",
    "_attn_res_core",
    "_weighted_situ_fwd_fused_dispatch",
    "_weighted_situ_bwd_fused_dispatch",
    "_dense_situ_dispatch",
    "_SITU_CORES_COMPILED",
)


@pytest.fixture
def restore_situ_cores():
    """Snapshot and restore the module-level SiTU cores around a test."""
    snapshot = {name: getattr(kimi_k3_situ, name) for name in _SITU_STATE_ATTRS}
    rw_snapshot = (
        optimized_ops._rw_fwd_dispatch,
        optimized_ops._rw_bwd_x_dispatch,
        optimized_ops._rw_bwd_probs_dispatch,
        optimized_ops._RW_CORES_COMPILED,
    )
    yield
    for name, value in snapshot.items():
        setattr(kimi_k3_situ, name, value)
    (
        optimized_ops._rw_fwd_dispatch,
        optimized_ops._rw_bwd_x_dispatch,
        optimized_ops._rw_bwd_probs_dispatch,
        optimized_ops._RW_CORES_COMPILED,
    ) = rw_snapshot


def test_compile_situ_defaults_to_false():
    assert BackendConfig().compile_situ is False


def test_default_backend_leaves_cores_eager(restore_situ_cores):
    orig_fwd = kimi_k3_situ._situ_fwd_core
    orig_bwd = kimi_k3_situ._situ_bwd_core

    orig_attn_res = kimi_k3_situ._attn_res_core

    _build_moe(_torch_backend())

    assert kimi_k3_situ._situ_fwd_core is orig_fwd
    assert kimi_k3_situ._situ_bwd_core is orig_bwd
    assert kimi_k3_situ._attn_res_core is orig_attn_res
    assert kimi_k3_situ._SITU_CORES_COMPILED is False


def test_compile_situ_wraps_cores_once(restore_situ_cores):
    orig_fwd = kimi_k3_situ._situ_fwd_core
    orig_bwd = kimi_k3_situ._situ_bwd_core

    orig_attn_res = kimi_k3_situ._attn_res_core

    _build_moe(_torch_backend(compile_situ=True))

    assert kimi_k3_situ._SITU_CORES_COMPILED is True
    assert kimi_k3_situ._situ_fwd_core is not orig_fwd
    assert kimi_k3_situ._situ_bwd_core is not orig_bwd
    assert kimi_k3_situ._attn_res_core is not orig_attn_res

    # A second layer/model construction must not re-wrap the shared cores.
    compiled_fwd = kimi_k3_situ._situ_fwd_core
    compiled_bwd = kimi_k3_situ._situ_bwd_core
    _build_moe(_torch_backend(compile_situ=True))
    assert kimi_k3_situ._situ_fwd_core is compiled_fwd
    assert kimi_k3_situ._situ_bwd_core is compiled_bwd
    # The fused whole-tensor passes and the dense core are wrapped by the same flag.
    assert kimi_k3_situ._weighted_situ_fwd_fused_dispatch is not kimi_k3_situ._weighted_situ_fwd_fused
    assert kimi_k3_situ._weighted_situ_bwd_fused_dispatch is not kimi_k3_situ._weighted_situ_bwd_fused
    assert kimi_k3_situ._dense_situ_dispatch is not kimi_k3_situ._dense_situ_core


def _build_decoder_layer(backend: BackendConfig, **config_knobs) -> KimiDecoderLayer:
    config = _tiny_config(**config_knobs)  # the K3-only kernel knobs live on the model config
    moe_config = _build_moe_config(config, torch.float32, None)
    # layer 1 is a full-attention MoE layer of the tiny config (first_k_dense_replace=0), with attention
    # residuals on (attn_res_block_size=1), so both kernel flags have a call site to reach.
    return KimiDecoderLayer(config, 1, moe_config, backend)


def test_situ_triton_flag_wires_decoder_layer(monkeypatch):
    calls = []
    monkeypatch.setattr(kimi_k3_model, "_enable_situ_triton", lambda: calls.append("situ"))
    assert _tiny_config().situ_triton is False and not hasattr(BackendConfig(), "situ_triton")
    _build_decoder_layer(_torch_backend())
    assert calls == []
    _build_decoder_layer(_torch_backend(), situ_triton=True)
    assert calls == ["situ"]


def test_attn_res_triton_flag_wires_decoder_layer(monkeypatch):
    calls = []
    monkeypatch.setattr(kimi_k3_model, "_enable_attn_res_triton", lambda: calls.append("attn_res"))
    assert _tiny_config().attn_res_triton is False and not hasattr(BackendConfig(), "attn_res_triton")
    layer = _build_decoder_layer(_torch_backend())
    assert layer.use_attn_residuals and calls == []
    _build_decoder_layer(_torch_backend(), attn_res_triton=True)
    assert calls == ["attn_res"]


def test_compile_router_weight_defaults_false_and_wires_k3_moe(restore_situ_cores):
    assert BackendConfig().compile_router_weight is False
    optimized_ops._RW_CORES_COMPILED = False
    _build_moe(_torch_backend())
    assert optimized_ops._RW_CORES_COMPILED is False
    _build_moe(_torch_backend(compile_router_weight=True))
    assert optimized_ops._RW_CORES_COMPILED is True
    assert optimized_ops._rw_fwd_dispatch is not optimized_ops._rw_fwd_core
