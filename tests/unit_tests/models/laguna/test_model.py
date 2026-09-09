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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.laguna.config import LagunaConfig
from nemo_automodel.components.models.laguna.model import LagunaForCausalLM
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.megatron import moe_utils


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


def _tiny_config() -> LagunaConfig:
    cfg = LagunaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_attention_heads_per_layer=[2, 4],
        num_key_value_heads=1,
        head_dim=4,
        gating="per-head",
        gating_types=["per_head", "per_head"],
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=4,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        moe_routed_scaling_factor=2.5,
        mlp_layer_types=["dense", "sparse"],
        router_aux_loss_coef=0.0,
        torch_dtype="float32",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dense_tiny_config() -> LagunaConfig:
    cfg = _tiny_config()
    cfg.mlp_layer_types = ["dense", "dense"]
    cfg.mlp_only_layers = [0, 1]
    return cfg


def _patch_weighted_swiglu(monkeypatch) -> None:
    def weighted_swiglu_eager(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        y1, y2 = torch.chunk(y, 2, -1)
        return (torch.nn.functional.silu(y1) * y2 * weights).to(y.dtype)

    # Keep Laguna CPU smokes independent of torch.compile availability; shared MoE tests cover the compiled helper.
    monkeypatch.setattr(moe_utils, "weighted_swiglu", weighted_swiglu_eager)


def test_laguna_forward_tiny_config():
    model = LagunaForCausalLM(_dense_tiny_config(), backend=_backend())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 4, 32)


def test_laguna_forward_tiny_config_with_moe_layer(monkeypatch):
    _patch_weighted_swiglu(monkeypatch)
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())
    model.eval()

    assert isinstance(model.model.layers["1"].mlp, MoE)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 4, 32)


def test_laguna_initialize_weights_cpu_path(monkeypatch):
    _patch_weighted_swiglu(monkeypatch)
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    sparse_layer = model.model.layers["1"].mlp

    assert sparse_layer.gate.e_score_correction_bias.dtype == torch.float32
    assert all(torch.isfinite(param).all().item() for param in model.parameters())

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.isfinite(output.logits).all()


def test_laguna_moe_defaults_match_checkpoint_routing():
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())
    sparse_layer = model.model.layers["1"].mlp

    assert isinstance(sparse_layer, MoE)
    assert model.model.moe_config.score_func == "sigmoid"
    assert model.model.moe_config.softmax_before_topk is False
    assert model.model.moe_config.norm_topk_prob is True
    assert model.model.moe_config.route_scale == 2.5
    assert model.model.moe_config.n_shared_experts == 1
    assert sparse_layer.gate.e_score_correction_bias is not None
    assert model.backend.gate_precision is torch.float32


def test_laguna_rejects_unsupported_swa_attention_sink():
    cfg = _tiny_config()
    cfg.swa_attention_sink_enabled = True

    with pytest.raises(NotImplementedError, match="swa_attention_sink_enabled=True"):
        LagunaForCausalLM(cfg, backend=_backend())


def test_laguna_attention_uses_per_layer_head_counts_and_per_head_gate():
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())

    layer0_attn = model.model.layers["0"].self_attn
    layer1_attn = model.model.layers["1"].self_attn

    assert layer0_attn.q_proj.weight.shape == (8, 16)
    assert layer0_attn.g_proj.weight.shape == (2, 16)
    assert layer1_attn.q_proj.weight.shape == (16, 16)
    assert layer1_attn.g_proj.weight.shape == (4, 16)
