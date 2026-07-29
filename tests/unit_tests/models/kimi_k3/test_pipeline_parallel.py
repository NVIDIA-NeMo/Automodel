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
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nemo_automodel.components.models.kimi_k3.model as kimi_k3_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import (
    KimiK3ForCausalLM,
    KimiK3MoE,
    KimiMLAAttention,
    _partition_attn_residual_blocks,
)
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP


def _torch_backend() -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )


@pytest.mark.parametrize(
    ("num_stages", "expected"),
    [
        (2, [(0, 48), (48, 93)]),
        (8, [(0, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, 84), (84, 93)]),
    ],
)
def test_real_k3_pipeline_ranges_preserve_attention_residual_blocks(num_stages, expected):
    ranges = _partition_attn_residual_blocks(93, 12, num_stages)
    assert [(layer_range.start, layer_range.stop) for layer_range in ranges] == expected


def test_more_pipeline_stages_than_attention_residual_blocks_is_rejected():
    with pytest.raises(ValueError, match="8 attention-residual blocks"):
        _partition_attn_residual_blocks(93, 12, 9)


def test_partial_block_can_use_an_output_only_pipeline_stage():
    ranges = _partition_attn_residual_blocks(4, 12, 2, allow_output_only_stage=True)
    assert [(layer_range.start, layer_range.stop) for layer_range in ranges] == [(0, 4), (4, 4)]


def test_partial_checkpoint_pipeline_keeps_block_on_first_stage_and_outputs_on_last():
    model = nn.Module()
    model.model = nn.Module()
    model.model.config = SimpleNamespace(num_hidden_layers=4, attn_res_block_size=12)
    model.model.output_attn_res_norm = nn.Identity()
    model.model.output_attn_res_proj = nn.Identity()

    generated = [
        ["model.embed_tokens", "model.layers.0", "model.layers.1"],
        ["model.layers.2", "model.layers.3", "model.norm", "lm_head"],
    ]
    fixed = KimiK3ForCausalLM.customize_pipeline_stage_modules(
        model,
        generated,
        layers_prefix="model.",
    )

    assert [name for name in fixed[0] if name.startswith("model.layers.")] == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.layers.3",
    ]
    assert not any(name.startswith("model.layers.") for name in fixed[1])
    assert "model.output_attn_res_norm" in fixed[1]
    assert "model.output_attn_res_proj" in fixed[1]


def test_customize_pipeline_modules_places_vision_on_first_stage():
    model = nn.Module()
    model.model = nn.Module()
    model.model.config = SimpleNamespace(num_hidden_layers=93, attn_res_block_size=12)
    model.model.output_attn_res_norm = nn.Identity()
    model.model.output_attn_res_proj = nn.Identity()
    model.vision_tower = nn.Identity()
    model.mm_projector = nn.Identity()

    generated = [
        ["model.embed_tokens", "model.vision_tower", "model.mm_projector", *[f"model.layers.{i}" for i in range(47)]],
        [*[f"model.layers.{i}" for i in range(47, 93)], "model.norm", "lm_head"],
    ]
    fixed = KimiK3ForCausalLM.customize_pipeline_stage_modules(
        model,
        generated,
        layers_prefix="model.",
    )

    assert [name for name in fixed[0] if name.startswith("model.layers.")] == [f"model.layers.{i}" for i in range(48)]
    assert [name for name in fixed[1] if name.startswith("model.layers.")] == [
        f"model.layers.{i}" for i in range(48, 93)
    ]
    assert "vision_tower" in fixed[0]
    assert "mm_projector" in fixed[0]
    assert all("vision_tower" not in stage and "mm_projector" not in stage for stage in fixed[1:])
    assert "model.output_attn_res_norm" not in fixed[0]
    assert "model.output_attn_res_proj" not in fixed[0]
    assert "model.output_attn_res_norm" in fixed[-1]
    assert "model.output_attn_res_proj" in fixed[-1]


def _tiny_config(
    *,
    num_hidden_layers: int = 4,
    attn_res_block_size: int | None = 2,
) -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers + 1,
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
            "full_attn_layers": list(range(1, num_hidden_layers + 1)),
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=attn_res_block_size,
    )


def test_text_config_registers_causal_lm_architecture():
    assert KimiK3TextConfig().architectures == ["KimiK3ForCausalLM"]


def test_composite_checkpoint_config_can_be_truncated_for_text_only_loading():
    text_config = _tiny_config(num_hidden_layers=8, attn_res_block_size=12)
    text_config.linear_attn_config = {
        **text_config.linear_attn_config,
        "kda_layers": [1, 2, 3, 5, 6, 7],
        "full_attn_layers": [4, 8],
    }
    config = KimiK3Config(text_config=text_config)

    model = KimiK3ForCausalLM(
        config,
        num_hidden_layers=5,
        kda_mode="fused_recurrent",
        backend=_torch_backend(),
    )

    assert model.config.num_hidden_layers == 5
    assert model.config.kda_mode == "fused_recurrent"
    assert model.config.linear_attn_config["kda_layers"] == [1, 2, 3, 5]
    assert model.config.linear_attn_config["full_attn_layers"] == [4]
    assert config.text_config.num_hidden_layers == 8


def test_truncated_model_requests_only_retained_checkpoint_layer_keys():
    backend = _torch_backend()
    backend.enable_hf_state_dict_adapter = True
    model = KimiK3ForCausalLM(
        KimiK3Config(text_config=_tiny_config(num_hidden_layers=8)),
        num_hidden_layers=4,
        backend=backend,
    )

    checkpoint_destinations = model.state_dict_adapter.to_hf(model.state_dict())
    layer_keys = [key for key in checkpoint_destinations if ".model.layers." in key]

    assert layer_keys
    assert all(any(f".model.layers.{layer_idx}." in key for layer_idx in range(4)) for key in layer_keys)
    assert any(key == "language_model.model.embed_tokens.weight" for key in checkpoint_destinations)
    assert any(key == "language_model.lm_head.weight" for key in checkpoint_destinations)


def test_pipeline_stage_weight_initialization_handles_pruned_modules():
    first_stage = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    first_stage.model.norm = None
    first_stage.model.output_attn_res_norm = None
    first_stage.model.output_attn_res_proj = None
    first_stage.lm_head = None
    first_stage.initialize_weights(torch.device("cpu"), dtype=torch.float32)

    last_stage = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    last_stage.model.embed_tokens = None
    last_stage.initialize_weights(torch.device("cpu"), dtype=torch.float32)


def test_weight_initialization_and_forward_without_attention_residuals():
    model = KimiK3ForCausalLM(
        _tiny_config(num_hidden_layers=1, attn_res_block_size=None),
        backend=_torch_backend(),
    )
    assert model.model.use_attn_residuals is False
    assert not hasattr(model.model, "output_attn_res_norm")
    assert not hasattr(model.model, "output_attn_res_proj")

    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits

    assert torch.isfinite(logits).all()


def test_k3_mla_uses_te_backend_and_matches_eager(monkeypatch):
    torch.manual_seed(3)
    config = _tiny_config(num_hidden_layers=1, attn_res_block_size=None)
    eager = KimiMLAAttention(config, layer_idx=0, backend=_torch_backend())
    captured = {}

    def fake_initialize_attn_module_and_func(**kwargs):
        captured.update(kwargs)

        def attention(query, key, value, **attention_kwargs):
            assert attention_kwargs["window_size"] == (-1, 0)
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
                scale=kwargs["softmax_scale"],
            )
            return output.transpose(1, 2)

        return nn.Identity(), attention

    monkeypatch.setattr(
        kimi_k3_model,
        "initialize_attn_module_and_func",
        fake_initialize_attn_module_and_func,
    )
    te_backend = _torch_backend()
    te_backend.attn = "te"
    te = KimiMLAAttention(config, layer_idx=0, backend=te_backend)
    te.load_state_dict(eager.state_dict())

    hidden_states = torch.randn(2, 6, config.hidden_size)
    min_value = torch.finfo(hidden_states.dtype).min
    attention_mask = torch.triu(
        torch.full((6, 6), min_value, dtype=hidden_states.dtype),
        diagonal=1,
    )[None, None]
    eager.eval()
    te.eval()
    with torch.no_grad():
        expected = eager(hidden_states, attention_mask=attention_mask)
        actual = te(hidden_states, attention_mask=attention_mask)

    assert captured["attn_impl"] == "te"
    assert captured["num_qk_channels"] == config.qk_nope_head_dim + config.qk_rope_head_dim
    assert captured["num_v_channels"] == config.v_head_dim
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("experts_backend", ["torch", "torch_mm"])
def test_k3_moe_training_forward_matches_reference_order(experts_backend):
    torch.manual_seed(5)
    config = _tiny_config(num_hidden_layers=1, attn_res_block_size=1)
    config.first_k_dense_replace = 0
    config.num_experts = 4
    config.num_experts_per_token = 2
    config.num_expert_group = 2
    config.topk_group = 1
    backend = _torch_backend()
    backend.experts = experts_backend

    model = KimiK3ForCausalLM(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    moe = model.model.layers["0"].mlp
    assert isinstance(moe, KimiK3MoE)
    assert type(moe.experts) is GroupedExperts
    assert moe.experts.config.apply_router_weight_after_down is True

    hidden_states = torch.randn(2, 3, config.hidden_size)
    token_mask = torch.ones(hidden_states.numel() // config.hidden_size, dtype=torch.bool)
    weights, indices, _ = moe.gate(hidden_states.flatten(0, 1), token_mask)
    assert weights.dtype == torch.float32
    assert weights.shape == indices.shape == (6, 2)

    moe.eval()
    with torch.no_grad():
        reference = moe(hidden_states)
    moe.train()
    with torch.no_grad():
        actual = moe(hidden_states)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_k3_moe_uses_shared_flex_dispatcher_experts(monkeypatch):
    config = _tiny_config(num_hidden_layers=1, attn_res_block_size=1)
    config.first_k_dense_replace = 0
    backend = _torch_backend()
    backend.experts = "torch_mm"
    backend.dispatcher = "hybridep"
    monkeypatch.setattr(kimi_k3_model, "get_world_size_safe", lambda: 2)

    model = KimiK3ForCausalLM(config, backend=backend)
    moe = model.model.layers["0"].mlp

    assert type(moe.experts) is GroupedExpertsDeepEP
    assert moe.experts.dispatcher_backend == "hybridep"
    assert moe.experts.config.apply_router_weight_after_down is True


def test_two_pipeline_stage_handoff_matches_full_model():
    torch.manual_seed(7)
    model = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    model.eval()

    stage0 = copy.deepcopy(model)
    stage1 = copy.deepcopy(model)
    for layer_idx in ("2", "3"):
        del stage0.model.layers[layer_idx]
    stage0.model.norm = None
    stage0.lm_head = None
    for layer_idx in ("0", "1"):
        del stage1.model.layers[layer_idx]
    stage1.model.embed_tokens = None

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.no_grad():
        expected = model(input_ids).logits
        hidden_states, block_residual = stage0(input_ids)
        actual = stage1(hidden_states, block_residual).logits

    assert block_residual.shape == (8, 1, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_handoff_at_real_twelve_layer_block_boundary_matches_forward_and_backward():
    torch.manual_seed(17)
    model = KimiK3ForCausalLM(
        _tiny_config(num_hidden_layers=24, attn_res_block_size=12),
        backend=_torch_backend(),
    )
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)

    stage0 = copy.deepcopy(model)
    stage1 = copy.deepcopy(model)
    for layer_idx in range(12, 24):
        del stage0.model.layers[str(layer_idx)]
    stage0.model.norm = None
    stage0.model.output_attn_res_norm = None
    stage0.model.output_attn_res_proj = None
    stage0.lm_head = None
    for layer_idx in range(12):
        del stage1.model.layers[str(layer_idx)]
    stage1.model.embed_tokens = None

    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids).logits
    hidden_states, block_residual = stage0(input_ids)
    actual = stage1(hidden_states, block_residual).logits

    assert block_residual.shape == (4, 1, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(
        stage0.model.layers["11"].mlp.down_proj.weight.grad,
        model.model.layers["11"].mlp.down_proj.weight.grad,
    )
    torch.testing.assert_close(
        stage1.model.layers["12"].mlp.down_proj.weight.grad,
        model.model.layers["12"].mlp.down_proj.weight.grad,
    )


def test_pipeline_stage_metas_include_block_residual():
    model = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    for layer_idx in ("0", "1"):
        del model.model.layers[layer_idx]
    model.model.embed_tokens = None

    inputs, outputs = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=2,
        seq_len=4,
        dtype=torch.float32,
    )

    assert [tensor.shape for tensor in inputs] == [(2, 4, 32), (8, 1, 32)]
    assert [tensor.shape for tensor in outputs] == [(2, 4, 64)]
