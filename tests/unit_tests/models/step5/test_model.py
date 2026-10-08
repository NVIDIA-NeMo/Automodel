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

"""Step-5 configuration, conversion and sensitive-behavior regression tests."""

import math
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.step5.configuration import ConfigValue, Step5Config, Step5VLConfig
from nemo_automodel.components.models.step5.model import Step5ForCausalLM, Step5ForConditionalGeneration
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


def tiny_config(**overrides: ConfigValue) -> Step5Config:
    values = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        dtype="float32",
        num_nextn_predict_layers=0,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=2,
        ),
    )
    values.update(overrides)
    return Step5Config(**values)


def make_model(**overrides: ConfigValue) -> Step5ForCausalLM:
    torch.manual_seed(137)
    return Step5ForCausalLM(
        tiny_config(**overrides),
        backend=BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm"),
    )


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the shared grouped-expert activation backward")
def test_tiny_forward_backward():
    model = make_model()
    inputs = torch.arange(16).unsqueeze(0)
    output = model(inputs, labels=inputs)
    assert output.logits.shape == (1, 16, 64)
    assert torch.isfinite(output.logits).all()
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if ".indexer." in name:
            assert parameter.grad is None  # Discrete top-k has no LM-loss derivative.
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert model.model.layers["3"].moe.gate.weight.grad.abs().sum() > 0


def test_rope_independent_oracle_and_wrong_layout():
    cfg = tiny_config()
    rope = Step5RotaryEmbedding(cfg, 0, span=4)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
    actual = rope(x, torch.tensor([[3]]))
    a, b = 3.0, 3.0 / math.sqrt(cfg.rope_theta[0])
    expected = torch.tensor(
        [
            [
                [
                    [
                        math.cos(a) - 3 * math.sin(a),
                        2 * math.cos(b) - 4 * math.sin(b),
                        math.sin(a) + 3 * math.cos(a),
                        2 * math.sin(b) + 4 * math.cos(b),
                        5.0,
                        6.0,
                    ]
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)
    interleaved = torch.tensor(
        [
            [
                [
                    [
                        math.cos(a) - 2 * math.sin(a),
                        math.sin(a) + 2 * math.cos(a),
                        3 * math.cos(b) - 4 * math.sin(b),
                        3 * math.sin(b) + 4 * math.cos(b),
                        5.0,
                        6.0,
                    ]
                ]
            ]
        ]
    )
    assert not torch.allclose(actual, interleaved)


def test_cast_preserves_original_precision():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    with torch.no_grad():
        indexer.w_proj.weight.fill_(1.0031)
        model.model.layers["3"].moe.gate.e_score_correction_bias.fill_(34.03125)
    original = indexer.w_proj.weight.detach().clone()
    cast_model_to_dtype(model, torch.bfloat16)
    assert indexer.w_proj.weight.dtype == torch.float32
    torch.testing.assert_close(indexer.w_proj.weight, original, rtol=0, atol=0)
    bias = model.model.layers["3"].moe.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias, torch.full_like(bias, 34.03125))


def test_fp32_storage_under_bf16_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = make_model(dtype="bfloat16")
    finally:
        torch.set_default_dtype(previous)
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in model._keep_in_fp32_modules_strict):
            assert parameter.dtype == torch.float32, name
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16


def test_adapter_roundtrip_and_directly_loaded_model():
    """Component conversion check; native distributed loading uses the recipe CLI."""
    model = make_model()
    adapter = model.state_dict_adapter
    native = model.state_dict()
    checkpoint = adapter.to_hf(native)
    restored = adapter.from_hf(checkpoint)
    assert restored.keys() == native.keys()
    for name in native:
        torch.testing.assert_close(restored[name], native[name], rtol=0, atol=0)
    other = make_model()
    other.load_state_dict(restored, strict=True)
    inputs = torch.arange(16).unsqueeze(0)
    torch.testing.assert_close(model(inputs).logits, other(inputs).logits, rtol=0, atol=0)


def test_sparse_selection_drops_history_and_breaks_ties():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    for parameter in indexer.parameters():
        torch.nn.init.zeros_(parameter)
    positions = torch.arange(12).unsqueeze(0)
    hidden = torch.zeros(1, 12, 32)
    keys, logits = indexer.project_keys(hidden, positions)
    selected = indexer(hidden, positions, keys, logits, positions, torch.zeros_like(positions, dtype=torch.bool))
    # At q=6 there are three complete history blocks, but only two history slots.
    # Equal scores retain blocks 0 and 1; current block 3 is appended, and 2 drops.
    assert selected.region_ids[0, 0, 6].tolist() == [0, 1, 3]
    assert selected.region_ids[0, 0, 4].tolist() == [0, 1, 2]
    assert selected.region_ids[0, 0, 0].tolist() == [0, 6, 6]


def test_packed_documents_do_not_leak():
    model = make_model().eval()
    ids = torch.arange(16).unsqueeze(0)
    positions = torch.arange(8).repeat(2).unsqueeze(0)
    packed = model(ids, position_ids=positions).logits
    separate = torch.cat([model(ids[:, :8]).logits, model(ids[:, 8:]).logits], dim=1)
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    changed = ids.clone()
    changed[:, :8] = 63
    torch.testing.assert_close(model(changed, position_ids=positions).logits[:, 8:], packed[:, 8:], rtol=0, atol=0)


def test_local_auto_config(tmp_path: Path) -> None:
    assert resolve_custom_config_cls("step5") is Step5Config
    cfg = tiny_config()
    cfg.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, Step5Config)
    assert loaded.sparse_config.topk == 2


def test_unsupported_sparse_semantics_fail():
    with pytest.raises(ValueError, match="Unsupported Step5 sparse semantics"):
        tiny_config(sparse_config={"compression_method": "mean"})


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the VLM shared-expert activation and backward")
def test_vlm_image_path():
    text = tiny_config(num_hidden_layers=1, moe_layer_list=[])
    config = Step5VLConfig(
        text_config=text,
        vision_config=dict(width=16, layers=1, heads=2, image_size=28, patch_size=14, mlp_ratio=2),
        image_token_id=63,
    )
    model = Step5ForConditionalGeneration(
        config, backend=BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm")
    )
    tokens = torch.tensor([[0, 63, 1, 2]])
    pixels = torch.randn(1, 3, 28, 28)
    output = model(tokens, pixel_values=pixels)
    assert output.logits.shape == (1, 4, 64)
    output.logits.square().mean().backward()
    assert model.vision_model.conv1.weight.grad.abs().sum() > 0


def test_post_silu_clamp_is_not_pre_silu_clamp():
    from nemo_automodel.components.moe.experts import swiglu_clamped_deepep

    gate = torch.tensor([[6.5, 7.5, 9.0, -1.0]], requires_grad=True)
    up = torch.tensor([[8.0, -9.0, 2.0, 3.0]], requires_grad=True)
    packed = torch.cat((gate, up), -1)
    actual = swiglu_clamped_deepep(packed, torch.ones(1, 1), 7.0, clamp_after_activation=True)
    expected = torch.nn.functional.silu(gate).clamp(max=7) * up.clamp(-7, 7)
    wrong = swiglu_clamped_deepep(packed, torch.ones(1, 1), 7.0, clamp_after_activation=False)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong, rtol=1e-5, atol=1e-5)
    actual_grad = torch.autograd.grad(actual.sum(), (gate, up), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (gate, up))
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize(
    "excluded",
    [
        ["model.layers.3.moe.experts.w13_weight", "model.layers.3.moe.experts.w2_weight"],
        ["model.layers.3.moe.gate_proj", "model.layers.3.moe.up_proj", "model.layers.3.moe.down_proj"],
        ["model.layers.3.moe"],
    ],
)
def test_checkpoint_parts_preserve_unquantized_expert_weights(excluded):
    model = make_model(
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": excluded,
        }
    )
    adapter = model.state_dict_adapter
    native = {
        "model.layers.3.moe.experts.gate_and_up_projs": torch.zeros(4, 32, 32, dtype=torch.bfloat16),
        "model.layers.3.moe.experts.down_projs": torch.zeros(4, 16, 32, dtype=torch.bfloat16),
    }
    source = {
        "model.layers.3.moe.gate_proj.weight": (torch.arange(4 * 16 * 32).reshape(4, 16, 32) / 256).bfloat16(),
        "model.layers.3.moe.up_proj.weight": torch.full((4, 16, 32), 0.375, dtype=torch.bfloat16),
        "model.layers.3.moe.down_proj.weight": torch.full((4, 32, 16), -0.625, dtype=torch.bfloat16),
    }
    for part in adapter.iter_checkpoint_load_parts(native):
        assert set(part.checkpoint_tensors) <= source.keys()
        assert not part.temporary_checkpoint_keys
        for name, target in part.checkpoint_tensors.items():
            assert target.dtype == torch.bfloat16
            target.copy_(source[name])
        part.finish()
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.gate_and_up_projs"][..., :16],
        source["model.layers.3.moe.gate_proj.weight"].transpose(1, 2),
    )
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.gate_and_up_projs"][..., 16:],
        source["model.layers.3.moe.up_proj.weight"].transpose(1, 2),
    )
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.down_projs"], source["model.layers.3.moe.down_proj.weight"].transpose(1, 2)
    )


@pytest.mark.parametrize("bf16_gate", [False, True])
def test_checkpoint_parts_decode_fp8_with_optional_bf16_gate(bf16_gate):
    excluded = ["model.layers.3.moe.gate_proj"] if bf16_gate else []
    adapter = make_model(
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": excluded,
        }
    ).state_dict_adapter
    name = "model.layers.3.moe.experts.gate_and_up_projs"
    destination = torch.zeros(2, 128, 256, dtype=torch.bfloat16)
    parts = list(adapter.iter_checkpoint_load_parts({name: destination}))
    assert len(parts) == 1
    part = parts[0]
    for source_name, tensor in part.checkpoint_tensors.items():
        if source_name.endswith("_scale_inv"):
            tensor.fill_(0.25)
        elif bf16_gate and source_name.endswith("gate_proj.weight"):
            assert tensor.dtype == torch.bfloat16
            tensor.fill_(0.375)
            assert source_name not in part.temporary_checkpoint_keys
        else:
            assert tensor.dtype == torch.float8_e4m3fn
            tensor.copy_(torch.full(tensor.shape, 2.0).to(tensor.dtype))
    part.finish()
    torch.testing.assert_close(
        destination[..., :128], torch.full((2, 128, 128), 0.375 if bf16_gate else 0.5, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(destination[..., 128:], torch.full((2, 128, 128), 0.5, dtype=torch.bfloat16))


def test_checkpoint_parts_without_quantization_do_not_request_scales():
    adapter = make_model().state_dict_adapter
    destination = torch.zeros(4, 16, 32, dtype=torch.bfloat16)
    part = next(adapter.iter_checkpoint_load_parts({"model.layers.3.moe.experts.down_projs": destination}))
    assert set(part.checkpoint_tensors) == {"model.layers.3.moe.down_proj.weight"}
    assert not part.temporary_checkpoint_keys
    part.checkpoint_tensors["model.layers.3.moe.down_proj.weight"].fill_(0.375)
    part.finish()
    torch.testing.assert_close(destination, torch.full_like(destination, 0.375))
