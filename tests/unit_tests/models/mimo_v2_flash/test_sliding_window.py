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

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config, MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashModel
from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config as LegacyMiMoV2Config
from nemo_automodel.components.models.mimo_v25.model import MiMoV2Model
from nemo_automodel.components.moe.config import MoEConfig

_CONFIG_CLASSES = [MiMoV2FlashConfig, MiMoV2Config, LegacyMiMoV2Config]


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES)
@pytest.mark.parametrize(
    ("window_kwargs", "expected"),
    [
        ({"sliding_window": 128, "sliding_window_size": 128}, 129),
        ({"sliding_window_size": 4}, 5),
        ({"sliding_window_size": 0}, 1),
        ({"sliding_window": 17}, 17),
        ({"sliding_window": None, "sliding_window_size": None}, None),
    ],
)
def test_sliding_window_config_round_trip(
    config_cls: type[MiMoV2FlashConfig] | type[LegacyMiMoV2Config],
    window_kwargs: dict[str, int | None],
    expected: int | None,
    tmp_path: Path,
) -> None:
    config = config_cls(num_hidden_layers=1, **window_kwargs)
    for _ in range(2):
        assert config.sliding_window == expected
        config.save_pretrained(tmp_path)
        config = config_cls.from_pretrained(tmp_path)
    assert config.sliding_window == expected


@pytest.mark.parametrize("config_cls", [MiMoV2FlashConfig, MiMoV2Config])
def test_default_window_includes_query(config_cls: type[MiMoV2FlashConfig]) -> None:
    assert config_cls().sliding_window == 129


def test_legacy_full_attention_default_is_unchanged() -> None:
    assert LegacyMiMoV2Config().sliding_window is None


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES)
def test_explicit_from_pretrained_window_override(
    config_cls: type[MiMoV2FlashConfig] | type[LegacyMiMoV2Config], tmp_path: Path
) -> None:
    config_cls(num_hidden_layers=1, sliding_window=128, sliding_window_size=128).save_pretrained(tmp_path)
    assert config_cls.from_pretrained(tmp_path, sliding_window=17).sliding_window == 17


@pytest.mark.parametrize("config_cls", [MiMoV2FlashConfig, MiMoV2Config])
def test_checkpoint_config_resolution_includes_query(config_cls: type[MiMoV2FlashConfig], tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": config_cls.model_type,
                "num_hidden_layers": 1,
                "sliding_window": 128,
                "sliding_window_size": 128,
            }
        )
    )
    config = get_hf_config(tmp_path, attn_implementation="sdpa")
    assert isinstance(config, config_cls)
    assert config.sliding_window == 129


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES)
@pytest.mark.parametrize("attn_implementation", [None, "eager", "sdpa"])
@pytest.mark.parametrize("padding", [False, True])
def test_sliding_mask_matches_inclusive_reference_window(
    config_cls: type[MiMoV2FlashConfig] | type[LegacyMiMoV2Config],
    attn_implementation: str | None,
    padding: bool,
) -> None:
    # Checkpoint size is the number of PREVIOUS keys, as in SGLang's inclusive
    # flash-attention left window; HF #49170 translates it to 129 total keys.
    config = config_cls(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        v_head_dim=8,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        partial_rotary_factor=0.5,
        hybrid_layer_pattern=[1],
        moe_layer_freq=[0],
        n_routed_experts=None,
        num_experts_per_tok=1,
        sliding_window=128,
        sliding_window_size=128,
        torch_dtype="float32",
        attn_implementation=attn_implementation,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False)
    if config_cls is LegacyMiMoV2Config:
        model = MiMoV2Model(
            config,
            MoEConfig(
                dim=16,
                inter_dim=32,
                moe_inter_dim=16,
                n_routed_experts=0,
                n_shared_experts=0,
                n_activated_experts=1,
                n_expert_groups=1,
                n_limited_groups=1,
                train_gate=True,
                gate_bias_update_factor=0.0,
                aux_loss_coeff=0.0,
                score_func="sigmoid",
                route_scale=1.0,
                norm_topk_prob=True,
            ),
            backend,
        )
    else:
        model = MiMoV2FlashModel(config, backend)
    inputs = torch.zeros(2, 257, 16)
    positions = torch.arange(257).unsqueeze(0)
    padding_mask = torch.ones(2, 257, dtype=torch.long) if padding else None
    if padding:
        padding_mask[0, 200:] = 0
        padding_mask[1, 180:] = 0
    if config_cls is LegacyMiMoV2Config:
        masks = model._build_causal_mask_mapping(inputs, padding_mask, positions)
    else:
        masks = model._build_causal_mask_mapping(inputs, padding_mask, positions, positions[0])

    expected = torch.zeros(2, 1, 257, 257, dtype=torch.bool)
    for query in range(257):
        expected[:, :, query, max(0, query - 128) : query + 1] = True
    full_expected = torch.ones_like(expected).tril()
    if padding:
        expected &= padding_mask[:, None, None, :].bool()
        full_expected &= padding_mask[:, None, None, :].bool()
    torch.testing.assert_close(masks["sliding_attention"] == 0, expected)
    torch.testing.assert_close(masks["full_attention"] == 0, full_expected)
    assert expected[0, 0, 127].sum() == 128
    assert expected[0, 0, 128].sum() == 129
    assert expected[0, 0, 128, 0]
    assert not expected[0, 0, 129, 0]


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES)
def test_attention_module_forward_backward_matches_reference(
    config_cls: type[MiMoV2FlashConfig] | type[LegacyMiMoV2Config],
    record_property: Callable[[str, float], None],
) -> None:
    from transformers.models.mimo_v2_flash.configuration_mimo_v2_flash import MiMoV2FlashConfig as HFConfig
    from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import MiMoV2FlashAttention as HFAttention

    from nemo_automodel.components.models.mimo_v2_flash.model import (
        MiMoV2FlashAttention,
        _fallback_additive_mask,
    )
    from nemo_automodel.components.models.mimo_v25.model import MiMoV2Attention

    torch.manual_seed(20261013)
    config = config_cls(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        v_head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=4,
        partial_rotary_factor=0.5,
        hybrid_layer_pattern=[1],
        sliding_window=128,
        sliding_window_size=128,
        add_swa_attention_sink_bias=True,
        attention_value_scale=0.707,
        torch_dtype="float32",
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False)
    if config_cls is LegacyMiMoV2Config:
        attention = MiMoV2Attention(config, True, 0, "split", backend, torch.float32)
    else:
        attention = MiMoV2FlashAttention(config, backend, True, 0)
    # Explicit inclusive size is the independent oracle, not the old HF
    # checkpoint-config conversion. HF #49170 fixes that conversion separately.
    reference_config = HFConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        v_head_dim=4,
        layer_types=["sliding_attention"],
        sliding_window=129,
        attention_value_scale=0.707,
        attn_implementation="eager",
    )
    reference = HFAttention(reference_config, 0)
    with torch.no_grad():
        reference.sinks.copy_(torch.linspace(-0.25, 0.25, 4))
        attention.attention_sink_bias.copy_(reference.sinks)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(attention, name).weight.copy_(getattr(reference, name).weight)

    hidden = torch.randn(1, 257, 16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    angles = torch.arange(257).float()[:, None] * torch.tensor([1.0, 0.01])[None, :]
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(0)
    rope = (angles.cos(), angles.sin())
    actual_mask = _fallback_additive_mask(1, 257, torch.float32, torch.device("cpu"), None, config.sliding_window)
    reference_mask = torch.full((1, 1, 257, 257), float("-inf"))
    for query in range(257):
        reference_mask[:, :, query, max(0, query - 128) : query + 1] = 0
    actual = attention(hidden, position_embeddings=rope, attention_mask=actual_mask)
    if config_cls is not LegacyMiMoV2Config:
        actual = actual[0]
    expected = reference(reference_hidden, position_embeddings=rope, attention_mask=reference_mask)[0]
    for start, stop in ((0, 128), (128, 257)):
        torch.testing.assert_close(actual[:, start:stop], expected[:, start:stop], atol=1e-6, rtol=1e-5)
        record_property(
            f"output_{start}_{stop}_max_abs", (actual[:, start:stop] - expected[:, start:stop]).abs().max().item()
        )
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-6, rtol=1e-5)
    record_property("input_gradient_max_abs", (hidden.grad - reference_hidden.grad).abs().max().item())
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        torch.testing.assert_close(
            getattr(attention, name).weight.grad, getattr(reference, name).weight.grad, atol=2e-5, rtol=1e-5
        )
        record_property(
            name + "_gradient_max_abs",
            (getattr(attention, name).weight.grad - getattr(reference, name).weight.grad).abs().max().item(),
        )
