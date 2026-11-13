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
from nemo_automodel.components.models.inkling2.configuration import (
    Inkling2Config,
    Inkling2TextConfig,
    Inkling2VisionConfig,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.quantization import dequantize_mxfp8_experts, dequantize_nvfp4_experts
from nemo_automodel.components.models.inkling2.state_dict_adapter import (
    Inkling2StateDictAdapter,
    native_to_raw_key,
    raw_to_native_key,
)

from .parity_check_inkling2 import TINY_TEXT_CONFIG, _randomize

pytestmark = pytest.mark.cpu_only

_BACKEND = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
_COMPANIONS = ("_scale", "_amax", "_outer_scale_denominator")


def _released_text_config() -> Inkling2TextConfig:
    return Inkling2TextConfig(local_layer_ids=[i for i in range(48) if (i + 1) % 6], torch_dtype="bfloat16")


def _released_checkpoint_keys(config: Inkling2TextConfig, vision_layers: int = 23) -> list[str]:
    """Rebuild the key list of thinkingmachines/Inkling-2-Small model.safetensors.index.json.

    Experts of layers 0-39 are NVFP4 (weight, scale, amax, denominator) and of layers 40-47 MXFP8
    (weight, scale).
    """
    keys = ["model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.norm.weight"]
    keys += ["model.llm.unembed.weight", "model.llm.attn_res.output_query"]
    for layer, layer_type in enumerate(config.layer_types):
        prefix = f"model.llm.layers.{layer}."
        keys += [prefix + f"attn.{name}.weight" for name in ("wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud")]
        keys += [prefix + "attn.gate_attn.weight", prefix + "attn.attn_sink_proj.weight"]
        keys += [
            prefix + "attn.q_l2_clip.scale",
            prefix + "attn.rel_l2_clip.scale",
            prefix + "attn.rel_logits_proj.proj",
        ]
        if layer_type == "hybrid_sliding":
            keys += [prefix + "attn.k_sconv.weight", prefix + "attn.v_sconv.weight", prefix + "attn_norm.weight"]
        else:
            keys += [prefix + f"attn.{kv}_sconv.{name}" for kv in "kv" for name in ("static_w", "weight_proj")]
            keys += [prefix + f"{qkv}_input_norm.weight" for qkv in "qkv"]
        keys += [prefix + "attn_sconv.weight", prefix + "mlp_sconv.weight", prefix + "mlp_norm.weight"]
        keys += [prefix + "mlp.gate.weight", prefix + "mlp.gate.bias"]
        companions = _COMPANIONS if layer < 40 else ("_scale",)
        for weight in ("w13_weight", "w2_weight"):
            keys += [prefix + f"mlp.experts.{weight}{suffix}" for suffix in ("", *companions)]
        keys.append(f"model.llm.attn_res.mlp_queries.{layer}")
        if layer > 0:
            keys.append(f"model.llm.attn_res.attn_queries.{layer - 1}")
    vision = "model.visual.backbone."
    keys += [vision + f"embeddings.{name}" for name in ("class_embedding", "cls_position_embedding")]
    keys += [vision + "embeddings.patch_embedding.weight", vision + "embeddings.position_embedding"]
    keys += [
        vision + "ln_pre.weight",
        vision + "ln_pre.bias",
        "model.visual.proj_in.weight",
        "model.visual.proj_out.weight",
    ]
    for layer in range(vision_layers):
        prefix = f"{vision}layers.{layer}."
        modules = ("ln_1", "ln_2", "mlp.c_fc", "mlp.c_proj", *(f"self_attn.{p}_proj" for p in ("q", "k", "v", "out")))
        keys += [prefix + f"{module}.{kind}" for module in modules for kind in ("weight", "bias")]
        keys += [prefix + "ls_1.gamma", prefix + "ls_2.gamma"]
    return keys


def _is_companion(key: str) -> bool:
    return ".mlp.experts." in key and key.endswith(_COMPANIONS)


def _tiny_model(dtype: str = "bfloat16") -> Inkling2ForConditionalGeneration:
    config = Inkling2Config(text_config=TINY_TEXT_CONFIG, vision_config=None, torch_dtype=dtype)
    model = Inkling2ForConditionalGeneration(config, backend=_BACKEND)
    _randomize(model, seed=2)
    return model


def test_released_index_keys_all_map_and_invert() -> None:
    keys = _released_checkpoint_keys(_released_text_config())
    assert len(keys) == 1770 and len(set(keys)) == 1770

    for key in keys:
        if _is_companion(key):
            continue
        assert native_to_raw_key(raw_to_native_key(key)) == key


def test_released_keys_cover_the_native_model_exactly() -> None:
    config = Inkling2Config(
        text_config=_released_text_config(), vision_config=Inkling2VisionConfig(), torch_dtype="bfloat16"
    )
    with torch.device("meta"):
        model = Inkling2ForConditionalGeneration(config, backend=_BACKEND)
    native_keys = {
        raw_to_native_key(key) for key in _released_checkpoint_keys(config.text_config) if not _is_companion(key)
    }

    assert native_keys == set(model.state_dict())
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    layers = "model.language_model.layers"
    assert shapes[f"{layers}.0.mlp.experts.gate_and_up_projs"] == (384, 4608, 4608)
    assert shapes[f"{layers}.0.mlp.experts.down_projs"] == (384, 2304, 4608)
    assert shapes[f"{layers}.0.self_attn.k_sconv.conv1d.weight"] == (1024, 1, 4)
    assert shapes[f"{layers}.5.self_attn.k_sconv.static_weight"] == (4, 1024)
    assert shapes[f"{layers}.5.self_attn.k_sconv.weight_proj.weight"] == (32, 4608)
    assert shapes[f"{layers}.5.attention_input.query"] == (3, 4608)
    assert shapes[f"{layers}.4.attention_input.query"] == (4608,)
    assert shapes["model.vision_tower.embeddings.patch_embedding.weight"] == (1024, 1200)
    assert shapes["model.vision_tower.merger.linear_2.weight"] == (4608, 4096)


@pytest.mark.parametrize(
    "raw,native",
    [
        ("model.llm.attn_res.attn_queries.0", "model.language_model.layers.1.attention_input.query"),
        ("model.llm.attn_res.attn_queries.46", "model.language_model.layers.47.attention_input.query"),
        ("model.llm.attn_res.mlp_queries.0", "model.language_model.layers.0.mlp_input.query"),
        ("model.llm.attn_res.output_query", "model.language_model.final_input.query"),
        ("model.llm.embed_norm.weight", "model.language_model.embed_tokens.embed_norm.weight"),
        ("model.llm.unembed.weight", "lm_head.weight"),
        ("model.llm.layers.3.mlp.gate.bias", "model.language_model.layers.3.mlp.gate.e_score_correction_bias"),
        (
            "model.llm.layers.3.attn_sconv.weight",
            "model.language_model.layers.3.post_attention_convolution.conv1d.weight",
        ),
        ("model.llm.layers.5.k_input_norm.weight", "model.language_model.layers.5.k_input_layernorm.weight"),
        ("model.llm.layers.5.attn.v_sconv.static_w", "model.language_model.layers.5.self_attn.v_sconv.static_weight"),
        (
            "model.visual.backbone.embeddings.position_embedding",
            "model.vision_tower.embeddings.position_embedding.weight",
        ),
        ("model.visual.backbone.ln_pre.bias", "model.vision_tower.pre_layernorm.bias"),
        ("model.visual.backbone.layers.2.ls_1.gamma", "model.vision_tower.layers.2.lambda_1"),
        ("model.visual.backbone.layers.2.mlp.c_proj.weight", "model.vision_tower.layers.2.mlp.fc2.weight"),
        (
            "model.visual.backbone.layers.2.self_attn.out_proj.bias",
            "model.vision_tower.layers.2.self_attn.out_proj.bias",
        ),
        ("model.visual.proj_out.weight", "model.vision_tower.merger.linear_2.weight"),
    ],
)
def test_key_mapping_matches_the_reference_conversion(raw: str, native: str) -> None:
    assert raw_to_native_key(raw) == native
    assert native_to_raw_key(native) == raw


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(KeyError):
        raw_to_native_key("model.llm.layers.0.attn.q_norm.weight")
    with pytest.raises(KeyError):
        native_to_raw_key("model.language_model.layers.0.attention_input.query")
    adapter = Inkling2StateDictAdapter(Inkling2TextConfig(**TINY_TEXT_CONFIG), dtype=torch.float32)
    with pytest.raises(KeyError):
        adapter.from_hf({"model.mtp.layers.0.weight": torch.zeros(1)})


def test_raw_native_raw_round_trip_is_exact() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    raw = {key: value.detach().clone() for key, value in adapter.to_hf(model.state_dict()).items()}

    native = adapter.from_hf(raw)
    reloaded = _tiny_model()
    reloaded.load_state_dict(native, strict=True)
    exported = adapter.to_hf(reloaded.state_dict())

    assert exported.keys() == raw.keys()
    for key, value in raw.items():
        assert exported[key].dtype == value.dtype, key
        assert exported[key].shape == value.shape, key
        assert torch.equal(exported[key], value), key


def test_expert_layout_is_a_transpose_of_the_released_layout() -> None:
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG)
    experts, intermediate, hidden = config.n_routed_experts, config.moe_intermediate_size, config.hidden_size
    adapter = Inkling2StateDictAdapter(config, dtype=torch.bfloat16)
    w13 = torch.randn(experts, 2 * intermediate, hidden).to(torch.bfloat16)
    w2 = torch.randn(experts, hidden, intermediate).to(torch.bfloat16)
    prefix = "model.llm.layers.0.mlp.experts."
    native = adapter.from_hf({prefix + "w13_weight": w13, prefix + "w2_weight": w2})

    gate_and_up = native["model.language_model.layers.0.mlp.experts.gate_and_up_projs"]
    down = native["model.language_model.layers.0.mlp.experts.down_projs"]
    assert gate_and_up.shape == (experts, hidden, 2 * intermediate)
    # Released rows interleave gate and up: even rows are gate, odd rows are up.
    assert torch.equal(gate_and_up[..., 0::2], w13[:, 0::2, :].transpose(-1, -2))
    assert torch.equal(gate_and_up[..., 1::2], w13[:, 1::2, :].transpose(-1, -2))
    assert torch.equal(down, w2.transpose(-1, -2))

    exported = adapter.to_hf(native)
    assert torch.equal(exported[prefix + "w13_weight"], w13)
    assert torch.equal(exported[prefix + "w2_weight"], w2)


def test_exclude_key_regex_drops_matching_keys() -> None:
    model = _tiny_model()
    exported = model.state_dict_adapter.to_hf(model.state_dict(), exclude_key_regex=r".*experts.*")
    assert exported and not any("experts" in key for key in exported)


def test_quantized_experts_are_dequantized_for_the_local_experts() -> None:
    config = Inkling2TextConfig(**TINY_TEXT_CONFIG)
    experts, rows, hidden = config.n_routed_experts, 2 * config.moe_intermediate_size, config.hidden_size
    generator = torch.Generator().manual_seed(0)
    nvfp4 = {
        "": torch.randint(0, 256, (experts, rows, hidden // 2), dtype=torch.uint8, generator=generator),
        "_scale": torch.rand(experts, rows, hidden // 16, generator=generator).to(torch.float8_e4m3fn),
        "_amax": torch.rand(experts, generator=generator) + 0.5,
        "_outer_scale_denominator": torch.tensor([6.0 * 448.0]),
    }
    mxfp8 = {
        "": torch.randn(experts, hidden, rows // 2, generator=generator).to(torch.float8_e4m3fn),
        "_scale": torch.randint(120, 130, (experts, hidden, rows // 64), dtype=torch.uint8, generator=generator),
    }
    nvfp4_key, mxfp8_key = "model.llm.layers.0.mlp.experts.w13_weight", "model.llm.layers.6.mlp.experts.w2_weight"
    raw = {nvfp4_key + suffix: value for suffix, value in nvfp4.items()}
    raw |= {mxfp8_key + suffix: value for suffix, value in mxfp8.items()}

    native = Inkling2StateDictAdapter(config, dtype=torch.bfloat16).from_hf(raw)

    gate_and_up = native["model.language_model.layers.0.mlp.experts.gate_and_up_projs"]
    expected = dequantize_nvfp4_experts(*nvfp4.values(), dtype=torch.bfloat16).transpose(-1, -2)
    assert gate_and_up.shape == (experts, hidden, rows) and gate_and_up.dtype == torch.bfloat16
    assert torch.equal(gate_and_up, expected)
    down = native["model.language_model.layers.6.mlp.experts.down_projs"]
    expected = dequantize_mxfp8_experts(*mxfp8.values(), dtype=torch.bfloat16).transpose(-1, -2)
    assert down.shape == (experts, rows // 2, hidden) and torch.equal(down, expected)


def test_unsupported_expert_format_is_rejected() -> None:
    adapter = Inkling2StateDictAdapter(Inkling2TextConfig(**TINY_TEXT_CONFIG), dtype=torch.bfloat16)
    key = "model.llm.layers.0.mlp.experts.w2_weight"
    with pytest.raises(ValueError, match="Unsupported expert format"):
        adapter.from_hf({key: torch.zeros(8, 64, 32).to(torch.float8_e4m3fn)})
