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

"""CPU tests of the HunyuanImage-3.0 model, its state dict adapter and its flow-matching adapter."""

from __future__ import annotations

import re

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hunyuan_image3.config import HunyuanImage3Config, per_layer
from nemo_automodel.components.models.hunyuan_image3.flow_adapter import HunyuanImage3Adapter
from nemo_automodel.components.models.hunyuan_image3.model import (
    HunyuanImage3ForCausalMM,
    build_joint_attention_mask,
    build_moe_config,
)
from nemo_automodel.components.models.hunyuan_image3.rope import (
    apply_rope,
    image_grid_positions,
    rope_cos_sin,
    text_positions,
)

IMAGE_ID = 290
PAD_ID = 299
TIMESTEP_ID = 280


def _backend(**overrides) -> BackendConfig:
    fields = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        gate_precision="float32",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
    )
    fields.update(overrides)
    return BackendConfig(**fields)


def _config(**overrides) -> HunyuanImage3Config:
    fields = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=32,
        moe_intermediate_size=[32, 32],
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_head_dim=16,
        num_experts=4,
        moe_topk=[2, 2],
        num_shared_expert=[1, 1],
        patch_embed_hidden_dim=32,
        image_token_id=IMAGE_ID,
        pad_token_id=PAD_ID,
        vae={"latent_channels": 4},
        torch_dtype="float32",
    )
    fields.update(overrides)
    return HunyuanImage3Config(**fields)


def _model(seed: int = 0, **config_overrides) -> HunyuanImage3ForCausalMM:
    torch.manual_seed(seed)
    model = HunyuanImage3ForCausalMM(_config(**config_overrides), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    # Exercise the full computation: the released UNet output conv is zero-initialized only in its own init.
    return model.eval()


def _sequence(prompt_len: int, num_image: int) -> torch.Tensor:
    prompt = torch.randint(0, 200, (prompt_len,))
    return torch.cat([prompt, torch.tensor([TIMESTEP_ID]), torch.full((num_image,), IMAGE_ID), torch.tensor([3])])


# ---------------------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------------------


def test_config_defaults_match_release_and_head_dim_alias():
    config = HunyuanImage3Config(head_dim=128)
    assert config.model_type == "hunyuan_image_3_moe"
    assert config.head_dim == config.attention_head_dim == 128
    assert config.latent_channels == 32
    assert per_layer([3, 5], 1) == 5 and per_layer(7, 4) == 7


def test_moe_config_matches_release_routing():
    moe = build_moe_config(_config())
    assert moe.score_func == "softmax" and moe.softmax_before_topk and moe.norm_topk_prob
    assert (moe.n_routed_experts, moe.n_activated_experts, moe.moe_inter_dim) == (4, 2, 32)
    assert moe.n_shared_experts == 0  # attached separately as HunyuanSharedMLP
    assert moe.gate_dtype == torch.float32
    assert build_moe_config(_config(), {"aux_loss_coeff": 0.5}).aux_loss_coeff == 0.5


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"moe_topk": [2, 1]}, "moe_topk differs"),
        ({"hidden_act": "gelu"}, "SwiGLU"),
        ({"use_mixed_mlp_moe": False}, "shared expert"),
    ],
)
def test_moe_config_rejects_unsupported_layouts(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        build_moe_config(_config(**overrides))


def test_model_rejects_unsupported_options():
    with pytest.raises(NotImplementedError, match="patch_size=1"):
        HunyuanImage3ForCausalMM(_config(patch_size=2), backend=_backend())
    with pytest.raises(ValueError, match="both moe_config and moe_overrides"):
        config = _config()
        HunyuanImage3ForCausalMM(config, build_moe_config(config), _backend(), moe_overrides={"aux_loss_coeff": 0.1})


# ---------------------------------------------------------------------------------------------------------------
# RoPE and attention mask
# ---------------------------------------------------------------------------------------------------------------


def test_text_rope_equals_standard_1d_rope():
    head_dim, seq = 16, 7
    cos, sin = rope_cos_sin(text_positions(seq)[None], head_dim)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.arange(seq).float()[:, None] * inv_freq[None]
    angles = torch.cat([angles, angles], dim=-1)
    torch.testing.assert_close(cos[0], angles.cos())
    torch.testing.assert_close(sin[0], angles.sin())


def test_image_grid_positions_center_the_image():
    # 2 text tokens, a 2x3 image, 1 trailing token.
    pos = image_grid_positions(seq_len=9, image_start=2, token_h=2, token_w=3)
    assert pos[:2].tolist() == [[0, 0], [1, 1]]
    # beta_y = 2 + (6 - 2) / 2 = 4, beta_x = 2 + (6 - 3) / 2 = 3.5 -> truncated to 3, 4, 5
    assert pos[2:8].tolist() == [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
    assert pos[8].tolist() == [8, 8]
    with pytest.raises(ValueError, match="does not fit"):
        image_grid_positions(seq_len=5, image_start=2, token_h=2, token_w=3)
    with pytest.raises(ValueError, match="divisible by 4"):
        rope_cos_sin(pos, head_dim=6)


def test_apply_rope_preserves_norm_per_pair():
    x = torch.randn(1, 2, 5, 8)
    cos, sin = rope_cos_sin(text_positions(5)[None], 8)
    out = apply_rope(x, cos, sin)
    first, second = x.chunk(2, -1)
    out_first, out_second = out.chunk(2, -1)
    torch.testing.assert_close(first**2 + second**2, out_first**2 + out_second**2)


def test_joint_attention_mask():
    mask = build_joint_attention_mask(
        seq_len=6, image_starts=torch.tensor([1]), num_image_tokens=3, valid_lengths=torch.tensor([5])
    )[0, 0]
    assert mask.shape == (6, 6)
    assert mask[1, 3] and mask[3, 1]  # image tokens see each other in both directions
    assert not mask[0, 1]  # text before the image is causal
    assert mask[4, :5].all()  # the token after the image sees everything before it
    assert not mask[:5, 5].any()  # the padded key is hidden from real tokens
    assert mask[5, 5]  # the padded query attends to itself


# ---------------------------------------------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------------------------------------------


def test_forward_shapes_and_text_logits():
    model = _model()
    latents = torch.randn(2, 4, 2, 3)
    input_ids = torch.stack([_sequence(5, 6), _sequence(5, 6)])
    (velocity,) = model(input_ids, latents, torch.tensor([10.0, 900.0]))
    assert velocity.shape == latents.shape
    assert model(input_ids, latents, torch.tensor([10.0, 900.0]), return_dict=True)["sample"].shape == latents.shape
    logits = model.forward_text(input_ids[:, :5])
    assert logits.shape == (2, 5, 300) and logits.dtype == torch.float32
    assert model.get_input_embeddings() is model.model.embed_tokens
    assert model.get_output_embeddings() is model.lm_head


def test_forward_is_invariant_to_right_padding_and_batching():
    model = _model()
    latents = torch.randn(2, 4, 2, 3)
    short, long = _sequence(3, 6), _sequence(7, 6)
    timesteps = torch.tensor([250.0, 700.0])
    (alone,) = model(short[None], latents[:1], timesteps[:1])
    padded = torch.full((2, len(long)), PAD_ID)
    padded[0, : len(short)] = short
    padded[1] = long
    (batched,) = model(padded, latents, timesteps, valid_lengths=torch.tensor([len(short), len(long)]))
    torch.testing.assert_close(batched[:1], alone, rtol=1e-4, atol=1e-5)


def test_forward_rejects_wrong_image_token_count():
    model = _model()
    with pytest.raises(ValueError, match="image tokens"):
        model(_sequence(3, 5)[None], torch.randn(1, 4, 2, 3), torch.tensor([1.0]))


def test_timestep_changes_prediction():
    model = _model()
    ids, latents = _sequence(3, 6)[None], torch.randn(1, 4, 2, 3)
    (a,) = model(ids, latents, torch.tensor([10.0]))
    (b,) = model(ids, latents, torch.tensor([990.0]))
    assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------------------------------------------
# State dict adapter
# ---------------------------------------------------------------------------------------------------------------


def _expected_hf_keys(config: HunyuanImage3Config) -> set[str]:
    keys = {"model.wte.weight", "model.ln_f.weight", "lm_head.weight"}
    for layer in range(config.num_hidden_layers):
        p = f"model.layers.{layer}"
        keys |= {
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.self_attn.qkv_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.query_layernorm.weight",
            f"{p}.self_attn.key_layernorm.weight",
            f"{p}.mlp.gate.wg.weight",
            f"{p}.mlp.shared_mlp.gate_and_up_proj.weight",
            f"{p}.mlp.shared_mlp.down_proj.weight",
        }
        for expert in range(config.num_experts):
            keys |= {
                f"{p}.mlp.experts.{expert}.gate_and_up_proj.weight",
                f"{p}.mlp.experts.{expert}.down_proj.weight",
            }
    return keys


def test_to_hf_produces_release_keys_and_fused_up_gate_layout():
    model = _model()
    config = model.config
    hf = model.state_dict_adapter.to_hf(model.state_dict())
    image_keys = {k for k in hf if re.match(r"(patch_embed|final_layer|time_embed|time_embed_2|timestep_emb)\.", k)}
    assert set(hf) - image_keys == _expected_hf_keys(config)
    assert "patch_embed.model.1.skip_connection.weight" in image_keys
    assert "final_layer.model.1.2.weight" in image_keys

    inter = config.moe_intermediate_size[0]
    grouped = model.model.layers["0"].mlp.experts.gate_and_up_projs  # [E, H, 2I], columns [gate | up]
    fused = hf["model.layers.0.mlp.experts.1.gate_and_up_proj.weight"]  # [2I, H], rows [up; gate]
    torch.testing.assert_close(fused[:inter], grouped[1, :, inter:].T)
    torch.testing.assert_close(fused[inter:], grouped[1, :, :inter].T)
    torch.testing.assert_close(
        hf["model.layers.0.mlp.experts.1.down_proj.weight"], model.model.layers["0"].mlp.experts.down_projs[1].T
    )


def test_hf_round_trip_and_unused_release_modules():
    model = _model()
    native = {k: v.clone() for k, v in model.state_dict().items()}
    hf = model.state_dict_adapter.to_hf(native)
    hf["vae.encoder.conv_in.weight"] = torch.zeros(1)
    hf["vision_model.embeddings.weight"] = torch.zeros(1)
    hf["vision_aligner.layers.0.weight"] = torch.zeros(1)
    restored = model.state_dict_adapter.from_hf(hf)
    assert set(restored) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(restored[key], value, msg=key)


def test_from_hf_rejects_wrong_expert_shape():
    model = _model()
    hf = model.state_dict_adapter.to_hf(model.state_dict())
    hf["model.layers.0.mlp.experts.0.gate_and_up_proj.weight"] = torch.zeros(10, 64)
    with pytest.raises(ValueError, match="expected 64 rows"):
        model.state_dict_adapter.from_hf(hf)


def test_to_hf_honors_exclude_regex():
    model = _model()
    hf = model.state_dict_adapter.to_hf(model.state_dict(), exclude_key_regex=r".*experts.*")
    assert not any("experts" in key for key in hf)
    assert "model.wte.weight" in hf


def test_loaded_release_weights_reproduce_outputs():
    """A second model loaded from the first one's HF state dict computes identical velocities."""
    source = _model(seed=0)
    target = _model(seed=1)
    native = target.state_dict_adapter.from_hf(source.state_dict_adapter.to_hf(source.state_dict()))
    target.load_state_dict(native)
    ids, latents, t = _sequence(4, 6)[None], torch.randn(1, 4, 2, 3), torch.tensor([333.0])
    torch.testing.assert_close(target(ids, latents, t)[0], source(ids, latents, t)[0])


# ---------------------------------------------------------------------------------------------------------------
# Flow-matching adapter
# ---------------------------------------------------------------------------------------------------------------


def _context(cfg_dropout_prob: float = 0.0) -> FlowMatchingContext:
    batch = {
        "prompt_input_ids": [torch.tensor([1, 2, 3, TIMESTEP_ID]), torch.tensor([4, TIMESTEP_ID])],
        "uncond_prompt_input_ids": [torch.tensor([9, 9, 9, TIMESTEP_ID]), torch.tensor([9, TIMESTEP_ID])],
        "prompt_suffix_ids": [torch.tensor([3]), torch.tensor([3])],
    }
    latents = torch.randn(2, 4, 2, 3)
    return FlowMatchingContext(
        noisy_latents=latents,
        latents=latents,
        timesteps=torch.tensor([100.0, 800.0]),
        sigma=torch.tensor([0.1, 0.8]),
        task_type="t2i",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
        cfg_dropout_prob=cfg_dropout_prob,
    )


def test_adapter_builds_padded_sequences():
    adapter = HunyuanImage3Adapter(image_token_id=IMAGE_ID, pad_token_id=PAD_ID)
    inputs = adapter.prepare_inputs(_context())
    ids = inputs["input_ids"]
    assert ids.shape == (2, 4 + 6 + 1)
    assert ids[0].tolist() == [1, 2, 3, TIMESTEP_ID] + [IMAGE_ID] * 6 + [3]
    assert ids[1].tolist() == [4, TIMESTEP_ID] + [IMAGE_ID] * 6 + [3] + [PAD_ID] * 2
    assert inputs["valid_lengths"].tolist() == [11, 9]
    assert inputs["timestep"].dtype == torch.float32

    model = _model()
    pred = adapter.forward(model, inputs)
    assert pred.shape == (2, 4, 2, 3)


def test_adapter_uses_unconditional_prompt_when_dropped():
    adapter = HunyuanImage3Adapter(image_token_id=IMAGE_ID, pad_token_id=PAD_ID)
    ids = adapter.prepare_inputs(_context(cfg_dropout_prob=1.0))["input_ids"]
    assert ids[0, :4].tolist() == [9, 9, 9, TIMESTEP_ID]
    assert ids[1, :2].tolist() == [9, TIMESTEP_ID]


def test_adapter_validates_batch():
    adapter = HunyuanImage3Adapter()
    context = _context()
    del context.batch["prompt_suffix_ids"]
    with pytest.raises(KeyError, match="prompt_suffix_ids"):
        adapter.prepare_inputs(context)
    context = _context()
    context.noisy_latents = torch.randn(2, 4, 1, 2, 3)
    with pytest.raises(ValueError, match="4D latents"):
        adapter.prepare_inputs(context)
