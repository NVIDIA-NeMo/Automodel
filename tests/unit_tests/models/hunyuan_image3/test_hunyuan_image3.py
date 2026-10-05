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
    _model_dtype,
    build_joint_attention_mask,
    build_moe_config,
)
from nemo_automodel.components.models.hunyuan_image3.rope import (
    apply_rope,
    image_grid_positions_batched,
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
        timestep_token_id=TIMESTEP_ID,
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


@pytest.fixture(scope="module")
def model() -> HunyuanImage3ForCausalMM:
    """One tiny fp32 model shared by the tests that only read it."""
    return _model()


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


@pytest.mark.parametrize("dtype, expected", [("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def test_model_dtype_follows_config(dtype, expected):
    assert _model_dtype(HunyuanImage3Config(torch_dtype=dtype)) == expected


def test_moe_config_matches_release_routing():
    moe = build_moe_config(_config())
    assert moe.score_func == "softmax" and moe.softmax_before_topk and moe.norm_topk_prob
    assert (moe.n_routed_experts, moe.n_activated_experts, moe.moe_inter_dim) == (4, 2, 32)
    assert moe.n_shared_experts == 0  # the release shared expert is the block's own HunyuanSharedMLP
    assert moe.gate_dtype == torch.float32
    assert build_moe_config(_config(), {"aux_loss_coeff": 0.5}).aux_loss_coeff == 0.5


def test_shared_expert_lives_in_the_block(model):
    from nemo_automodel.components.models.hunyuan_image3.layers import HunyuanSharedMLP

    block = model.model.layers["0"]
    assert block.mlp.shared_experts is None and isinstance(block.shared_mlp, HunyuanSharedMLP)
    assert "model.layers.0.shared_mlp.gate_and_up_proj.weight" in model.state_dict()


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


@pytest.mark.runtime_budget(
    60, hard_timeout=120, reason="first use of get_hf_config imports the transformers bridge and model registry"
)
def test_checkpoint_config_resolves_to_local_class(tmp_path):
    """The release config.json (with remote-code auto_map) loads without trust_remote_code."""
    from nemo_automodel._transformers.model_init import get_hf_config, get_is_hf_model

    config = {
        "architectures": ["HunyuanImage3ForCausalMM"],
        "auto_map": {"AutoConfig": "configuration_hunyuan.HunyuanImage3Config"},
        "model_type": "hunyuan_image_3_moe",
        "num_experts": 64,
        "moe_topk": [8, 8],
        "head_dim": 128,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(__import__("json").dumps(config))
    resolved = get_hf_config(str(tmp_path), attn_implementation=None, trust_remote_code=False)
    assert isinstance(resolved, HunyuanImage3Config)
    assert resolved.moe_topk == [8, 8] and resolved.head_dim == 128
    assert _model_dtype(resolved) == torch.bfloat16
    # The diffusion pipeline builds the transformer from the custom implementation when this is False.
    assert not get_is_hf_model(resolved, force_hf=False)


def test_tied_embeddings_are_rejected():
    with pytest.raises(NotImplementedError, match="tie"):
        HunyuanImage3ForCausalMM(_config(tie_word_embeddings=True), backend=_backend())


@pytest.mark.runtime_budget(
    60, hard_timeout=70, reason="first MoE forward/backward compiles the torch.compile'd expert activation kernels"
)
def test_random_init_gives_finite_forward_and_backward():
    torch.manual_seed(0)
    model = HunyuanImage3ForCausalMM(_config(), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), name
    (pred,) = model(_sequence(3, 6)[None], torch.randn(1, 4, 2, 3), torch.tensor([500.0]))
    pred.backward(torch.randn_like(pred))
    assert torch.isfinite(pred).all()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in grads.values())
    # The image path reaches the decoder, the router and the routed experts.
    for name in ("patch_embed.model.0.weight", "model.layers.0.mlp.gate.weight", "final_layer.model.1.2.weight"):
        assert name in grads, name
    assert any(".mlp.experts." in name for name in grads)


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


def _reference_positions(seq_len: int, start: int, token_h: int, token_w: int) -> torch.Tensor:
    """Per-sample (y, x) positions written out the way the release builds them (text, centered grid, text)."""
    num_image = token_h * token_w
    beta_y, beta_x = start + (num_image - token_h) / 2, start + (num_image - token_w) / 2
    positions = [(float(i), float(i)) for i in range(start)]
    positions += [(beta_y + r, beta_x + c) for r in range(token_h) for c in range(token_w)]
    positions += [(float(i), float(i)) for i in range(start + num_image, seq_len)]
    return torch.tensor(positions, dtype=torch.float32).long()


def test_batched_image_positions_match_reference():
    starts = torch.tensor([2, 0, 5])
    batched = image_grid_positions_batched(seq_len=13, image_starts=starts, token_h=2, token_w=3)
    expected = torch.stack([_reference_positions(13, int(s), 2, 3) for s in starts])
    assert torch.equal(batched, expected)


def test_image_grid_positions_center_the_image():
    # 2 text tokens, a 2x3 image, 1 trailing token.
    pos = image_grid_positions_batched(seq_len=9, image_starts=torch.tensor([2]), token_h=2, token_w=3)[0]
    assert pos[:2].tolist() == [[0, 0], [1, 1]]
    # beta_y = 2 + (6 - 2) / 2 = 4, beta_x = 2 + (6 - 3) / 2 = 3.5 -> truncated to 3, 4, 5
    assert pos[2:8].tolist() == [[4, 3], [4, 4], [4, 5], [5, 3], [5, 4], [5, 5]]
    assert pos[8].tolist() == [8, 8]
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


def test_forward_shapes_and_text_logits(model):
    latents = torch.randn(2, 4, 2, 3)
    input_ids = torch.stack([_sequence(5, 6), _sequence(5, 6)])
    (velocity,) = model(input_ids, latents, torch.tensor([10.0, 900.0]))
    assert velocity.shape == latents.shape
    assert model(input_ids, latents, torch.tensor([10.0, 900.0]), return_dict=True)["sample"].shape == latents.shape
    logits = model.forward_text(input_ids[:, :5])
    assert logits.shape == (2, 5, 300) and logits.dtype == torch.float32
    assert model.get_input_embeddings() is model.model.embed_tokens
    assert model.get_output_embeddings() is model.lm_head


def test_forward_is_invariant_to_right_padding_and_batching(model):
    latents = torch.randn(2, 4, 2, 3)
    short, long = _sequence(3, 6), _sequence(7, 6)
    timesteps = torch.tensor([250.0, 700.0])
    (alone,) = model(short[None], latents[:1], timesteps[:1])
    padded = torch.full((2, len(long)), PAD_ID)
    padded[0, : len(short)] = short
    padded[1] = long
    (batched,) = model(padded, latents, timesteps, valid_lengths=torch.tensor([len(short), len(long)]))
    torch.testing.assert_close(batched[:1], alone, rtol=1e-4, atol=1e-5)


def test_forward_rejects_wrong_image_token_count(model):
    with pytest.raises(ValueError, match="image tokens"):
        model(_sequence(3, 5)[None], torch.randn(1, 4, 2, 3), torch.tensor([1.0]))


def test_forward_rejects_malformed_image_spans(model):
    latents, t = torch.randn(1, 4, 2, 3), torch.tensor([1.0])
    no_timestep = _sequence(3, 6)
    no_timestep[3] = 7  # the slot before the image is not <timestep>
    with pytest.raises(ValueError, match="<timestep>"):
        model(no_timestep[None], latents, t)
    at_start = torch.cat([torch.full((6,), IMAGE_ID), torch.tensor([3, 3])])  # nothing before the image
    with pytest.raises(ValueError, match="preceded by the <timestep> token"):
        model(at_start[None], latents, t)
    split = _sequence(3, 6)  # [p p p <timestep> img*6 <eoi>]
    split[6], split[10] = 5, IMAGE_ID  # still six image tokens, but not contiguous
    with pytest.raises(ValueError, match="contiguous span"):
        model(split[None], latents, t)


def test_timestep_changes_prediction(model):
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


def test_to_hf_produces_release_keys_and_fused_up_gate_layout(model):
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


def test_hf_round_trip_and_unused_release_modules(model):
    native = {k: v.clone() for k, v in model.state_dict().items()}
    hf = model.state_dict_adapter.to_hf(native)
    hf["vae.encoder.conv_in.weight"] = torch.zeros(1)
    hf["vision_model.embeddings.weight"] = torch.zeros(1)
    hf["vision_aligner.layers.0.weight"] = torch.zeros(1)
    restored = model.state_dict_adapter.from_hf(hf)
    assert set(restored) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(restored[key], value, msg=key)


def test_from_hf_rejects_wrong_expert_shape(model):
    hf = model.state_dict_adapter.to_hf(model.state_dict())
    hf["model.layers.0.mlp.experts.0.gate_and_up_proj.weight"] = torch.zeros(10, 64)
    with pytest.raises(ValueError, match="expected 64 rows"):
        model.state_dict_adapter.from_hf(hf)


@pytest.mark.parametrize("registered_in_place", [False, True])
def test_checkpoint_load_destinations_fill_model_weights(registered_in_place):
    """DCP writes the released fused expert tensors into host buffers; ``from_hf`` moves them into the weights.

    With ``registered_in_place`` the grouped keys are marked as loaded in place (what the mixin does on CUDA), so
    ``from_hf`` must leave them out of its result and report them as view-loaded instead of rebuilding them.
    """
    target = _model(seed=0)
    source = _model(seed=1)
    adapter = target.state_dict_adapter
    source_hf = source.state_dict_adapter.to_hf(source.state_dict())
    destinations = adapter.to_hf(target.state_dict(), for_checkpoint_load=True)
    assert set(destinations) == set(source_hf)
    weight_storages = {
        block.mlp.experts.gate_and_up_projs.untyped_storage().data_ptr() for block in target.model.layers.values()
    }
    fused_keys = [k for k in destinations if k.endswith(".gate_and_up_proj.weight") and ".experts." in k]
    assert len(fused_keys) == 2 * 4  # layers x experts
    for key in fused_keys:
        buffer = destinations[key]
        assert buffer.device.type == "cpu" and buffer.shape == (64, 64)  # [2 * expert_hidden, hidden]
        assert buffer.untyped_storage().data_ptr() not in weight_storages
    if registered_in_place:
        for layer in ("0", "1"):
            adapter._register_inplace_loaded_key(f"model.layers.{layer}.mlp.experts.gate_and_up_projs", None)
    with torch.no_grad():  # what DCP does: write the checkpoint into the destinations
        for key, value in destinations.items():
            value.copy_(source_hf[key])
    native = adapter.from_hf(destinations)
    assert not adapter._fused_load_destinations
    grouped_keys = {f"model.layers.{layer}.mlp.experts.gate_and_up_projs" for layer in ("0", "1")}
    if registered_in_place:
        assert not grouped_keys & set(native)
        assert grouped_keys <= adapter.view_loaded_native_keys
    else:
        assert grouped_keys <= set(native)
    target.load_state_dict(native, strict=False)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value, msg=key)


def test_to_hf_honors_exclude_regex(model):
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


@pytest.mark.parametrize("v4_compatible", [False, True])
def test_exported_lora_reloads_all_release_projections(tmp_path, v4_compatible):
    """The exported targets and weights reload into PEFT with the release's module names."""
    from peft import LoraConfig, PeftModel
    from safetensors.torch import save_file

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.checkpoint.addons import _extract_target_modules

    source = _model()
    release_layout = _model()
    # Only exercise projections, using the released shared-expert names and the same base weights.
    for block in release_layout.model.layers.values():
        block.mlp.shared_mlp = block._modules.pop("shared_mlp")
    config = PeftConfig(
        dim=4,
        alpha=8,
        use_memory_efficient_lora=False,
        target_modules=[
            "*.self_attn.qkv_proj",
            "*.self_attn.o_proj",
            "*.shared_mlp.gate_and_up_proj",
            "*.shared_mlp.down_proj",
        ],
    )
    assert apply_lora_to_linear_modules(source, config) == 8
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    targets = _extract_target_modules(source, v4_compatible=v4_compatible)
    expected = {
        f"model.layers.{layer}.{projection}"
        for layer in range(2)
        for projection in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.shared_mlp.gate_and_up_proj",
            "mlp.shared_mlp.down_proj",
        )
    }
    assert set(targets) == expected
    weights = source.state_dict_adapter.to_hf(
        {f"base_model.model.{key}": value for key, value in source.state_dict().items() if "lora_" in key},
        v4_compatible=v4_compatible,
    )
    save_file(weights, str(tmp_path / "adapter_model.safetensors"))
    LoraConfig(r=4, lora_alpha=8, target_modules=targets).save_pretrained(tmp_path)
    restored = PeftModel.from_pretrained(release_layout, tmp_path).eval()
    source.eval()
    for target in targets:
        native_name = target.replace(".mlp.shared_mlp.", ".shared_mlp.")
        original = source.get_submodule(native_name)
        loaded = restored.base_model.model.get_submodule(target)
        torch.testing.assert_close(loaded.lora_A["default"].weight, original.lora_A.weight)
        torch.testing.assert_close(loaded.lora_B["default"].weight, original.lora_B.weight)
        inputs = torch.randn(2, 3, original.in_features)
        torch.testing.assert_close(loaded(inputs), original(inputs))


def _run_sharded_checkpoint_round_trip(rank: int, init_file: str, checkpoint_dir: str) -> None:
    """Exercise real DCP loading with both expert and inner-dimension sharding on CPU."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=4)
    try:
        for axis, placement in (("ep_shard", Shard(1)), ("ep_replicate", Replicate())):
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=(axis, "ep"))
            adapter = _model().state_dict_adapter
            prefix = "model.layers.0.mlp.experts"
            native = {
                f"{prefix}.gate_and_up_projs": torch.arange(4 * 64 * 64).reshape(4, 64, 64).float(),
                f"{prefix}.down_projs": torch.arange(4 * 32 * 64).reshape(4, 32, 64).float(),
            }
            released = adapter.to_hf(native)
            path = f"{checkpoint_dir}/{axis}"
            dcp.save(released, checkpoint_id=path)
            sharded = {
                key: distribute_tensor(value.clone(), mesh, [placement, Shard(0)]) for key, value in native.items()
            }
            exported = adapter.to_hf(sharded)
            for key, value in exported.items():
                assert isinstance(value, DTensor)
                # Both projections transpose the sharded native axis to the final checkpoint axis.
                assert value.placements == (placement,)
                torch.testing.assert_close(value.full_tensor(), released[key])
            destinations = adapter.to_hf(sharded, for_checkpoint_load=True)
            assert not adapter._fused_load_destinations
            for value in destinations.values():
                value.to_local().zero_()
            dcp.load(destinations, checkpoint_id=path)
            restored = adapter.from_hf(destinations, device_mesh=mesh)
            assert set(restored) == set(native)
            for key, value in restored.items():
                assert value.placements == (placement, Shard(0))
                torch.testing.assert_close(value.full_tensor(), native[key])
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(100, hard_timeout=120, reason="four CPU workers exercise real DCP and DTensor collectives")
def test_sharded_checkpoint_round_trip(tmp_path):
    torch.multiprocessing.spawn(
        _run_sharded_checkpoint_round_trip,
        args=(str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint")),
        nprocs=4,
        join=True,
    )


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


def test_adapter_builds_padded_sequences(model):
    adapter = HunyuanImage3Adapter(image_token_id=IMAGE_ID, pad_token_id=PAD_ID)
    inputs = adapter.prepare_inputs(_context())
    ids = inputs["input_ids"]
    assert ids.shape == (2, 4 + 6 + 1)
    assert ids[0].tolist() == [1, 2, 3, TIMESTEP_ID] + [IMAGE_ID] * 6 + [3]
    assert ids[1].tolist() == [4, TIMESTEP_ID] + [IMAGE_ID] * 6 + [3] + [PAD_ID] * 2
    assert inputs["valid_lengths"].tolist() == [11, 9]
    assert inputs["timestep"].dtype == torch.float32

    pred = adapter.forward(model, inputs)
    assert pred.shape == (2, 4, 2, 3)


def test_adapter_uses_unconditional_prompt_when_dropped():
    adapter = HunyuanImage3Adapter(image_token_id=IMAGE_ID, pad_token_id=PAD_ID)
    ids = adapter.prepare_inputs(_context(cfg_dropout_prob=1.0))["input_ids"]
    assert ids[0, :4].tolist() == [9, 9, 9, TIMESTEP_ID]
    assert ids[1, :2].tolist() == [9, TIMESTEP_ID]


def test_adapter_cfg_dropout_follows_torch_seed():
    adapter = HunyuanImage3Adapter(image_token_id=IMAGE_ID, pad_token_id=PAD_ID)

    def pattern(seed: int) -> list[list[int]]:
        torch.manual_seed(seed)
        return [adapter.prepare_inputs(_context(cfg_dropout_prob=0.5))["input_ids"][:, 0].tolist() for _ in range(8)]

    assert pattern(1) == pattern(1)
    assert pattern(1) != pattern(2)


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


def test_adapter_batch_keys_match_the_dataset():
    from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import PROMPT_TOKEN_ID_KEYS
    from nemo_automodel.components.models.hunyuan_image3 import flow_adapter

    keys = (flow_adapter.PROMPT_IDS_KEY, flow_adapter.UNCOND_PROMPT_IDS_KEY, flow_adapter.PROMPT_SUFFIX_IDS_KEY)
    assert keys == PROMPT_TOKEN_ID_KEYS
