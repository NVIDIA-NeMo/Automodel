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

"""Native Transformers Super 3.5 config, vision, and export regression tests."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import NemotronHConfig, RadioConfig
from transformers.models.nemotron_h_omni.configuration_nemotron_h_omni import NemotronH_Omni_Reasoning_V3_Config

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration


def tiny_config() -> NemotronH_Omni_Reasoning_V3_Config:
    """Construct the native schema, including a non-square-capable vision tower."""
    return NemotronH_Omni_Reasoning_V3_Config(
        architectures=["NemotronH_Omni_Reasoning_V3"],
        vision_config=RadioConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            mlp_ratio=2.0,
            patch_size=2,
            image_size=8,
            max_img_size=8,
            num_cls_tokens=2,
            num_registers=0,
            video_temporal_patch_size=2,
            dtype="float32",
            attn_implementation="eager",
        ),
        text_config=NemotronHConfig(
            hidden_size=8,
            vocab_size=32,
            layer_types=["full_attention", "moe"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=16,
            moe_intermediate_size=8,
            moe_shared_expert_intermediate_size=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            num_nextn_predict_layers=0,
            dtype="float32",
            use_cache=False,
        ),
        vision_hidden_size=8,
        projector_hidden_size=16,
        force_image_size=8,
        image_token_id=18,
        video_pruning_rate=0.0,
        dtype="float32",
        attn_implementation="eager",
    )


def test_native_config_without_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native config must construct without consulting checkpoint Python."""

    def forbidden(*args, **kwargs):
        raise AssertionError("remote code was requested")

    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", forbidden)
    model = NemotronOmniForConditionalGeneration(
        tiny_config(),
        backend=BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
        ),
    )
    assert type(model.vision_model).__module__ == "transformers.models.radio.modeling_radio"
    assert model.patch_size == 2
    assert model.vision_projector.vision_final_layernorm is not None
    assert model.language_model.config.num_nextn_predict_layers == 0


@pytest.fixture
def native_model() -> NemotronOmniForConditionalGeneration:
    torch.manual_seed(31)
    model = NemotronOmniForConditionalGeneration(
        tiny_config(),
        backend=BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
        ),
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        model.vision_projector.vision_final_layernorm.weight.copy_(torch.linspace(0.7, 1.3, 8))
        model.vision_projector.vision_final_layernorm.bias.copy_(torch.linspace(-0.1, 0.2, 8))
    return model


def test_packed_images_match_individual_forward_and_gradients(
    native_model: NemotronOmniForConditionalGeneration,
) -> None:
    """Packing cannot leak information or gradients across differently shaped images."""
    model = native_model
    grids = torch.tensor([[4, 2], [2, 4]])
    patches = torch.randn(16, 12, requires_grad=True)
    together = model.extract_feature_packed(patches, grids)
    separate = torch.cat(
        [
            model.extract_feature_packed(patches[:8], grids[:1]),
            model.extract_feature_packed(patches[8:], grids[1:]),
        ]
    )
    torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-5)
    grad_together = torch.autograd.grad(together.square().sum(), patches, retain_graph=True)[0]
    grad_separate = torch.autograd.grad(separate.square().sum(), patches)[0]
    torch.testing.assert_close(grad_together, grad_separate, atol=2e-6, rtol=2e-5)
    # Adversarial second image must not affect the first image's features.
    perturbed = patches.detach().clone()
    perturbed[8:] = torch.randn_like(perturbed[8:]) * 100
    changed = model.extract_feature_packed(perturbed, grids)
    torch.testing.assert_close(together[:2], changed[:2], atol=2e-6, rtol=2e-5)
    assert model.vision_model.training


@pytest.mark.parametrize("grids", [[[4, 4]], [[1, 8]], [[0, 8]]])
def test_invalid_image_grids_fail(native_model: NemotronOmniForConditionalGeneration, grids: list[list[int]]) -> None:
    with pytest.raises(ValueError):
        native_model.extract_feature_packed(torch.randn(8, 12), torch.tensor(grids))


def test_native_placeholder_mismatch_is_not_truncated(native_model: NemotronOmniForConditionalGeneration) -> None:
    with pytest.raises(ValueError, match="placeholders"):
        native_model(
            input_ids=torch.tensor([[18, 2]]),
            pixel_values=torch.randn(8, 12),
            image_grid_hw=torch.tensor([[4, 2]]),
        )


def test_native_image_forward_backward(native_model: NemotronOmniForConditionalGeneration) -> None:
    ids = torch.tensor([[3, 18, 18, 7, 9, 10]])
    pixels = torch.randn(8, 12, requires_grad=True)
    output = native_model(input_ids=ids, pixel_values=pixels, image_grid_hw=torch.tensor([[4, 2]]))
    assert output.logits.shape == (1, 6, 32)
    assert torch.isfinite(output.logits).all()
    output.logits.square().mean().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert pixels.grad.abs().sum() > 0
    norm = native_model.vision_projector.vision_final_layernorm
    assert norm.weight.grad is not None and norm.weight.grad.abs().sum() > 0


def test_export_native_vision_names_and_reload(native_model: NemotronOmniForConditionalGeneration) -> None:
    """Both released and native keys must retain nonidentity final-norm values."""
    adapter = native_model.state_dict_adapter
    state = {k: v.clone() for k, v in native_model.state_dict().items() if not k.startswith("language_model.")}
    exported = adapter.to_hf(dict(state))
    assert "vision_final_layernorm.weight" in exported
    assert "multi_modal_projector.layer_norm.weight" in exported
    assert "vision_model.radio_model.model.blocks.0.attn.qkv.weight" in exported
    assert not any(key.startswith("mlp1.") for key in exported)
    restored = adapter.from_hf(dict(exported))
    assert set(restored) == set(state)
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
    legacy = {
        "vision_projector.vision_final_layernorm.weight": exported["vision_final_layernorm.weight"],
        "vision_projector.vision_final_layernorm.bias": exported["vision_final_layernorm.bias"],
    }
    loaded_legacy = adapter.from_hf(legacy)
    for key, value in loaded_legacy.items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_native_reference_image_features(native_model: NemotronOmniForConditionalGeneration) -> None:
    """Compare against native HF, supplying the documented final norm explicitly."""
    from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3

    config = deepcopy(native_model.config)
    # HF currently ties final-norm construction to MTP. The pretrained vision
    # architecture includes it even when AM disables the language MTP objective.
    config.text_config.num_nextn_predict_layers = 1
    reference = NemotronH_Omni_Reasoning_V3(config).eval()
    # Component check only: these explicit mappings do not certify distributed loading.
    reference.vision_model.load_state_dict(native_model.vision_model.state_dict(), strict=False)
    reference.vision_final_layernorm.load_state_dict(native_model.vision_projector.vision_final_layernorm.state_dict())
    reference.multi_modal_projector.layer_norm.load_state_dict(native_model.vision_projector.norm.state_dict())
    reference.multi_modal_projector.linear_1.load_state_dict(native_model.vision_projector.linear1.state_dict())
    reference.multi_modal_projector.linear_2.load_state_dict(native_model.vision_projector.linear2.state_dict())
    grids = torch.tensor([[4, 2], [2, 4]])
    pixels = torch.randn(16, 12)
    actual = native_model.extract_feature_packed(pixels, grids)
    expected = torch.cat(reference.get_image_features(pixels, grids).pooler_output)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    # A lost checkpoint norm must make this check fail.
    with torch.no_grad():
        reference.vision_final_layernorm.weight.fill_(1)
        reference.vision_final_layernorm.bias.zero_()
    wrong = torch.cat(reference.get_image_features(pixels, grids).pooler_output)
    assert not torch.allclose(actual, wrong, atol=2e-6, rtol=2e-5)


def test_nonpersistent_native_buffer_reinitialized(native_model: NemotronOmniForConditionalGeneration) -> None:
    native_model.vision_model.summary_idxs.fill_(10000)
    native_model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    assert native_model.vision_model.summary_idxs.tolist() == native_model.config.vision_config.summary_idxs
    assert "vision_model.summary_idxs" not in native_model.state_dict()


def test_released_radio_qkv_mapping(native_model: NemotronOmniForConditionalGeneration) -> None:
    """Released fused QKV must populate all three native attention projections."""
    fused = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    output = torch.randn(8, 8)
    state = native_model.state_dict_adapter.from_hf(
        {
            "vision_model.radio_model.model.blocks.0.attn.qkv.weight": fused,
            "vision_model.radio_model.model.blocks.0.attn.proj.weight": output,
        }
    )
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
        key = f"vision_model.encoder.layer.0.attention.{projection}.weight"
        torch.testing.assert_close(state[key], fused[index * 8 : (index + 1) * 8], atol=0, rtol=0)
    torch.testing.assert_close(state["vision_model.encoder.layer.0.attention.o_proj.weight"], output)


def test_native_dcp_projector_aliases(native_model: NemotronOmniForConditionalGeneration) -> None:
    aliases = native_model.state_dict_adapter.checkpoint_key_aliases
    assert aliases["vision_final_layernorm.weight"] == "vision_projector.vision_final_layernorm.weight"
    assert aliases["multi_modal_projector.layer_norm.weight"] == "mlp1.0.weight"


@pytest.mark.parametrize("released_norm_names", [False, True])
def test_native_hf_load_export_and_full_tiny_forward(
    native_model: NemotronOmniForConditionalGeneration, tmp_path: Path, released_norm_names: bool
) -> None:
    """Exercise the real HF converter, not a self-consistent adapter round trip."""
    from safetensors.torch import save_file
    from transformers.models.nemotron_h_omni.modeling_nemotron_h_omni import NemotronH_Omni_Reasoning_V3

    exported = native_model.state_dict_adapter.to_hf(dict(native_model.state_dict()))
    if released_norm_names:
        for suffix in ("weight", "bias"):
            exported[f"vision_projector.vision_final_layernorm.{suffix}"] = exported.pop(
                f"vision_final_layernorm.{suffix}"
            )
    save_file({key: value.contiguous() for key, value in exported.items()}, str(tmp_path / "model.safetensors"))
    config = deepcopy(native_model.config)
    # Known HF construction issue: preserve the pretrained final vision norm.
    # Extra MTP weights are irrelevant to this ordinary forward (no labels/cache).
    config.text_config.num_nextn_predict_layers = 1
    config.save_pretrained(tmp_path)
    reference, info = NemotronH_Omni_Reasoning_V3.from_pretrained(
        tmp_path,
        trust_remote_code=False,
        output_loading_info=True,
        dtype=torch.float32,
        key_mapping={r"^vision_projector\.vision_final_layernorm\.": "vision_final_layernorm."},
    )
    assert not info["unexpected_keys"]
    assert set(info["missing_keys"]) == {"vision_model.summary_idxs"}
    assert reference.vision_model.summary_idxs.tolist() == config.vision_config.summary_idxs
    torch.testing.assert_close(
        reference.vision_final_layernorm.weight,
        native_model.vision_projector.vision_final_layernorm.weight,
        atol=0,
        rtol=0,
    )
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        torch.testing.assert_close(
            reference.vision_model.encoder.layer[0].attention.get_submodule(projection).weight,
            native_model.vision_model.encoder.layer[0].attention.get_submodule(projection).weight,
            atol=0,
            rtol=0,
        )
    native_model.eval()
    reference.eval()
    inputs = {
        "input_ids": torch.tensor([[3, 18, 18, 7, 9, 10]]),
        "pixel_values": torch.randn(8, 12),
        "image_grid_hw": torch.tensor([[4, 2]]),
    }
    actual = native_model(**inputs).logits
    expected = reference(**inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("frames", [3, 4])
def test_native_radio_video_forward_backward(native_model: NemotronOmniForConditionalGeneration, frames: int) -> None:
    """Actual native RADIO temporal projection handles complete and padded tubelets."""
    native_model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    pixels = torch.randn(frames, 3, 8, 4, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[3, 18, 18, 18, 18, 7, 9]])
    output = native_model(input_ids=ids, pixel_values_videos=pixels).logits
    assert output.shape == (1, 7, 32)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert pixels.grad.abs().sum() > 0
    assert native_model.vision_model.embeddings.video_patch_projection.weight.grad is not None


def test_video_pruning_is_not_silently_ignored(native_model: NemotronOmniForConditionalGeneration) -> None:
    native_model.config.video_pruning_rate = 0.7
    with pytest.raises(ValueError, match="video_pruning_rate=0"):
        native_model(input_ids=torch.tensor([[18, 18]]), pixel_values_videos=torch.randn(2, 3, 8, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention 2 requires CUDA")
def test_native_packed_flash_attention_image_boundaries() -> None:
    """Execute packed vision CUDA kernels with adversarial cross-image gradients."""
    pytest.importorskip("flash_attn")
    config = tiny_config()
    config.vision_config.hidden_size = 128
    config.vision_config._attn_implementation = "flash_attention_2"
    config.vision_hidden_size = 128
    model = NemotronOmniForConditionalGeneration(
        config,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch"),
    ).cuda()
    model.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.bfloat16)
    grids = torch.tensor([[4, 2], [2, 4]], device="cuda")
    pixels = torch.randn(16, 12, device="cuda", requires_grad=True)
    together = model.extract_feature_packed(pixels, grids)
    separate = torch.cat(
        [
            model.extract_feature_packed(pixels[:8], grids[:1]),
            model.extract_feature_packed(pixels[8:], grids[1:]),
        ]
    )
    torch.testing.assert_close(together, separate, atol=0, rtol=0)
    gradient = torch.autograd.grad(together[:2].float().square().sum(), pixels)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:8].abs().sum() > 0
    assert torch.count_nonzero(gradient[8:]) == 0
