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

"""D1-3B tied readout and causal image/text gradient contracts."""

import inspect

import pytest
import torch
from transformers import Lfm2VlConfig
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlForConditionalGeneration as HFLfm2VlForConditionalGeneration,
)

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.model_init import _init_model
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.conversion_mapping import get_combined_key_mapping
from nemo_automodel.components.models.d1.causal import D1CausalRotaryEmbedding
from nemo_automodel.components.models.d1.decision_head import D1TokenReadoutConfig
from nemo_automodel.components.models.d1.records import D1TextExample
from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration


def tiny_config():
    return Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        image_token_id=31,
        tie_word_embeddings=True,
        projector_hidden_size=64,
        projector_use_layernorm=False,
        projector_bias=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": False,
            "pad_token_id": 0,
            "use_cache": False,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 16,
            "vision_use_head": False,
        },
    )


def test_tie_identity_and_native_registry():
    model = Lfm2VlForConditionalGeneration(tiny_config())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert ModelRegistry.get_model_cls_from_model_arch("Lfm2VlForConditionalGeneration") is type(model)
    model.to(torch.bfloat16)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.model.language_model.rotary_emb.inv_freq.dtype == torch.float32


@pytest.mark.parametrize("model_reference", ["modeling_d1.D1Model", "LiquidAI/d1-3B--modeling_d1.D1Model"])
def test_loader_selects_native_d1_only_for_explicit_publisher_mapping(model_reference: str) -> None:
    ordinary = tiny_config()
    assert ModelRegistry.resolve_custom_model_cls("Lfm2VlForConditionalGeneration", ordinary) is None
    publisher = Lfm2VlConfig(**ordinary.to_dict(), auto_map={"AutoModel": model_reference})
    assert (
        ModelRegistry.resolve_custom_model_cls("Lfm2VlForConditionalGeneration", publisher)
        is Lfm2VlForConditionalGeneration
    )


@pytest.mark.parametrize("rope_type", ["linear", "yarn"])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_ordinary_scaled_rotary_preserves_hf_factors_after_dtype_moves(rope_type: str, device: str) -> None:
    config = tiny_config()
    config.text_config.rope_parameters = {
        "rope_type": rope_type,
        "rope_theta": 1000000.0,
        "factor": 4.0,
        **({"attention_factor": 1.5} if rope_type == "yarn" else {}),
    }
    model = Lfm2VlForConditionalGeneration(config).eval()
    rotary = model.model.language_model.rotary_emb
    reference = Lfm2RotaryEmbedding(config.text_config, device=device)
    positions = torch.tensor([[0, 17, 4095]], device=device)
    expected = reference(torch.empty(1, 3, config.text_config.hidden_size, device=device), positions)
    for dtype in (torch.float32, torch.bfloat16, torch.float32):
        model.to(device=device, dtype=dtype)
        assert rotary.inv_freq.dtype == torch.float32
        assert rotary.attention_scaling == reference.attention_scaling
        torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
        torch.testing.assert_close(rotary.original_inv_freq, reference.original_inv_freq, rtol=0, atol=0)
        actual = rotary(torch.empty(1, 3, config.text_config.hidden_size, device=device, dtype=dtype), positions)
        for factors, expected_factors in zip(actual, expected, strict=True):
            torch.testing.assert_close(factors, expected_factors, rtol=0, atol=0)


def test_checkpoint_conversion_preserves_released_parameter_names():
    model = Lfm2VlForConditionalGeneration(tiny_config())
    mapping = get_combined_key_mapping("lfm2_vl", {}, model=model)
    names = set(model.state_dict())
    converted = {mapping(name) if callable(mapping) else mapping.get(name, name) for name in names}
    assert converted == names


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_released_width_rotary_matches_device_local_frequencies_after_dtype_moves(device):
    config = tiny_config().text_config
    config.hidden_size = 256
    rotary = D1CausalRotaryEmbedding(config).to(device=device, dtype=torch.bfloat16)
    width = config.hidden_size // config.num_attention_heads
    assert width == 64
    expected = 1.0 / config.rope_parameters["rope_theta"] ** (torch.arange(0, width, 2, device=device).float() / width)
    for dtype in (torch.bfloat16, torch.float32):
        rotary.to(dtype=dtype)
        assert rotary.inv_freq.dtype == torch.float32
        torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
        torch.testing.assert_close(rotary.original_inv_freq, expected, rtol=0, atol=0)
        positions = torch.tensor([[0, 17, 159, 4095, 32767], [1, 18, 160, 4096, 32766]], device=device)
        cosine, sine = rotary(torch.empty(2, 5, config.hidden_size, device=device, dtype=dtype), positions)
        angles = positions.float().unsqueeze(-1) * expected
        angles = torch.cat((angles, angles), dim=-1)
        assert cosine.dtype == sine.dtype == torch.float32
        torch.testing.assert_close(cosine, angles.cos(), rtol=0, atol=0)
        torch.testing.assert_close(sine, angles.sin(), rtol=0, atol=0)


def test_causal_future_tokens_do_not_change_past():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lfm2VlForConditionalGeneration(tiny_config()).eval().to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    changed = torch.tensor([[1, 2, 3, 7, 8]], device=device)
    a = model(ids, use_cache=False).logits
    b = model(changed, use_cache=False).logits
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(a[:, 4], b[:, 4])


def test_image_context_has_gradients_and_frozen_adapter_preserves_tie():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lfm2VlForConditionalGeneration(tiny_config()).eval().to(device)
    pixels = torch.randn(1, 16, 768, requires_grad=True, device=device)
    ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device=device)
    output = model(
        ids,
        pixel_values=pixels,
        spatial_shapes=torch.tensor([[4, 4]], device=device),
        pixel_attention_mask=torch.ones(1, 16, dtype=torch.int32, device=device),
        use_cache=False,
    )
    output.logits[:, -1, :3].sum().backward()
    assert pixels.grad.abs().sum() > 0
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    adapter = D1TokenReadoutConfig(64, (2, 3, 4)).build(model.get_output_embeddings().weight)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
    torch.nn.functional.cross_entropy(adapter(hidden, ((0, 1), (2,))), torch.tensor([1], device=device)).backward()
    assert adapter.delta_weight.grad.abs().sum() > 0
    assert model.get_output_embeddings().weight.grad is None
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_ordinary_untied_checkpoint_keeps_independent_vocabulary_head():
    config = tiny_config()
    config.tie_word_embeddings = False
    model = Lfm2VlForConditionalGeneration(config)
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    model.tie_weights()
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    assert torch.isfinite(model.get_output_embeddings().weight).all()


def test_causal_text_readout_preserves_full_model_alias_scores_and_head_gradients():
    torch.manual_seed(145)
    model = Lfm2VlForConditionalGeneration(tiny_config()).eval().requires_grad_(False)
    examples = [
        D1TextExample("record-a", (1, 2, 3), "flag", ("true", "false"), (), ((7, 4), (9,)), 0, 1.0),
        D1TextExample("record-b", (5, 6), "flag", ("false", "true"), (), ((9,), (4, 7)), 0, 1.0),
    ]
    readout = model.prepare_text_readout(examples)
    features, expected = [], []
    for ex in examples:
        ids = torch.tensor([ex.input_ids])
        with torch.no_grad():
            features.append(readout.extract_features(ids, torch.ones_like(ids, dtype=torch.bool)))
            vocabulary_logits = model(ids, use_cache=False).logits[0, -1]
            expected.append(torch.stack([vocabulary_logits[list(group)].amax() for group in ex.alias_token_ids]))
    actual = readout.score(features, examples, head=readout.head)
    for logits, full_model_logits in zip(actual, expected, strict=True):
        torch.testing.assert_close(logits, full_model_logits, atol=2e-6, rtol=2e-5)
    torch.nn.functional.cross_entropy(torch.stack(actual), torch.tensor([0, 1])).backward()
    assert readout.head.delta_weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    with pytest.raises(ValueError, match="counts"):
        readout.score(features, examples[:1], head=readout.head)
    with pytest.raises(ValueError, match="all-true"):
        readout.extract_features(torch.tensor([[1, 2]]), torch.tensor([[True, False]]))


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selected_model_generates_like_hf_for_ordinary_lfm2_vl(
    tied: bool, use_cache: bool, with_image: bool, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    """The actual default loader preserves text/image generation and cache behavior."""
    torch.manual_seed(731)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # The installed CUDA extension can override HF's CPU-capable reference
        # functions. Select HF's own PyTorch implementations for this CPU case.
        for name in ("causal_conv1d_fn", "causal_conv1d_update"):
            monkeypatch.setattr(modeling_lfm2, name, inspect.unwrap(getattr(modeling_lfm2, name)))
    config = tiny_config()
    config.tie_word_embeddings = tied
    is_custom, model = _init_model(
        cls=NeMoAutoModelForImageTextToText,
        pretrained_model_name_or_path_or_config=config,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        quantization_config=None,
        force_hf=False,
    )
    assert not is_custom
    assert isinstance(model, HFLfm2VlForConditionalGeneration)
    assert model.can_generate()
    reference = HFLfm2VlForConditionalGeneration._from_config(config, dtype=dtype, attn_implementation="sdpa")
    reference.load_state_dict(model.state_dict(), strict=True)
    model.to(device).eval()
    reference.to(device).eval()
    ids = (
        torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device=device)
        if with_image
        else torch.tensor([[1, 2, 3]], device=device)
    )
    image_args = (
        {
            "pixel_values": torch.randn(1, 16, 768, device=device, dtype=dtype),
            "spatial_shapes": torch.tensor([[4, 4]], device=device),
            "pixel_attention_mask": torch.ones(1, 16, dtype=torch.int32, device=device),
        }
        if with_image
        else {}
    )
    generation_args = {
        "max_new_tokens": 3,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": 0,
        "use_cache": use_cache,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    with torch.no_grad():
        actual = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **image_args, **generation_args)
        expected = reference.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), **image_args, **generation_args
        )
    assert actual.sequences.shape == (1, ids.shape[1] + 3)
    torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
    for score, reference_score in zip(actual.scores, expected.scores, strict=True):
        torch.testing.assert_close(score, reference_score, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("with_image", [False, True])
def test_native_d1_fp32_logits_and_every_parameter_gradient_match_hf(with_image, monkeypatch):
    # Use HF's own CPU convolution body, rather than its installed CUDA override.
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(modeling_lfm2, name, inspect.unwrap(getattr(modeling_lfm2, name)))
    torch.manual_seed(1739)
    config = tiny_config()
    config.auto_map = {"AutoModel": "modeling_d1.D1Model"}
    with torch.device("cpu"):
        reference = HFLfm2VlForConditionalGeneration._from_config(
            config, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        native = Lfm2VlForConditionalGeneration(config).float().eval()
    with torch.no_grad():
        for module in reference.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.copy_(torch.linspace(0.65, 1.35, module.weight.numel()))
    native.load_state_dict(reference.state_dict(), strict=True)
    assert isinstance(native.model.language_model.rotary_emb, D1CausalRotaryEmbedding)
    ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]]) if with_image else torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    valid = torch.ones_like(ids) if with_image else ids.ne(0).long()
    image = torch.randn(1, 16, 768) if with_image else None
    image_args = (
        {"spatial_shapes": torch.tensor([[4, 4]]), "pixel_attention_mask": torch.ones(1, 16, dtype=torch.int32)}
        if with_image
        else {}
    )
    reference_pixels = image.clone().requires_grad_() if image is not None else None
    native_pixels = image.clone().requires_grad_() if image is not None else None
    expected = reference(
        input_ids=ids, attention_mask=valid, pixel_values=reference_pixels, use_cache=False, **image_args
    ).logits
    actual = native(
        input_ids=ids, attention_mask=valid, pixel_values=native_pixels, use_cache=False, **image_args
    ).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    reference_parameters = dict(reference.named_parameters())
    native_parameters = dict(native.named_parameters())
    assert native_parameters.keys() == reference_parameters.keys()
    for name, parameter in native_parameters.items():
        gradient = reference_parameters[name].grad
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, atol=3e-6, rtol=3e-5, msg=name)
    if with_image:
        torch.testing.assert_close(native_pixels.grad, reference_pixels.grad, atol=3e-6, rtol=3e-5)
