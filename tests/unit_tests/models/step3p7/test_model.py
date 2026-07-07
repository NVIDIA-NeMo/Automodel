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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p7 import model as step3p7_model
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration, Step3p7Model


def small_config(**kwargs):
    values = dict(
        vision_config={
            "width": 8,
            "layers": 0,
            "heads": 2,
            "num_channels": 3,
            "image_size": 8,
            "patch_size": 2,
            "mlp_ratio": 2.0,
            "hidden_act": "gelu",
            "use_ln_pre": False,
            "use_ln_post": False,
            "use_abs_posemb": False,
            "use_rope2d": False,
        },
        text_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_attention_groups": 1,
            "num_hidden_layers": 0,
            "vocab_size": 32,
            "moe_num_experts": 2,
            "moe_top_k": 1,
            "moe_intermediate_size": 4,
            "share_expert_dims": 4,
            "head_dim": 4,
            "torch_dtype": "float32",
            "moe_layers_enum": (),
            "layer_types": [],
        },
        image_token_id=31,
    )
    values.update(kwargs)
    return Step3p7Config(**values)


def backend(**kwargs):
    values = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    values.update(kwargs)
    return BackendConfig(**values)


def test_debug_helpers_and_rank(monkeypatch):
    monkeypatch.delenv("NEMO_STEP3P7_DEBUG_VISION", raising=False)
    assert step3p7_model._debug_vision_enabled() is False
    monkeypatch.setenv("NEMO_STEP3P7_DEBUG_VISION", "true")
    assert step3p7_model._debug_vision_enabled() is True

    step3p7_model._debug_vision_log("value=%s", 7)
    assert step3p7_model._rank() == 0

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    assert step3p7_model._rank() == 3


def test_step3p7_model_properties_and_embedding_accessors():
    model = Step3p7Model(small_config(), backend())

    assert model.layers is model.language_model.layers
    assert model.embed_tokens is model.language_model.embed_tokens
    assert model.norm is model.language_model.norm
    assert model.get_input_embeddings() is model.language_model.embed_tokens

    new_embed = nn.Embedding(4, 8)
    model.set_input_embeddings(new_embed)
    assert model.language_model.embed_tokens is new_embed

    decoder = nn.Module()
    model.set_decoder(decoder)
    assert model.get_decoder() is decoder


def test_vision_dtype_device_falls_back_when_no_parameters():
    model = Step3p7Model(small_config(), backend())
    model.vision_model = nn.Module()
    dtype, device = model._vision_dtype_device()
    assert dtype == torch.float32
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_process_image_features_square_check_and_forward():
    model = Step3p7Model(small_config(), backend()).float()
    with pytest.raises(ValueError, match="square grid"):
        model._process_image_features(torch.randn(1, 3, 8))

    features = model._process_image_features(torch.randn(1, 16, 8))
    assert features.shape == (1, 1, 8)


def test_process_image_input_merges_patches_and_logs(monkeypatch):
    model = Step3p7Model(small_config(), backend()).float()
    monkeypatch.setenv("NEMO_STEP3P7_DEBUG_VISION", "1")
    pixel_values = torch.randn(2, 3, 8, 8)
    patch_pixel_values = torch.randn(1, 3, 8, 8)

    no_patch = model._process_image_input(pixel_values[:1])
    assert no_patch[0].shape == (1, 8)

    tensor_patches = model._process_image_input(pixel_values[:1], num_patches=torch.tensor([0]))
    assert tensor_patches[0].shape == (1, 8)

    merged = model._process_image_input(pixel_values, patch_pixel_values=patch_pixel_values, num_patches=[1, 0])
    assert [item.shape for item in merged] == [torch.Size([2, 8]), torch.Size([1, 8])]

    with pytest.raises(ValueError, match="patch_pixel_values is missing"):
        model._process_image_input(pixel_values, num_patches=[1, 0])

    model.vision_model = None
    with pytest.raises(ValueError, match="vision_model is not present"):
        model._process_image_input(pixel_values)


def test_multimodal_embeddings_and_prepare_inputs_embeds_paths():
    config = small_config()
    model = Step3p7Model(config, backend()).float()
    input_ids = torch.tensor([[1, config.image_token_id, 2]])

    assert model.get_multimodal_embeddings() is None
    assert model.get_multimodal_embeddings(pixel_values=torch.randn(1, 3, 8, 8))[0].shape == (1, 8)
    image_embeds = torch.randn(1, 1, 8)
    flattened = model.get_multimodal_embeddings(image_embeds=image_embeds)
    assert flattened.shape == (1, 8)

    embeds = model.prepare_inputs_embeds(input_ids, flattened)
    torch.testing.assert_close(embeds[0, 1], flattened[0].to(embeds.dtype))

    with pytest.raises(ValueError, match="image token mismatch"):
        model.prepare_inputs_embeds(input_ids, torch.randn(2, 8))

    model.language_model.embed_tokens = None
    float_inputs = torch.randn(1, 3, 8)
    assert model.prepare_inputs_embeds(float_inputs) is float_inputs
    with pytest.raises(ValueError, match="embed_tokens is not present"):
        model.prepare_inputs_embeds(input_ids)


def test_step3p7_model_forward_accepts_input_ids_inputs_embeds_and_adjusts_mask(monkeypatch):
    model = Step3p7Model(small_config(), backend())
    called = {}

    def fake_language_forward(**kwargs):
        called.update(kwargs)
        return kwargs["inputs_embeds"]

    model.language_model.forward = fake_language_forward
    input_ids = torch.tensor([[1, 2, 3]])
    out = model(input_ids, attention_mask=torch.ones(1, 2))
    assert out.shape == (1, 3, 8)
    assert called["attention_mask"] is None

    float_inputs = torch.randn(1, 3, 8)
    assert model(float_inputs).shape == (1, 3, 8)

    explicit_embeds = torch.randn(1, 3, 8)
    assert model(inputs_embeds=explicit_embeds).shape == (1, 3, 8)

    with pytest.raises(ValueError, match="requires input_ids or inputs_embeds"):
        model()


def test_prepare_inputs_embeds_debug_logging(monkeypatch):
    config = small_config()
    model = Step3p7Model(config, backend())
    monkeypatch.setenv("NEMO_STEP3P7_DEBUG_VISION", "true")
    input_ids = torch.tensor([[1, config.image_token_id, 2]])

    embeds = model.prepare_inputs_embeds(input_ids, torch.randn(1, 8))

    assert embeds.shape == (1, 3, 8)


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = SimpleNamespace(embed_tokens=nn.Embedding(4, 8), rotary_emb=SimpleNamespace(device=None))
        self.last_kwargs = None

    def get_multimodal_embeddings(self, **kwargs):
        return kwargs.get("image_embeds")

    def prepare_inputs_embeds(self, input_ids, multimodal_embeddings=None):
        embeds = torch.randn(input_ids.shape[0], input_ids.shape[1], 8)
        if multimodal_embeddings is not None:
            embeds[input_ids == 31] = multimodal_embeddings.reshape(-1, 8).to(embeds.dtype)
        return embeds

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        batch = (
            kwargs["inputs_embeds"].shape[0]
            if kwargs.get("inputs_embeds") is not None
            else kwargs["input_ids"].shape[0]
        )
        seq = (
            kwargs["inputs_embeds"].shape[1]
            if kwargs.get("inputs_embeds") is not None
            else kwargs["input_ids"].shape[1]
        )
        return torch.ones(batch, seq, 8)


def _wrapper_with_fake_inner(attn="sdpa"):
    wrapper = Step3p7ForConditionalGeneration(small_config(), backend=backend(attn=attn))
    fake_inner = FakeInnerModel()
    wrapper.model = fake_inner
    wrapper.lm_head = nn.Identity()
    return wrapper, fake_inner


def test_for_conditional_generation_properties_accessors_and_adapter():
    cfg = small_config()
    wrapper = Step3p7ForConditionalGeneration.from_config(
        cfg,
        backend=backend(enable_hf_state_dict_adapter=True),
    )

    assert wrapper.language_model is wrapper.model.language_model
    assert wrapper.visual is wrapper.model.vision_model
    assert wrapper.get_input_embeddings() is wrapper.model.language_model.embed_tokens
    assert wrapper.get_output_embeddings() is wrapper.lm_head
    assert hasattr(wrapper, "state_dict_adapter")

    new_embed = nn.Embedding(4, 8)
    wrapper.set_input_embeddings(new_embed)
    assert wrapper.model.language_model.embed_tokens is new_embed

    new_head = nn.Linear(8, 4)
    wrapper.set_output_embeddings(new_head)
    assert wrapper.get_output_embeddings() is new_head

    decoder = nn.Module()
    wrapper.set_decoder(decoder)
    assert wrapper.get_decoder() is decoder


def test_from_pretrained_uses_step3p7_config(monkeypatch):
    cfg = small_config()
    monkeypatch.setattr(step3p7_model.Step3p7Config, "from_pretrained", MagicMock(return_value=cfg))
    monkeypatch.setattr(step3p7_model.Step3p7ForConditionalGeneration, "from_config", MagicMock(return_value="model"))
    assert Step3p7ForConditionalGeneration.from_pretrained("/tmp/model") == "model"
    step3p7_model.Step3p7Config.from_pretrained.assert_called_once_with("/tmp/model")


def test_prepare_model_inputs_for_cp_and_pre_embed_only_error():
    wrapper, _ = _wrapper_with_fake_inner()
    input_ids = torch.tensor([[1, 31, 2]])
    image_embeds = torch.randn(1, 8)

    result = wrapper.prepare_model_inputs_for_cp({"input_ids": input_ids, "image_embeds": image_embeds})
    assert set(result) == {"inputs_embeds"}
    assert result["inputs_embeds"].shape == (1, 3, 8)

    assert wrapper(input_ids=input_ids, image_embeds=image_embeds, _pre_embed_only=True)["inputs_embeds"].shape == (
        1,
        3,
        8,
    )
    with pytest.raises(ValueError, match="CP pre-embedding requires input_ids"):
        wrapper(_pre_embed_only=True)


def test_forward_consumes_pp_vlm_chunks_and_drops_mismatched_masks(monkeypatch):
    wrapper, fake_inner = _wrapper_with_fake_inner()
    monkeypatch.setenv("NEMO_STEP3P7_DEBUG_VISION", "1")
    wrapper._vlm_chunk_idx = 0
    wrapper._vlm_pixel_values_chunks = [torch.randn(1, 3, 8, 8)]
    wrapper._vlm_patch_pixel_values_chunks = [torch.randn(2, 3, 8, 8)]
    wrapper._vlm_num_patches_chunks = [torch.tensor([2])]
    wrapper._vlm_patch_newline_mask_chunks = [torch.tensor([True, False])]

    input_ids = torch.tensor([[31, 1, 2]])
    logits = wrapper(input_ids=input_ids, inputs_embeds=torch.randn(1, 3, 8), attention_mask=torch.ones(1, 2))

    assert logits.shape == (1, 3, 8)
    assert wrapper._vlm_chunk_idx == 1
    assert fake_inner.last_kwargs["pixel_values"].shape == (1, 3, 8, 8)
    assert fake_inner.last_kwargs["patch_pixel_values"].shape == (2, 3, 8, 8)
    assert torch.equal(fake_inner.last_kwargs["num_patches"], torch.tensor([2]))
    assert fake_inner.last_kwargs["attention_mask"] is None


def test_forward_te_attention_drops_attention_mask_and_qkv_thd_squeezes(monkeypatch):
    wrapper, fake_inner = _wrapper_with_fake_inner(attn="te")
    squeezed_padding_mask = torch.tensor([True])

    def fake_squeeze(input_ids, position_ids, padding_mask, kwargs):
        kwargs = dict(kwargs)
        kwargs["squeezed"] = True
        return input_ids, position_ids, squeezed_padding_mask, kwargs

    monkeypatch.setattr(step3p7_model, "squeeze_input_for_thd", fake_squeeze)
    logits = wrapper(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3),
        padding_mask=torch.ones(1, 3, dtype=torch.bool),
        qkv_format="thd",
    )

    assert logits.shape == (1, 3, 8)
    assert fake_inner.last_kwargs["attention_mask"] is None
    assert fake_inner.last_kwargs["padding_mask"] is squeezed_padding_mask
    assert fake_inner.last_kwargs["squeezed"] is True

    wrapper, fake_inner = _wrapper_with_fake_inner(attn="te")
    wrapper(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
    assert fake_inner.last_kwargs["attention_mask"] is None


def test_initialize_weights_initializes_lm_head_and_casts(monkeypatch):
    wrapper = Step3p7ForConditionalGeneration(small_config(), backend=backend())
    wrapper.model.language_model.init_weights = MagicMock()

    wrapper.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    wrapper.model.language_model.init_weights.assert_called_once()
    assert wrapper.model.language_model.rotary_emb.device == torch.device("cpu")


def test_model_class_alias():
    assert step3p7_model.ModelClass is Step3p7ForConditionalGeneration
