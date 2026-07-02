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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers.image_utils import PILImageResampling
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.models.llama_nemotron_vl.model import (
    LlamaBidirectionalConfig,
    LlamaNemotronVLConfig,
    LlamaNemotronVLEncoderStateDictAdapter,
    LlamaNemotronVLModel,
    replace_siglip_encoder_layers_with_te_fused,
)
from nemo_automodel.components.models.llama_nemotron_vl.processor import (
    LlamaNemotronVLImageProcessor,
    LlamaNemotronVLProcessor,
    dynamic_preprocess,
    find_closest_aspect_ratio,
)

IMG_CONTEXT_TOKEN_ID = 99


class FakeTokenizer:
    def __init__(self):
        self.model_input_names = ["input_ids", "attention_mask"]
        self.model_max_length = 99
        self.padding_side = "right"
        self.calls = []

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.append({"texts": list(texts), "kwargs": kwargs})
        input_ids = []
        attention_mask = []
        for text in texts:
            num_image_tokens = text.count("<IMG_CONTEXT>")
            ids = [1] + [IMG_CONTEXT_TOKEN_ID] * num_image_tokens + [2]
            input_ids.append(ids)
        max_length = max(len(ids) for ids in input_ids)
        for ids in input_ids:
            pad_length = max_length - len(ids)
            attention_mask.append([1] * len(ids) + [0] * pad_length)
            ids.extend([0] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def convert_tokens_to_ids(self, token):
        if token == "<IMG_CONTEXT>":
            return IMG_CONTEXT_TOKEN_ID
        return 0


class FakeVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeLanguageModel(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=8):
        super().__init__()
        self.config = type("FakeLanguageConfig", (), {"vocab_size": vocab_size})()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_calls = []
        with torch.no_grad():
            token_values = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(1)
            self.embedding.weight.copy_(token_values.repeat(1, hidden_size))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        bidirectional_mask=None,
        bidirectional_mask_precomputed=False,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        self.forward_calls.append(
            {
                "inputs_embeds": inputs_embeds.detach().clone(),
                "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
                "position_ids": position_ids,
                "bidirectional_mask": bidirectional_mask,
                "bidirectional_mask_precomputed": bidirectional_mask_precomputed,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
            }
        )
        hidden_states = (inputs_embeds, inputs_embeds + 1.0)
        return CausalLMOutputWithPast(
            logits=self.lm_head(inputs_embeds),
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr(ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    return LlamaNemotronVLProcessor(
        tokenizer=FakeTokenizer(),
        q_max_length=7,
        p_max_length=11,
        pad_to_multiple_of=4,
        query_prefix="query:",
        passage_prefix="passage:",
        image_size=4,
        num_image_token=2,
    )


@pytest.fixture
def tiny_model(monkeypatch, processor):
    import nemo_automodel.components.models.llama_nemotron_vl.model as model_module

    monkeypatch.setattr(model_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    config = LlamaNemotronVLConfig(
        vision_config=_tiny_vision_config(),
        llm_config=_tiny_llm_config(),
        pooling="avg",
        img_context_token_id=IMG_CONTEXT_TOKEN_ID,
    )
    model = LlamaNemotronVLModel(
        config,
        vision_model=FakeVisionModel(),
        language_model=FakeLanguageModel(
            vocab_size=128,
            hidden_size=config.llm_config.hidden_size,
        ),
    )
    with torch.no_grad():
        token_values = torch.arange(128, dtype=torch.float32).unsqueeze(1)
        model.language_model.embedding.weight.copy_(token_values.repeat(1, config.llm_config.hidden_size))
    model.eval()
    return model


def _tiny_vision_config():
    return {
        "model_type": "siglip_vision_model",
        "hidden_size": 8,
        "image_size": 4,
        "patch_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 16,
    }


def _tiny_llm_config():
    return {
        "architectures": ["LlamaBidirectionalModel"],
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
    }


def _adapter_with_all_fusions():
    model = SimpleNamespace(
        config=SimpleNamespace(
            llm_config=LlamaBidirectionalConfig(
                architectures=["LlamaBidirectionalModel"],
                model_type="llama",
                vocab_size=32,
                hidden_size=4,
                intermediate_size=6,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
            ),
            vision_config=SiglipVisionConfig(
                hidden_size=4,
                image_size=4,
                patch_size=2,
                num_hidden_layers=1,
                num_attention_heads=1,
                intermediate_size=8,
            ),
        ),
        _nemo_use_te_fused_llama_mlp=True,
        _nemo_use_te_fused_llama_qkv=True,
        _nemo_use_te_fused_siglip_layer=True,
        _nemo_use_te_fused_vision_projection=True,
    )
    return LlamaNemotronVLEncoderStateDictAdapter(model)


def test_vl_encoder_state_dict_adapter_exports_hf_keys_for_fused_layers():
    adapter = _adapter_with_all_fusions()
    state_dict = {
        "model.language_model.layers.0.self_attn.pre_attention_qkv.fused.layer_norm_weight": torch.ones(4),
        "model.language_model.layers.0.self_attn.pre_attention_qkv.fused.weight": torch.arange(32).reshape(8, 4),
        "model.language_model.layers.0.mlp.fused.layer_norm_weight": torch.ones(4) * 2,
        "model.language_model.layers.0.mlp.fused.fc1_weight": torch.arange(48).reshape(12, 4),
        "model.language_model.layers.0.mlp.fused.fc2_weight": torch.arange(24).reshape(4, 6),
        "model.vision_model.vision_model.encoder.layers.0.mlp.fused.self_attention.layernorm_qkv.layer_norm_weight": torch.ones(
            4
        ),
        "model.vision_model.vision_model.encoder.layers.0.mlp.fused.self_attention.layernorm_qkv.weight": torch.arange(
            48
        ).reshape(12, 4),
        "model.vision_model.vision_model.encoder.layers.0.mlp.fused.self_attention.proj.weight": torch.arange(
            16
        ).reshape(4, 4),
        "model.vision_model.vision_model.encoder.layers.0.mlp.fused.layernorm_mlp.fc1_weight": torch.arange(32).reshape(
            8, 4
        ),
        "model.mlp1.norm_fc1.layer_norm_weight": torch.ones(8),
        "model.mlp1.norm_fc1.weight": torch.arange(32).reshape(4, 8),
        "model.mlp1.fc2.weight": torch.arange(16).reshape(4, 4),
    }

    hf_state_dict = adapter.to_hf(state_dict)

    assert "language_model.layers.0.self_attn.pre_attention_qkv.fused.weight" not in hf_state_dict
    assert torch.equal(
        hf_state_dict["language_model.layers.0.self_attn.q_proj.weight"],
        torch.arange(32).reshape(8, 4)[:4],
    )
    assert torch.equal(
        hf_state_dict["language_model.layers.0.self_attn.k_proj.weight"],
        torch.arange(32).reshape(8, 4)[4:6],
    )
    assert torch.equal(
        hf_state_dict["language_model.layers.0.mlp.gate_proj.weight"],
        torch.arange(48).reshape(12, 4)[:6],
    )
    assert (
        "vision_model.vision_model.encoder.layers.0.mlp.fused.self_attention.layernorm_qkv.weight" not in hf_state_dict
    )
    assert torch.equal(
        hf_state_dict["vision_model.vision_model.encoder.layers.0.self_attn.v_proj.weight"],
        torch.arange(48).reshape(12, 4)[8:12],
    )
    assert torch.equal(hf_state_dict["mlp1.0.weight"], torch.ones(8))
    assert torch.equal(hf_state_dict["mlp1.1.weight"], torch.arange(32).reshape(4, 8))


def test_vl_encoder_state_dict_adapter_exports_single_fused_tensor_to_hf_keys():
    adapter = _adapter_with_all_fusions()
    tensor = torch.arange(32).reshape(8, 4)

    converted = adapter.convert_single_tensor_to_hf(
        "model.language_model.layers.0.self_attn.pre_attention_qkv.fused.weight",
        tensor,
    )

    assert [key for key, _ in converted] == [
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.self_attn.k_proj.weight",
        "language_model.layers.0.self_attn.v_proj.weight",
    ]
    assert torch.equal(converted[0][1], tensor[:4])
    assert torch.equal(converted[1][1], tensor[4:6])
    assert torch.equal(converted[2][1], tensor[6:8])


def test_vl_encoder_state_dict_adapter_imports_hf_keys_for_fused_layers():
    adapter = _adapter_with_all_fusions()
    hf_state_dict = {
        "language_model.layers.0.input_layernorm.weight": torch.ones(4),
        "language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "language_model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4) * 2,
        "language_model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4) * 3,
        "language_model.layers.0.post_attention_layernorm.weight": torch.ones(4) * 4,
        "language_model.layers.0.mlp.gate_proj.weight": torch.ones(6, 4),
        "language_model.layers.0.mlp.up_proj.weight": torch.ones(6, 4) * 2,
        "language_model.layers.0.mlp.down_proj.weight": torch.ones(4, 6),
        "mlp1.0.weight": torch.ones(8),
        "mlp1.1.weight": torch.ones(4, 8),
        "mlp1.3.weight": torch.ones(4, 4),
    }

    internal_state_dict = adapter.from_hf(hf_state_dict)

    assert "model.language_model.layers.0.self_attn.q_proj.weight" not in internal_state_dict
    fused_qkv = internal_state_dict["model.language_model.layers.0.self_attn.pre_attention_qkv.fused.weight"]
    assert fused_qkv.shape == (8, 4)
    assert torch.equal(fused_qkv[:4], torch.ones(4, 4))
    assert torch.equal(fused_qkv[4:6], torch.ones(2, 4) * 2)
    assert "model.language_model.layers.0.mlp.gate_proj.weight" not in internal_state_dict
    assert internal_state_dict["model.language_model.layers.0.mlp.fused.fc1_weight"].shape == (12, 4)
    assert "model.mlp1.0.weight" not in internal_state_dict
    assert torch.equal(internal_state_dict["model.mlp1.norm_fc1.layer_norm_weight"], torch.ones(8))


def test_siglip_te_fusion_rejects_intermediate_select_layer():
    model = SimpleNamespace(select_layer=0, config=SimpleNamespace(select_layer=0))
    with pytest.raises(ValueError, match="select_layer=-1"):
        replace_siglip_encoder_layers_with_te_fused(model)


def test_processor_initializes_tokenizer_for_pixel_values(processor):
    assert processor.tokenizer.padding_side == "left"
    assert processor.tokenizer.model_input_names == ["input_ids", "attention_mask", "pixel_values"]


def test_processor_process_queries_prefixes_and_tokenizes(processor):
    output = processor.process_queries(["What is shown?"])

    assert set(output) == {"input_ids", "attention_mask"}
    call = processor.tokenizer.calls[-1]
    assert call["texts"] == ["query: What is shown?"]
    assert call["kwargs"] == {
        "truncation": True,
        "max_length": 7,
        "padding": True,
        "pad_to_multiple_of": 4,
        "return_tensors": "pt",
    }


def test_processor_process_queries_handles_multiple_queries_and_options(processor):
    processor.process_queries(["First query", "Second query"], return_tensors="np", padding=False, truncation=False)

    call = processor.tokenizer.calls[-1]
    assert call["texts"] == ["query: First query", "query: Second query"]
    assert call["kwargs"] == {
        "truncation": False,
        "max_length": None,
        "padding": False,
        "pad_to_multiple_of": 4,
        "return_tensors": "np",
    }


def test_processor_process_documents_supports_text_only_inputs(processor):
    output = processor.process_documents({"images": ["", None], "texts": ["Document A", "Document B"]})

    assert set(output) == {"input_ids", "attention_mask", "pixel_values"}
    assert output["pixel_values"] is None
    call = processor.tokenizer.calls[-1]
    assert call["texts"] == ["passage: Document A", "passage: Document B"]
    assert call["kwargs"] == {
        "truncation": True,
        "max_length": 11,
        "padding": True,
        "pad_to_multiple_of": 4,
        "return_tensors": "pt",
    }


def test_processor_process_documents_supports_pil_images_and_text(processor):
    images = [
        Image.new("RGB", (4, 4), (255, 0, 0)),
        Image.new("RGB", (8, 4), (0, 255, 0)),
    ]

    output = processor.process_documents({"images": images, "texts": ["text 1", "text 2"]})

    assert set(output) == {"input_ids", "attention_mask", "pixel_values", "image_token_indices"}
    assert output["pixel_values"].shape == (4, 3, 4, 4)
    assert output["pixel_values"].dtype == torch.bfloat16
    assert output["image_token_indices"].numel() == 8

    call = processor.tokenizer.calls[-1]
    assert call["texts"][0].startswith("passage: <img>")
    assert call["texts"][0].count("<IMG_CONTEXT>") == 2
    assert call["texts"][0].endswith("</img> text 1")
    assert call["texts"][1].startswith("passage: <img>")
    assert call["texts"][1].count("<IMG_CONTEXT>") == 6
    assert call["texts"][1].endswith("</img> text 2")
    assert call["kwargs"] == {
        "truncation": True,
        "max_length": 11,
        "padding": True,
        "pad_to_multiple_of": 4,
        "return_tensors": "pt",
    }


def test_processor_process_documents_can_return_per_image_tiles(processor):
    images = [
        Image.new("RGB", (4, 4), (255, 0, 0)),
        Image.new("RGB", (8, 4), (0, 255, 0)),
    ]

    output = processor.process_documents(
        {"images": images, "texts": ["text 1", "text 2"]},
        pixel_values_layout="per_image",
    )

    assert len(output["pixel_values"]) == 2
    assert output["pixel_values"][0].shape == (1, 3, 4, 4)
    assert output["pixel_values"][1].shape == (3, 3, 4, 4)


def test_processor_biencoder_collator_merges_query_document_batches(processor):
    features = [
        {"question": "Question 0", "doc_text": ["Doc 0", "Doc 1"], "doc_image": ["", ""]},
        {"question": "Question 1", "doc_text": ["Doc 2", "Doc 3"], "doc_image": ["", ""]},
    ]

    output = processor.process_queries_documents_biencoder(features)

    assert set(output) == {
        "q_input_ids",
        "q_attention_mask",
        "d_input_ids",
        "d_attention_mask",
        "d_pixel_values",
        "labels",
    }
    assert output["q_input_ids"].shape[0] == 2
    assert output["d_input_ids"].shape[0] == 4
    assert output["d_pixel_values"] is None
    assert output["labels"].dtype == torch.long
    assert torch.equal(output["labels"], torch.zeros(2, dtype=torch.long))


def test_model_encode_queries_uses_processor_and_pools_text_embeddings(tiny_model):
    embeddings = tiny_model.encode_queries(["First query", "Second query"])

    assert embeddings.shape == (2, tiny_model.config.llm_config.hidden_size)
    assert torch.allclose(embeddings, torch.full_like(embeddings, 2.5))
    assert tiny_model.processor.tokenizer.calls[-1]["texts"] == ["query: First query", "query: Second query"]
    forward_call = tiny_model.language_model.forward_calls[-1]
    assert forward_call["inputs_embeds"].shape == (2, 2, tiny_model.config.llm_config.hidden_size)
    assert forward_call["output_hidden_states"] is True


def test_model_encode_documents_with_images_and_text_uses_vision_embeddings(tiny_model, monkeypatch):
    captured = {}

    def fake_extract_feature(pixel_values):
        captured["pixel_values_shape"] = tuple(pixel_values.shape)
        hidden_size = tiny_model.config.llm_config.hidden_size
        vision_features = torch.arange(pixel_values.shape[0] * 2 * hidden_size, dtype=torch.float32).reshape(
            pixel_values.shape[0],
            2,
            hidden_size,
        )
        captured["vision_features"] = vision_features
        return vision_features

    monkeypatch.setattr(tiny_model, "extract_feature", fake_extract_feature)
    images = [
        Image.new("RGB", (4, 4), (255, 0, 0)),
        Image.new("RGB", (8, 4), (0, 255, 0)),
    ]

    embeddings = tiny_model.encode_documents(images=images, texts=["text 1", "text 2"])

    assert embeddings.shape == (2, tiny_model.config.llm_config.hidden_size)
    assert captured["pixel_values_shape"] == (4, 3, 4, 4)
    assert tiny_model.processor.tokenizer.calls[-1]["texts"][0].endswith("</img> text 1")
    assert tiny_model.processor.tokenizer.calls[-1]["texts"][1].endswith("</img> text 2")

    forward_call = tiny_model.language_model.forward_calls[-1]
    input_embeds = forward_call["inputs_embeds"]
    attention_mask = forward_call["attention_mask"]
    assert input_embeds.shape[:2] == attention_mask.shape == (2, 8)
    assert attention_mask.tolist() == [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]
    vision_features = captured["vision_features"].reshape(-1, tiny_model.config.llm_config.hidden_size)
    assert torch.allclose(input_embeds[0, 1:3], vision_features[:2])
    assert torch.allclose(input_embeds[1, 1:7], vision_features[2:8])


def test_model_forward_injects_image_embeddings_into_text_sequence(tiny_model, monkeypatch):
    hidden_size = tiny_model.config.llm_config.hidden_size
    vision_features = torch.stack(
        [
            torch.full((hidden_size,), -3.0),
            torch.full((hidden_size,), -7.0),
        ]
    ).reshape(2, 1, hidden_size)

    def fake_extract_feature(pixel_values):
        assert tuple(pixel_values.shape) == (2, 3, 4, 4)
        return vision_features

    monkeypatch.setattr(tiny_model, "extract_feature", fake_extract_feature)
    input_ids = torch.tensor([[1, IMG_CONTEXT_TOKEN_ID, IMG_CONTEXT_TOKEN_ID, 2]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.ones((2, 3, 4, 4))

    output = tiny_model.forward(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    injected_embeds = tiny_model.language_model.forward_calls[-1]["inputs_embeds"]
    assert torch.allclose(injected_embeds[0, 0], torch.full((hidden_size,), 1.0))
    assert torch.allclose(injected_embeds[0, 1], vision_features[0, 0])
    assert torch.allclose(injected_embeds[0, 2], vision_features[1, 0])
    assert torch.allclose(injected_embeds[0, 3], torch.full((hidden_size,), 2.0))
    assert output.logits.shape == (1, 4, tiny_model.language_model.config.vocab_size)
    assert torch.allclose(output.hidden_states[-1], injected_embeds + 1.0)


def _target_ratios(max_num=6):
    return sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )


@pytest.mark.parametrize(
    ("size", "expected_ratio", "expected_tiles"),
    [
        ((4, 4), (1, 1), 1),
        ((8, 4), (2, 1), 2),
        ((4, 8), (1, 2), 2),
        ((12, 4), (3, 1), 3),
        ((4, 12), (1, 3), 3),
        ((16, 8), (3, 2), 6),
        ((8, 16), (2, 3), 6),
        ((20, 4), (5, 1), 5),
        ((4, 20), (1, 5), 5),
    ],
)
def test_find_closest_aspect_ratio_for_image_shapes(size, expected_ratio, expected_tiles):
    ratio = find_closest_aspect_ratio(
        aspect_ratio=size[0] / size[1],
        target_ratios=_target_ratios(),
        width=size[0],
        height=size[1],
        image_size=4,
    )

    assert ratio == expected_ratio
    assert ratio[0] * ratio[1] == expected_tiles


@pytest.mark.parametrize(
    ("size", "expected_tiles"),
    [
        ((4, 4), 1),
        ((8, 4), 2),
        ((4, 8), 2),
        ((12, 4), 3),
        ((4, 12), 3),
        ((16, 8), 6),
        ((8, 16), 6),
        ((20, 4), 5),
        ((4, 20), 5),
    ],
)
def test_dynamic_preprocess_tile_counts_for_image_shapes(size, expected_tiles):
    tiles = dynamic_preprocess(Image.new("RGB", size), image_size=4, max_num=6, use_thumbnail=False)

    assert len(tiles) == expected_tiles
    assert [tile.size for tile in tiles] == [(4, 4)] * expected_tiles


def test_dynamic_preprocess_adds_thumbnail_only_for_multi_tile_images():
    square_tiles = dynamic_preprocess(Image.new("RGB", (4, 4)), image_size=4, max_num=6, use_thumbnail=True)
    wide_tiles = dynamic_preprocess(Image.new("RGB", (8, 4)), image_size=4, max_num=6, use_thumbnail=True)

    assert len(square_tiles) == 1
    assert len(wide_tiles) == 3
    assert [tile.size for tile in wide_tiles] == [(4, 4), (4, 4), (4, 4)]


def test_fast_image_processor_uses_bicubic_resize_by_default(monkeypatch):
    image_processor = LlamaNemotronVLImageProcessor(image_size=4, max_num_tiles=2)
    seen_resample = []

    def fake_resize(image, size, resample=None, **kwargs):
        seen_resample.append(resample)
        return torch.zeros((3, size.height, size.width), dtype=image.dtype)

    monkeypatch.setattr(image_processor, "resize", fake_resize)

    patches = image_processor.dynamic_preprocess(
        torch.zeros((3, 4, 8)),
        image_size=4,
        max_num_tiles=2,
        use_thumbnail=True,
    )

    assert len(patches) == 3
    assert seen_resample == [PILImageResampling.BICUBIC, PILImageResampling.BICUBIC]


def test_fast_image_processor_preprocess_honors_image_kwargs(monkeypatch):
    image_processor = LlamaNemotronVLImageProcessor(image_size=4)
    captured = {}

    def fake_resize(image, size, resample=None, **kwargs):
        captured["resample"] = resample
        return torch.ones((3, size.height, size.width), dtype=image.dtype)

    def fake_rescale_and_normalize(pixel_values, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        captured["do_rescale"] = do_rescale
        captured["rescale_factor"] = rescale_factor
        captured["do_normalize"] = do_normalize
        captured["image_mean"] = image_mean
        captured["image_std"] = image_std
        return pixel_values

    monkeypatch.setattr(image_processor, "resize", fake_resize)
    monkeypatch.setattr(image_processor, "rescale_and_normalize", fake_rescale_and_normalize)

    output = image_processor._preprocess(
        [torch.zeros((3, 4, 4))],
        image_size=4,
        dynamic_image_size=False,
        do_rescale=False,
        rescale_factor=0.5,
        do_normalize=False,
        image_mean=[0.1, 0.2, 0.3],
        image_std=[0.4, 0.5, 0.6],
        resample=PILImageResampling.NEAREST,
        return_tensors="pt",
    )

    assert output["pixel_values"].shape == (1, 3, 4, 4)
    assert captured == {
        "resample": PILImageResampling.NEAREST,
        "do_rescale": False,
        "rescale_factor": 0.5,
        "do_normalize": False,
        "image_mean": [0.1, 0.2, 0.3],
        "image_std": [0.4, 0.5, 0.6],
    }


def test_llama_nemotron_vl_config_builds_composed_subconfigs():
    config = LlamaNemotronVLConfig(
        vision_config=_tiny_vision_config(),
        llm_config=_tiny_llm_config(),
        q_max_length=13,
        p_max_length=21,
        pooling="avg",
    )

    assert config.model_type == "llama_nemotron_vl"
    assert isinstance(config.vision_config, SiglipVisionConfig)
    assert isinstance(config.llm_config, LlamaBidirectionalConfig)
    assert config.llm_config.architectures == ["LlamaBidirectionalModel"]
    assert config.q_max_length == 13
    assert config.p_max_length == 21
    assert config.pooling == "avg"


def test_llama_nemotron_vl_config_rejects_unsupported_vision_model():
    vision_config = _tiny_vision_config()
    vision_config["model_type"] = "unsupported_vision"

    with pytest.raises(ValueError, match="Unsupported model_type"):
        LlamaNemotronVLConfig(vision_config=vision_config, llm_config=_tiny_llm_config())


def test_llama_nemotron_vl_config_rejects_unsupported_llm_architecture():
    llm_config = _tiny_llm_config()
    llm_config["architectures"] = ["UnsupportedArchitecture"]

    with pytest.raises(ValueError, match="Unsupported architecture"):
        LlamaNemotronVLConfig(vision_config=_tiny_vision_config(), llm_config=llm_config)
