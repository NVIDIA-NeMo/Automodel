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

"""Omni configuration and differentiable packed-text wrapper contracts.

Perception is stubbed in these unit tests; the full parity script exercises the
actual pinned image/audio modules separately.
"""

import pytest
import torch
from torch import nn

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.d1 import model as model_module
from nemo_automodel.components.models.d1.config import D1OmniConfig
from nemo_automodel.components.models.d1.model import D1OmniModel
from nemo_automodel.components.models.d1.records import D1TextExample


class StubPerception(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.projection = nn.Linear(64, 64)


@pytest.fixture
def config():
    return D1OmniConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "block_multiple_of": 16,
        },
        vision_config={"fixture": True},
        audio_config={"fixture": True},
        projector_hidden_size=64,
        max_length=128,
        temperatures={"noul:2": 1.7},
    )


@pytest.fixture
def model(monkeypatch, config):
    monkeypatch.setattr(model_module, "get_class_from_dynamic_module", lambda *args, **kwargs: StubPerception)
    return D1OmniModel(config, trust_remote_code=True).eval()


def test_config_roundtrip_and_native_registry(config, tmp_path):
    config.save_pretrained(tmp_path)
    restored = D1OmniConfig.from_pretrained(tmp_path)
    assert config.to_dict()["text_config"]["rope_theta"] == 1000000.0
    assert restored.text_config.hidden_size == 64
    assert restored.text_config.layer_types == ["conv", "full_attention"]
    assert restored.temperatures == {"noul:2": 1.7}
    assert restored.architectures == ["D1OmniModel"]
    assert ModelRegistry.get_model_cls_from_model_arch("D1OmniModel") is D1OmniModel


def test_unequal_prefix_lengths_match_individual_text_encoding(model):
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    valid = torch.tensor([[True] * 4, [True] * 3 + [False]])
    prefix = torch.randn(2, 3, 64)
    lengths = torch.tensor([2, 1])
    text, text_mask = model.encode_text(ids, valid, prefix, lengths)
    for index, length in enumerate([4, 3]):
        separate, _ = model.encode_text(
            ids[index : index + 1, :length],
            valid[index : index + 1, :length],
            prefix[index : index + 1],
            lengths[index : index + 1],
        )
        torch.testing.assert_close(text[index, :length], separate[0], atol=2e-6, rtol=2e-5)
    assert torch.equal(text_mask, valid)


def test_head_only_training_leaves_encoder_frozen(model):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    result = model(
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([2]),
        labels=torch.tensor([1]),
    )
    assert result.logits.shape == (1, 2)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.head.scorer[3].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_model_rejects_invalid_prefix_and_text_padding(model):
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="right-padded"):
        model.encode_text(ids, torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="media_lengths"):
        model.encode_text(ids, torch.ones_like(ids, dtype=torch.bool), torch.randn(1, 2, 64), torch.tensor([3]))


def test_model_rejects_vocabulary_tying(config):
    config.tie_word_embeddings = True
    with pytest.raises(NotImplementedError, match="tie_word_embeddings=True"):
        D1OmniModel(config)


def test_model_requires_consent_before_loading_publisher_code(config, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_download(*args: object, **kwargs: object) -> None:
        pytest.fail("publisher code must not load without explicit consent")

    monkeypatch.setattr("nemo_automodel.components.models.d1.model.get_class_from_dynamic_module", forbidden_download)
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        D1OmniModel(config)


def test_native_loader_forwards_publisher_code_consent(config, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel._transformers.model_init import _init_model

    monkeypatch.setattr(model_module, "get_class_from_dynamic_module", lambda *args, **kwargs: StubPerception)
    native, loaded = _init_model(
        cls=NeMoAutoModelForImageTextToText,
        pretrained_model_name_or_path_or_config=config,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
        quantization_config=None,
        force_hf=False,
        trust_remote_code=True,
    )
    assert native
    assert isinstance(loaded, D1OmniModel)


def test_checkpoint_config_resolves_through_shared_automodel_path(config, tmp_path):
    from nemo_automodel._transformers.model_init import get_hf_config

    config.save_pretrained(tmp_path)
    loaded = get_hf_config(str(tmp_path), "sdpa", trust_remote_code=False, local_files_only=True)
    assert isinstance(loaded, D1OmniConfig)
    assert loaded.text_config.hidden_size == 64
    assert loaded.architectures == ["D1OmniModel"]


def test_omni_text_readout_preserves_full_model_scores_calibration_and_gradients(model):
    model.requires_grad_(False)
    examples = [
        D1TextExample("record-a", (1, 2, 3, 4), "flag", ("false", "true"), (1, 3), (), 2, 1.7),
        D1TextExample("record-b", (5, 6, 7), "flag", ("false", "true"), (0, 2), (), 2, 0.8),
    ]
    readout = model.prepare_text_readout(examples)
    readout.head.requires_grad_(True)
    features, expected = [], []
    for ex in examples:
        ids = torch.tensor([ex.input_ids])
        valid = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            features.append(readout.extract_features(ids, valid))
            expected.append(
                model(
                    ids,
                    valid,
                    torch.tensor([ex.marker_positions]),
                    torch.ones(1, 2, dtype=torch.bool),
                    torch.tensor([2]),
                ).logits[0]
                / ex.temperature
            )
    actual = readout.score(features, examples, head=readout.head)
    for logits, full_model_logits in zip(actual, expected, strict=True):
        torch.testing.assert_close(logits, full_model_logits, atol=2e-6, rtol=2e-5)
    torch.nn.functional.cross_entropy(torch.stack(actual), torch.tensor([1, 0])).backward()
    assert readout.head.scorer[3].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    with pytest.raises(ValueError, match="sequence features"):
        readout.score([features[0][0]], examples[:1], head=readout.head)
