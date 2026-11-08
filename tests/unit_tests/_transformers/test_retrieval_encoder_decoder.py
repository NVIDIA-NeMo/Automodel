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

"""Native encoder-decoder scoring must survive retrieval wrapping and loading."""

import pytest
import torch
from transformers import (
    BartConfig,
    BartForSequenceClassification,
    PreTrainedModel,
    T5Config,
    T5ForSequenceClassification,
)

from nemo_automodel._transformers import retrieval


@pytest.fixture(params=["t5", "bart"])
def native_scorer(request) -> PreTrainedModel:
    """Return a tiny real scorer with bidirectional, causal, and cross-attention."""
    if request.param == "t5":
        model = T5ForSequenceClassification(
            T5Config(
                vocab_size=32,
                d_model=16,
                d_kv=8,
                d_ff=32,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                num_labels=1,
                decoder_start_token_id=0,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    else:
        model = BartForSequenceClassification(
            BartConfig(
                vocab_size=32,
                d_model=16,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                encoder_ffn_dim=32,
                decoder_ffn_dim=32,
                max_position_embeddings=32,
                num_labels=1,
                decoder_start_token_id=0,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    return model.eval()


def _attention_flags(model: PreTrainedModel) -> dict[str, tuple[bool | None, bool | None]]:
    """Capture encoder, decoder, and cross-attention policies without naming model internals."""
    return {
        name: (getattr(module, "is_causal", None), getattr(module, "is_decoder", None))
        for name, module in model.named_modules()
    }


def _attention_configs(model: PreTrainedModel) -> list[dict]:
    """Capture attention settings without unrelated checkpoint metadata."""
    fields = {"is_encoder_decoder", "is_decoder", "is_causal", "add_cross_attention"}
    return [
        {key: value for key, value in tower.config.to_dict().items() if key in fields}
        for tower in (model, model.get_encoder(), model.get_decoder())
    ]


@torch.no_grad()
def test_cross_encoder_preserves_native_encoder_decoder_scores(native_scorer):
    input_ids = torch.tensor([[3, 4, 2], [5, 6, 2]])
    attention_mask = torch.ones_like(input_ids)
    expected = native_scorer(input_ids=input_ids, attention_mask=attention_mask).logits
    native_flags = _attention_flags(native_scorer)
    native_configs = _attention_configs(native_scorer)

    encoder = retrieval.CrossEncoderModel(native_scorer).eval()

    assert encoder.is_causal is None  # Native encoder, decoder, and cross-attention differ.
    assert _attention_flags(native_scorer) == native_flags
    assert _attention_configs(native_scorer) == native_configs
    actual = encoder(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_cross_encoder_round_trip_preserves_native_encoder_decoder_scores(tmp_path, native_scorer):
    input_ids = torch.tensor([[3, 4, 2], [5, 6, 2]])
    attention_mask = torch.ones_like(input_ids)
    expected = native_scorer(input_ids=input_ids, attention_mask=attention_mask).logits
    native_flags = _attention_flags(native_scorer)
    source_dir = tmp_path / "source"
    native_scorer.save_pretrained(source_dir)

    encoder = retrieval.CrossEncoderModel.build(str(source_dir)).eval()
    saved_dir = tmp_path / "saved"
    encoder.save_pretrained(str(saved_dir))
    reloaded = retrieval.CrossEncoderModel.build(str(saved_dir)).eval()

    for loaded in (encoder, reloaded):
        assert loaded.is_causal is None
        assert "is_causal" not in vars(loaded.model.config)
        assert _attention_flags(loaded.model) == native_flags
        torch.testing.assert_close(loaded.model.state_dict(), native_scorer.state_dict(), rtol=0, atol=0)
        actual = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
        # Computed scores allow FP32 roundoff across loads; checkpoint weights remain exact.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("policy_source", ["explicit", "saved"])
@pytest.mark.parametrize("is_causal", [False, True])
def test_cross_encoder_rejects_encoder_decoder_overrides_before_mutation(native_scorer, policy_source, is_causal):
    if policy_source == "saved":
        config = type(native_scorer.config).from_dict({**native_scorer.config.to_dict(), "is_causal": is_causal})
        native_scorer = type(native_scorer)(config).eval()
    policy_kwargs = {"is_causal": is_causal} if policy_source == "explicit" else {}
    native_flags = _attention_flags(native_scorer)
    towers = (native_scorer, native_scorer.get_encoder(), native_scorer.get_decoder())
    native_configs = [tower.config.to_dict() for tower in towers]

    with pytest.raises(ValueError, match="encoder-decoder"):
        retrieval.CrossEncoderModel(native_scorer, **policy_kwargs)

    assert _attention_flags(native_scorer) == native_flags
    assert [tower.config.to_dict() for tower in towers] == native_configs
