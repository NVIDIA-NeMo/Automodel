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

"""Qwen3RerankerForCausalReranking parity with the published scoring recipe.

The reference is the ``compute_logits`` function on the Qwen/Qwen3-Reranker-4B model
card, which scores by reading full-vocabulary next-token logits at the final position::

    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector, false_vector = batch_scores[:, yes_id], batch_scores[:, no_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    scores = torch.nn.functional.log_softmax(batch_scores, dim=1)[:, 1].exp()

This class returns the raw log-odds ``logit(yes) - logit(no)``; a sigmoid recovers the
same probability.
"""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from nemo_automodel.components.models.common.tie_word_embeddings import TieSupport
from nemo_automodel.components.models.qwen3_reranker.model import (
    Qwen3RerankerConfig,
    Qwen3RerankerForCausalReranking,
)

YES_ID, NO_ID = 5, 7


def _tiny_model() -> Qwen3RerankerForCausalReranking:
    """Build a randomly-initialised tiny reranker; only the scoring path is under test."""
    config = Qwen3RerankerConfig(
        yes_token_id=YES_ID,
        no_token_id=NO_ID,
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        head_dim=8,
    )
    torch.manual_seed(0)
    model = Qwen3RerankerForCausalReranking(config)
    model.eval()
    return model


def _reference_p_yes(model, input_ids, attention_mask, last_idx) -> torch.Tensor:
    """The model-card computation, run through the underlying causal LM."""
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
    logits = model.lm_head(hidden)  # [batch, sequence, vocab]
    last = logits[torch.arange(logits.shape[0]), last_idx]  # [batch, vocab]
    stacked = torch.stack([last[:, NO_ID], last[:, YES_ID]], dim=1)
    return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()


def test_score_matches_model_card_compute_logits_left_padded():
    """Left padding is what the model card uses (padding_side='left')."""
    model = _tiny_model()
    input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # left padding -> final position for every row
        expected = _reference_p_yes(model, input_ids, attention_mask, torch.tensor([4, 4]))

    assert out.logits.shape == (2, 1)
    torch.testing.assert_close(torch.sigmoid(out.logits.squeeze(-1)), expected, rtol=1e-4, atol=1e-4)


def test_score_matches_model_card_compute_logits_right_padded():
    """Right padding must score the last REAL token, not the trailing pad."""
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        expected = _reference_p_yes(model, input_ids, attention_mask, torch.tensor([2, 3]))

    torch.testing.assert_close(torch.sigmoid(out.logits.squeeze(-1)), expected, rtol=1e-4, atol=1e-4)


def test_returned_score_is_the_yes_minus_no_log_odds():
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
        last_logits = model.lm_head(hidden)[:, -1, :]

    expected = last_logits[:, YES_ID] - last_logits[:, NO_ID]
    torch.testing.assert_close(out.logits.squeeze(-1), expected, rtol=1e-4, atol=1e-4)


def test_weight_keys_match_the_causal_lm_backbone():
    """Checkpoints must stay loadable as stock Qwen3ForCausalLM."""
    keys = set(_tiny_model().state_dict())

    assert any(k.startswith("model.") for k in keys)
    assert not any(k.startswith("score.") or k.startswith("classifier.") for k in keys)


def test_config_serializes_as_plain_causal_lm():
    serialized = _tiny_model().config.to_dict()

    assert serialized["model_type"] == "qwen3"
    assert serialized["architectures"] == ["Qwen3ForCausalLM"]
    assert "auto_map" not in serialized
    # the yes/no ids must survive so a reloaded checkpoint can still score
    assert serialized["yes_token_id"] == YES_ID
    assert serialized["no_token_id"] == NO_ID


def test_logits_to_keep_is_rejected_not_ignored():
    """A generation caller must fail loudly rather than receive [batch, 1]."""
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])

    with pytest.raises(ValueError, match="does not support logits_to_keep"):
        model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)


def test_default_logits_to_keep_is_accepted():
    """0 / None are upstream's own defaults and must not trip the guard."""
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])

    with torch.no_grad():
        assert model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=0).logits.shape == (1, 1)


def test_missing_attention_mask_raises():
    model = _tiny_model()
    with pytest.raises(ValueError, match="attention_mask is required"):
        model(input_ids=torch.tensor([[1, 2, 3]]))


# ---------------------------------------------------------------------------
# Save/reload round trip.
#
# to_dict() rewriting the serialized identity is only worth anything if a checkpoint
# written from this class actually comes back through the stock loader and scores the
# same. Asserting on to_dict() alone leaves the whole of save_pretrained/from_pretrained
# untested: a weight-key rename, a dropped config field, or a tie_word_embeddings mismatch
# would all pass that check and still produce a checkpoint that reloads wrong or not at
# all. These reload it for real and compare scores.
# ---------------------------------------------------------------------------


def _yes_minus_no_from_causal_lm(model, input_ids, attention_mask) -> torch.Tensor:
    """Score a plain causal LM the way the model card does, at the final position."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
    return logits[:, YES_ID] - logits[:, NO_ID]


def test_checkpoint_reloads_as_stock_causal_lm_and_scores_identically(tmp_path):
    """The point of the identity rewrite: no custom code, no trust_remote_code, same score.

    Loaded through ``AutoModelForCausalLM``, which is how a consumer actually reaches this
    checkpoint and the only path that exercises the rewrite. Naming ``Qwen3ForCausalLM``
    directly would prove less: the concrete class does not consult ``model_type`` at all, so
    it loads the weights happily even if the serialized identity still said this package's
    class -- the very thing that would send a real caller looking for custom code.
    """
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        original = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
    reloaded.eval()

    # Resolution, not just loadability: the checkpoint must land on upstream's class, and
    # its config must come back as the stock Qwen3Config rather than the reranker subclass.
    assert type(reloaded) is Qwen3ForCausalLM
    assert type(AutoConfig.from_pretrained(tmp_path)) is Qwen3Config
    assert json.loads((tmp_path / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
    torch.testing.assert_close(
        _yes_minus_no_from_causal_lm(reloaded, input_ids, attention_mask), original, rtol=1e-4, atol=1e-4
    )


def test_checkpoint_reloads_as_the_reranker_and_keeps_the_yes_no_ids(tmp_path):
    """Resuming training must recover the token ids; without them the score is meaningless."""
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        original = model(input_ids=input_ids, attention_mask=attention_mask).logits

    model.save_pretrained(tmp_path)
    reloaded = Qwen3RerankerForCausalReranking.from_pretrained(
        tmp_path, config=Qwen3RerankerConfig.from_pretrained(tmp_path)
    )
    reloaded.eval()

    assert reloaded.config.yes_token_id == YES_ID
    assert reloaded.config.no_token_id == NO_ID
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(input_ids=input_ids, attention_mask=attention_mask).logits, original, rtol=1e-4, atol=1e-4
        )


def test_round_trip_preserves_no_weights_beyond_the_backbone(tmp_path):
    """A reranker-only parameter would break the stock load; there must be none.

    Also via ``AutoModelForCausalLM``, so a stray parameter is caught on the same path a
    consumer uses rather than one this test picked.
    """
    model = _tiny_model()
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert set(reloaded.state_dict()) == set(model.state_dict())


def test_supports_tied_embeddings():
    """The published checkpoints set tie_word_embeddings=True, so the tie must actually apply.

    Scoring reads the yes/no rows of ``lm_head``; a tie that silently failed would leave the
    head drifting from the embeddings it is meant to share, and the published checkpoint's
    scores would not reproduce.
    """
    config = Qwen3RerankerConfig(
        yes_token_id=YES_ID,
        no_token_id=NO_ID,
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        head_dim=8,
        tie_word_embeddings=True,
    )
    model = Qwen3RerankerForCausalReranking(config)

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_declares_tie_support_both():
    """Registered lm_head-bearing classes must state their policy rather than default to one."""
    assert Qwen3RerankerForCausalReranking.tie_word_embeddings_support is TieSupport.BOTH


def test_untied_config_leaves_the_head_independent():
    """BOTH means both layouts work, so the untied case must not alias."""
    config = Qwen3RerankerConfig(
        yes_token_id=YES_ID,
        no_token_id=NO_ID,
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        head_dim=8,
        tie_word_embeddings=False,
    )
    model = Qwen3RerankerForCausalReranking(config)

    assert model.lm_head.weight is not model.model.embed_tokens.weight
