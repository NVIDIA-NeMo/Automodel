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

"""Unit tests for the ViSpec draft model, training objective, and target wrapper."""

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama_v12 import LlamaEagleDraftModel
from nemo_automodel.components.speculative.eagle.registry import (
    resolve_vispec_draft_spec,
    resolve_vispec_stage1_draft_spec,
)
from nemo_automodel.components.speculative.eagle.vispec_core import (
    VispecStepMetrics,
    VispecTrainerModule,
    listmle_loss,
)
from nemo_automodel.components.speculative.eagle.vispec_draft import VispecDraftModel, VispecImageAdaptor
from nemo_automodel.components.speculative.eagle.vispec_target import (
    HFVispecTargetModel,
    _shift_left_with_zero,
)

HIDDEN = 32
VOCAB = 64
IMAGE_TOKEN_ID = 7


def _draft_config(num_query_tokens: int = 2) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        draft_num_hidden_layers=1,
        vispec_num_query_tokens=num_query_tokens,
    )


def _draft(num_query_tokens: int = 2) -> VispecDraftModel:
    torch.manual_seed(0)
    return VispecDraftModel(_draft_config(num_query_tokens))


def _batch(seq_len: int = 12, image_span: tuple[int, int] = (2, 7)):
    """Build one sample: inputs_embeds / hidden / attention / image mask."""
    torch.manual_seed(1)
    inputs_embeds = torch.randn(1, seq_len, HIDDEN)
    hidden_states = torch.randn(1, seq_len, HIDDEN)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    image_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    image_mask[0, image_span[0] : image_span[1]] = True
    return inputs_embeds, hidden_states, attention_mask, image_mask


class TestImageAdaptor:
    def test_compresses_to_num_query_tokens(self):
        adaptor = VispecImageAdaptor(_draft_config(num_query_tokens=3), 3)
        out = adaptor(torch.randn(1, 57, HIDDEN))
        assert out.shape == (1, 3, HIDDEN)

    def test_output_is_independent_of_image_token_count(self):
        """The whole point of the adaptor: output length does not track input length."""
        adaptor = VispecImageAdaptor(_draft_config(), 2)
        short = adaptor(torch.randn(1, 4, HIDDEN))
        long = adaptor(torch.randn(1, 400, HIDDEN))
        assert short.shape == long.shape == (1, 2, HIDDEN)

    def test_rejects_single_query(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            VispecImageAdaptor(_draft_config(), 1)


class TestDraftModel:
    def test_output_keeps_original_sequence_layout(self):
        draft = _draft()
        out = draft(*_batch())
        assert out.shape == (1, 12, HIDDEN)

    def test_compressed_image_positions_are_zero_filled(self):
        """Only the last (num_query_tokens - 1) positions of a span survive; the
        rest are never supervised and must come back as exact zeros."""
        draft = _draft()
        inputs_embeds, hidden_states, attention_mask, image_mask = _batch(image_span=(2, 7))
        out = draft(inputs_embeds, hidden_states, attention_mask, image_mask)
        # span [2, 7): positions 2..5 are dropped, position 6 carries the
        # single spliced token (num_query_tokens - 1 == 1).
        assert torch.equal(out[0, 2:6], torch.zeros(4, HIDDEN))
        assert not torch.equal(out[0, 6], torch.zeros(HIDDEN))
        assert not torch.equal(out[0, 7], torch.zeros(HIDDEN))

    def test_text_only_sample_still_uses_the_adaptor(self):
        """Without the zero-weighted dummy term, ``img_adaptor`` would get no
        gradient on a text-only batch and DDP would error."""
        draft = _draft()
        inputs_embeds, hidden_states, attention_mask, _ = _batch()
        no_image = torch.zeros(1, inputs_embeds.shape[1], dtype=torch.bool)
        out = draft(inputs_embeds, hidden_states, attention_mask, no_image)
        out.sum().backward()
        assert draft.img_adaptor.query.grad is not None
        assert torch.equal(draft.img_adaptor.query.grad, torch.zeros_like(draft.img_adaptor.query))

    def test_multiple_image_spans(self):
        draft = _draft()
        inputs_embeds, hidden_states, attention_mask, image_mask = _batch(seq_len=16)
        image_mask = torch.zeros(1, 16, dtype=torch.bool)
        image_mask[0, 1:4] = True
        image_mask[0, 8:12] = True
        out = draft(inputs_embeds, hidden_states, attention_mask, image_mask)
        assert out.shape == (1, 16, HIDDEN)
        # Each span keeps only its final position.
        assert torch.equal(out[0, 1:3], torch.zeros(2, HIDDEN))
        assert torch.equal(out[0, 8:11], torch.zeros(3, HIDDEN))
        assert not torch.equal(out[0, 3], torch.zeros(HIDDEN))
        assert not torch.equal(out[0, 11], torch.zeros(HIDDEN))

    def test_rejects_batch_size_above_one(self):
        draft = _draft()
        inputs_embeds, hidden_states, attention_mask, image_mask = _batch()
        with pytest.raises(NotImplementedError, match="micro_batch_size=1"):
            draft(
                inputs_embeds.repeat(2, 1, 1),
                hidden_states.repeat(2, 1, 1),
                attention_mask.repeat(2, 1),
                image_mask.repeat(2, 1),
            )

    def test_rejects_span_shorter_than_spliced_tokens(self):
        draft = _draft(num_query_tokens=4)
        inputs_embeds, hidden_states, attention_mask, _ = _batch()
        image_mask = torch.zeros(1, inputs_embeds.shape[1], dtype=torch.bool)
        image_mask[0, 5] = True  # a 1-token span, but 3 tokens must be spliced back
        with pytest.raises(ValueError, match="only 1 image token"):
            draft(inputs_embeds, hidden_states, attention_mask, image_mask)

    def test_img_fc_starts_as_identity_pass_through(self):
        """Stage 2 must start numerically equal to the stage-1 draft it loads."""
        config = _draft_config()
        torch.manual_seed(0)
        vispec = VispecDraftModel(config)
        torch.manual_seed(0)
        eagle = LlamaEagleDraftModel(config)
        eagle.load_state_dict(
            {k: v for k, v in vispec.state_dict().items() if not k.startswith(("img_adaptor.", "img_fc."))}
        )

        inputs_embeds, hidden_states, attention_mask, _ = _batch()
        no_image = torch.zeros(1, inputs_embeds.shape[1], dtype=torch.bool)
        with torch.no_grad():
            vispec_out = vispec(inputs_embeds, hidden_states, attention_mask, no_image)
            # The EAGLE draft embeds token ids; feed it the same features by
            # calling its post-embedding path with identical inputs.
            eagle_hidden = eagle.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
            position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
            from nemo_automodel.components.speculative.eagle.draft_llama_v12 import _build_causal_mask

            causal = _build_causal_mask(attention_mask, eagle_hidden.dtype)
            for layer in eagle.layers:
                eagle_hidden = layer(eagle_hidden, attention_mask=causal, position_ids=position_ids)
            eagle_out = eagle.norm(eagle_hidden)
        torch.testing.assert_close(vispec_out, eagle_out)

    def test_stage1_checkpoint_loads_non_strictly(self):
        config = _draft_config()
        stage1 = LlamaEagleDraftModel(config)
        draft = VispecDraftModel(config)
        missing, unexpected = draft.load_state_dict(stage1.state_dict(), strict=False)
        assert unexpected == []
        assert all(name.startswith(("img_adaptor.", "img_fc.")) for name in missing)
        torch.testing.assert_close(draft.fc.weight, stage1.fc.weight)

    @pytest.mark.parametrize(
        "architecture",
        [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
        ],
    )
    def test_stage1_registry_returns_the_plain_eagle_draft(self, architecture):
        draft_spec = resolve_vispec_stage1_draft_spec([architecture])
        assert draft_spec.draft_cls is LlamaEagleDraftModel


class TestListMLE:
    def test_zero_when_ordering_matches_and_gaps_are_large(self):
        target_probs = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        logits = torch.tensor([[60.0, 40.0, 20.0, 0.0]])
        assert listmle_loss(logits, target_probs, topk=3).item() == pytest.approx(0.0, abs=1e-5)

    def test_penalizes_inverted_ordering(self):
        target_probs = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        aligned = listmle_loss(torch.tensor([[6.0, 4.0, 2.0, 0.0]]), target_probs, topk=3)
        inverted = listmle_loss(torch.tensor([[2.0, 4.0, 6.0, 0.0]]), target_probs, topk=3)
        assert inverted.item() > aligned.item()

    def test_matches_explicit_plackett_luce_formula(self):
        target_probs = torch.tensor([[0.5, 0.3, 0.2]])
        logits = torch.tensor([[1.0, 0.5, -0.5]])
        expected = 0.0
        for rank in range(3):
            tail = logits[0, rank:]
            expected -= (logits[0, rank] - torch.logsumexp(tail, dim=-1)).item()
        assert listmle_loss(logits, target_probs, topk=3).item() == pytest.approx(expected, abs=1e-5)


class TestTrainerModule:
    def _module(self, mtp_steps: int = 1) -> VispecTrainerModule:
        lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        lm_head.requires_grad_(False)
        return VispecTrainerModule(_draft(), target_lm_head=lm_head, mtp_steps=mtp_steps)

    def _inputs(self, seq_len: int = 12):
        inputs_embeds, hidden_states, attention_mask, image_mask = _batch(seq_len=seq_len)
        loss_mask = torch.zeros(1, seq_len, dtype=torch.long)
        loss_mask[0, 8:] = 1
        target_logits = torch.randn(1, seq_len, VOCAB)
        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            input_hidden_states=hidden_states,
            target_logits=target_logits,
            image_mask=image_mask,
        )

    def test_returns_finite_metrics_and_trains(self):
        module = self._module()
        metrics = module(**self._inputs())
        assert isinstance(metrics, VispecStepMetrics)
        assert torch.isfinite(metrics.loss)
        assert metrics.valid_tokens.item() == 4
        assert 0.0 <= metrics.accuracy.item() <= 1.0

        metrics.loss.backward()
        assert module.draft_model.img_fc.weight.grad is not None
        assert module.draft_model.img_fc.weight.grad.abs().sum() > 0

    def test_rollouts_change_the_loss(self):
        """``mtp_steps`` must actually add supervised passes, not be a no-op flag."""
        inputs = self._inputs()
        torch.manual_seed(3)
        without = self._module(mtp_steps=0)(**inputs).loss
        torch.manual_seed(3)
        with_rollout = self._module(mtp_steps=2)(**inputs).loss
        assert not torch.isclose(without, with_rollout)

    def test_rejects_negative_mtp_steps(self):
        with pytest.raises(ValueError, match="mtp_steps must be >= 0"):
            VispecTrainerModule(_draft(), target_lm_head=nn.Linear(HIDDEN, VOCAB), mtp_steps=-1)

    def test_empty_loss_mask_gives_finite_zero_loss(self):
        module = self._module()
        inputs = self._inputs()
        inputs["loss_mask"] = torch.zeros_like(inputs["loss_mask"])
        metrics = module(**inputs)
        assert metrics.loss.item() == 0.0
        assert torch.isfinite(metrics.loss)
        metrics.loss.backward()
        assert module.draft_model.img_fc.weight.grad is not None

    def test_loss_weights_are_applied(self):
        inputs = self._inputs()
        lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        draft = _draft()
        weighted = VispecTrainerModule(
            draft, target_lm_head=lm_head, prob_loss_weight=2.0, rank_loss_weight=0.5, mtp_steps=0
        )(**inputs)
        expected = 2.0 * weighted.prob_loss + 0.5 * weighted.rank_loss
        torch.testing.assert_close(weighted.loss, expected)

    def test_prob_loss_is_bounded_by_two(self):
        """L1 between two probability vectors is at most 2 by construction."""
        metrics = self._module()(**self._inputs())
        assert 0.0 <= metrics.prob_loss.item() <= 2.0


class _StubBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)

    def forward(self, input_ids, attention_mask, output_hidden_states=False, pixel_values=None, **kwargs):
        """Return HF-shaped hidden states.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence].
            output_hidden_states: Whether to return the per-layer states.
            pixel_values: Tensor of shape [patches, patch_dim], unused by the stub.

        Returns:
            An object whose ``hidden_states`` is a 2-tuple of [batch, sequence, hidden].
        """
        embeds = self.embed(input_ids)
        from types import SimpleNamespace

        return SimpleNamespace(hidden_states=(embeds, embeds * 2.0))


class _StubTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _StubBaseModel()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_input_embeddings(self):
        return self.model.embed


class TestTargetWrapper:
    def test_shift_left_with_zero(self):
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
        torch.testing.assert_close(_shift_left_with_zero(tensor), torch.tensor([[[2.0], [3.0], [0.0]]]))

    def test_generate_batch_alignment(self):
        target = _StubTarget()
        wrapper = HFVispecTargetModel(target, image_token_id=IMAGE_TOKEN_ID)
        input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.tensor([[0, 0, 0, 1, 1]])

        batch = wrapper.generate_batch(input_ids, attention_mask, loss_mask)
        # image_mask is shifted so index i answers "is token i+1 an image token?"
        assert batch.image_mask.tolist() == [[True, True, False, False, False]]
        assert batch.loss_mask.tolist() == [[0, 0, 1, 1, 0]]

        embeds = target.model.embed(input_ids)
        # The draft's features: shifted embeddings, unshifted last hidden state.
        torch.testing.assert_close(batch.inputs_embeds, _shift_left_with_zero(embeds))
        torch.testing.assert_close(batch.input_hidden_states, embeds * 2.0)
        torch.testing.assert_close(batch.target_logits, _shift_left_with_zero(target.lm_head(embeds * 2.0)))

    def test_drops_unsupported_multimodal_kwargs(self):
        """The stub's forward has no ``image_grid_thw``; passing it must not raise."""
        wrapper = HFVispecTargetModel(_StubTarget(), image_token_id=IMAGE_TOKEN_ID)
        input_ids = torch.tensor([[1, 2, 3]])
        batch = wrapper.generate_batch(
            input_ids,
            torch.ones_like(input_ids),
            torch.ones_like(input_ids),
            pixel_values=torch.randn(4, 8),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        assert batch.target_logits.shape == (1, 3, VOCAB)


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    ],
)
def test_registry_resolves_supported_qwen_vl(architecture):
    spec = resolve_vispec_draft_spec([architecture])
    assert spec.draft_cls is VispecDraftModel


def test_registry_rejects_unregistered_architecture():
    with pytest.raises(ValueError, match="TrainVispecRecipe"):
        resolve_vispec_draft_spec(["LlamaForCausalLM"])
