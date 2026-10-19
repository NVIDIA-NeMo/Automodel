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

"""Unit tests for Megatron document-boundary features (upstream issue #4157).

Covers ``_get_ltor_masks_and_position_ids`` with a real EOD token id and the
config threading of ``reset_position_ids`` / ``reset_attention_mask`` /
``eod_mask_loss`` through ``MegatronPretrainingConfig`` and
``MegatronPretraining``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_automodel.components.datasets.llm import megatron_dataset
from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    Split,
    _get_ltor_masks_and_position_ids,
)
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining, MegatronPretrainingConfig
from nemo_automodel.components.datasets.loader import DatasetBuildSchedule
from nemo_automodel.components.datasets.utils import default_collater
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.utils import _count_label_tokens, calculate_loss

EOD = 99


@pytest.mark.parametrize("mask_kind", ["padding_2d", "blocked_3d", "additive_4d"])
def test_collater_preserves_existing_attention_mask_layout_and_padding(mask_kind):
    if mask_kind == "padding_2d":
        mask = torch.tensor([[1, 1, 0]])
    else:
        blocked = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        mask = blocked.unsqueeze(0)
        if mask_kind == "additive_4d":
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, float("-inf")).unsqueeze(0)
    batch = default_collater([{"input_ids": torch.tensor([5, EOD, 0]), "attention_mask": mask} for _ in range(2)])
    expected = torch.cat([mask, mask], dim=0)
    torch.testing.assert_close(batch["attention_mask"], expected, rtol=0, atol=0)
    torch.testing.assert_close(batch["padding_mask"], ~expected.bool(), rtol=0, atol=0)


def _tokens():
    # Two documents: [5, EOD] and [7, 8, EOD], then a partial third: [9].
    return torch.tensor([5, EOD, 7, 8, EOD, 9])


class TestGetLtorMasksAndPositionIds:
    """Behavior of document-boundary features with a real EOD token id."""

    def test_eod_mask_loss_zeros_loss_at_eod_positions(self):
        _, loss_mask, _ = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=True,
            create_attention_mask=False,
        )
        assert loss_mask.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    def test_reset_position_ids_restarts_after_each_eod(self):
        _, _, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=True,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=False,
        )
        assert position_ids.tolist() == [0, 1, 0, 1, 2, 0]

    def test_reset_attention_mask_blocks_cross_document_attention(self):
        attention_mask, _, _ = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=True,
            eod_mask_loss=False,
            create_attention_mask=True,
        )
        # Bool mask, True = masked. Tokens after the first EOD (rows 2+)
        # must not attend to the first document (cols 0-1).
        assert attention_mask.shape == (1, 6, 6)
        assert attention_mask[0, 2, 0].item() is True
        assert attention_mask[0, 2, 1].item() is True
        # Within-document attention is untouched.
        assert attention_mask[0, 1, 0].item() is False
        assert attention_mask[0, 3, 2].item() is False

    def test_all_flags_off_matches_legacy_behavior(self):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            EOD,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=True,
        )
        assert loss_mask.tolist() == [1.0] * 6
        assert position_ids.tolist() == [0, 1, 2, 3, 4, 5]
        # Plain causal mask: lower triangle (incl. diagonal) unmasked.
        assert attention_mask[0, 3, 2].item() is False
        assert attention_mask[0, 2, 3].item() is True

    def test_unmatched_eod_token_is_noop(self):
        # The historical -10000 sentinel never occurs: features stay no-ops.
        _, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            _tokens(),
            -10000,
            reset_position_ids=True,
            reset_attention_mask=False,
            eod_mask_loss=True,
            create_attention_mask=False,
        )
        assert loss_mask.tolist() == [1.0] * 6
        assert position_ids.tolist() == [0, 1, 2, 3, 4, 5]


class TestEodConfigThreading:
    """The three flags travel from MegatronPretrainingConfig to GPTDatasetConfig."""

    def test_config_defaults_are_backward_compatible(self):
        config = MegatronPretrainingConfig(paths="dummy")
        assert config.reset_position_ids is False
        assert config.reset_attention_mask is False
        assert config.eod_mask_loss is False

    def test_existing_positional_config_arguments_are_preserved(self):
        config = MegatronPretrainingConfig("dummy", 8, False, 17, "1,0,0")
        assert config.seed == 17
        assert config.split == "1,0,0"
        assert config.reset_position_ids is False

    def test_existing_positional_init_arguments_are_preserved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(megatron_dataset, "compile_helper", lambda: None)
        monkeypatch.setattr(megatron_dataset, "get_blend_from_list", lambda p: (["dummy"], None))
        monkeypatch.setattr(megatron_dataset, "validate_dataset_asset_accessibility", lambda *a, **k: None)
        mp = MegatronPretraining([str(tmp_path)], 8, None, 4, 8, False, 17, "1,0,0")
        assert mp.seed == 17
        assert mp.split == "1,0,0"
        assert mp.reset_position_ids is False

    def test_init_stores_eod_flags(self, monkeypatch, tmp_path):
        monkeypatch.setattr(megatron_dataset, "compile_helper", lambda: None)
        monkeypatch.setattr(megatron_dataset, "get_blend_from_list", lambda p: (["dummy"], None))
        monkeypatch.setattr(megatron_dataset, "validate_dataset_asset_accessibility", lambda *a, **k: None)
        mp = MegatronPretraining(
            paths=[str(tmp_path)],
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
            create_attention_mask=True,
        )
        assert mp.reset_position_ids is True
        assert mp.reset_attention_mask is True
        assert mp.eod_mask_loss is True

    def test_gpt_dataset_config_forwards_eod_flags(self, monkeypatch):
        captured = {}

        class FakeGPTDatasetConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(megatron_dataset, "GPTDatasetConfig", FakeGPTDatasetConfig)
        mp = MegatronPretraining.__new__(MegatronPretraining)
        mp.seed = 1234
        mp.seq_length = 2048
        mp.tokenizer = SimpleNamespace(name_or_path="dummy")
        mp.index_mapping_dir = None
        mp.create_attention_mask = True
        mp.reset_position_ids = True
        mp.reset_attention_mask = True
        mp.eod_mask_loss = True
        mp.num_dataset_builder_threads = 1
        mp.object_storage_config = None
        mp.build_kwargs = {}

        mp.gpt_dataset_config

        assert captured["reset_position_ids"] is True
        assert captured["reset_attention_mask"] is True
        assert captured["eod_mask_loss"] is True

    def test_config_build_constructs_real_gpt_config_with_eod_flags(self, monkeypatch, tmp_path):
        monkeypatch.setattr(megatron_dataset, "compile_helper", lambda: None)
        monkeypatch.setattr(megatron_dataset, "get_blend_from_list", lambda p: (["dummy"], None))
        monkeypatch.setattr(megatron_dataset, "validate_dataset_asset_accessibility", lambda *a, **k: None)
        monkeypatch.setattr(MegatronPretraining, "build", lambda self: None)
        monkeypatch.setattr(MegatronPretraining, "get_dataset", lambda self, split: self.gpt_dataset_config)
        config = MegatronPretrainingConfig(
            paths=[str(tmp_path)],
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
            create_attention_mask=True,
        )
        result = config.build(
            tokenizer=SimpleNamespace(name_or_path="dummy", eos_token_id=EOD, pad_token_id=0),
            training_schedule=DatasetBuildSchedule(
                local_batch_size=1,
                global_batch_size=1,
                max_steps=2,
                val_check_interval=None,
            ),
        )
        assert isinstance(result, GPTDatasetConfig)
        assert result.reset_position_ids is True
        assert result.reset_attention_mask is True
        assert result.eod_mask_loss is True
        assert result.create_attention_mask is True


@pytest.mark.parametrize("eos_token_id", [EOD, [EOD, 98], (EOD, 98), None, [], "99"])
def test_constructor_resolves_eod_from_tokenizer_in_actual_samples(eos_token_id, monkeypatch, caplog):
    text = np.array([5, EOD, 7, 8, EOD, 9], dtype=np.int64)
    monkeypatch.setattr(GPTDataset, "_build_document_sample_shuffle_indices", lambda self: (None, None, None))
    monkeypatch.setattr(GPTDataset, "_query_document_sample_shuffle_indices", lambda self, idx: (text, None))
    dataset = GPTDataset(
        indexed_dataset=SimpleNamespace(),
        dataset_path=None,
        indexed_indices=np.array([0]),
        num_samples=1,
        index_split=Split.train,
        config=GPTDatasetConfig(
            random_seed=0,
            sequence_length=5,
            tokenizer=SimpleNamespace(name_or_path="dummy", eos_token_id=eos_token_id, pad_token_id=0),
            reset_position_ids=True,
            reset_attention_mask=False,
            eod_mask_loss=True,
            create_attention_mask=False,
        ),
    )
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [5, EOD, 7, 8, EOD]
    if eos_token_id in (None, [], "99"):
        assert sample["labels"].tolist() == [EOD, 7, 8, EOD, 9]
        assert sample["position_ids"].tolist() == [0, 1, 2, 3, 4]
        assert "no usable eos_token_id" in caplog.text
    else:
        assert sample["labels"].tolist() == [EOD, -100, 8, EOD, -100]
        assert sample["position_ids"].tolist() == [0, 1, 0, 1, 2]


class TestGetItemDocumentBoundaryOutput:
    """__getitem__ must surface the newly enabled features in the sample dict."""

    def _dataset(self, *, text=None, pad_token_id=0, eod_token_id=EOD, add_extra_token=True, **flags):
        ds = GPTDataset.__new__(GPTDataset)
        text = np.array([5, EOD, 7, 8, EOD, 9] if text is None else text, dtype=np.int64)

        def query(idx):
            return text, None

        ds._query_document_sample_shuffle_indices = query
        ds.config = SimpleNamespace(
            add_extra_token_to_sequence=add_extra_token,
            reset_position_ids=flags.get("reset_position_ids", False),
            reset_attention_mask=flags.get("reset_attention_mask", False),
            eod_mask_loss=flags.get("eod_mask_loss", False),
            create_attention_mask=flags.get("create_attention_mask", False),
        )
        ds._pad_token_id = pad_token_id
        ds._eos_token_id = eod_token_id
        ds._eod_token_id = eod_token_id
        ds.masks_and_position_ids_are_cacheable = False
        ds.masks_and_position_ids_are_cached = False
        return ds

    @pytest.mark.parametrize("create_attention_mask", [False, True])
    def test_position_ids_returned_when_reset_enabled(self, create_attention_mask):
        sample = self._dataset(reset_position_ids=True, create_attention_mask=create_attention_mask)[0]
        assert "position_ids" in sample
        # tokens are [5, EOD, 7, 8, EOD]: positions restart after each EOD.
        assert sample["position_ids"].tolist() == [0, 1, 0, 1, 2]
        assert sample["attention_mask"].tolist() == [1, 1, 1, 1, 1]
        padded = self._dataset(
            text=[5, EOD, 7, 0, 0, 0], reset_position_ids=True, create_attention_mask=create_attention_mask
        )[0]
        assert padded["attention_mask"].tolist() == [1, 1, 1, 0, 0]

    def test_position_ids_absent_when_reset_disabled(self):
        sample = self._dataset()[0]
        assert "position_ids" not in sample
        assert "attention_mask" not in sample

    def test_eod_mask_loss_encodes_ignore_index_in_labels(self):
        sample = self._dataset(eod_mask_loss=True)[0]
        assert sample["loss_mask"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]
        # The helper masks loss_mask at input-EOD positions (1, 4), matching
        # Megatron semantics. Those label positions must not contribute to the
        # CE loss or to _count_label_tokens normalization, which both key off
        # -100.
        assert sample["labels"].tolist() == [EOD, -100, 8, EOD, -100]
        assert sample["input_ids"].tolist() == [5, EOD, 7, 8, EOD]

    def test_eod_mask_loss_does_not_mutate_source_or_repeated_samples(self):
        dataset = self._dataset(eod_mask_loss=True, reset_position_ids=True)
        source, _ = dataset._query_document_sample_shuffle_indices(0)
        original = source.copy()
        first = dataset[0]
        np.testing.assert_array_equal(source, original)
        second = dataset[0]
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_eod_mask_loss_without_extra_token_preserves_input(self):
        sample = self._dataset(eod_mask_loss=True, add_extra_token=False)[0]
        assert sample["input_ids"].tolist() == [5, EOD, 7, 8, EOD, 9]
        assert sample["labels"].tolist() == [EOD, -100, 8, EOD, -100, 0]

    def test_eod_mask_loss_without_eod_id_is_noop(self):
        sample = self._dataset(eod_mask_loss=True, eod_token_id=None)[0]
        assert sample["input_ids"].tolist() == [5, EOD, 7, 8, EOD]
        assert sample["labels"].tolist() == [EOD, 7, 8, EOD, 9]

    @pytest.mark.parametrize("ignore_index", [-100, -7])
    def test_eod_mask_reaches_loss_gradients_and_token_count(self, ignore_index):
        batch = torch.utils.data.default_collate([self._dataset(eod_mask_loss=True)[0]])
        logits = torch.linspace(-1, 1, 500).reshape(1, 5, 100).requires_grad_()
        count = _count_label_tokens(batch["labels"], ignore_index)
        assert count == 3
        loss = calculate_loss(
            MaskedCrossEntropy(ignore_index=ignore_index),
            logits=logits,
            labels=batch["labels"],
            num_label_tokens=count,
        )
        reference = torch.nn.functional.cross_entropy(logits[0, [0, 2, 3]], torch.tensor([EOD, 8, EOD]))
        torch.testing.assert_close(loss, reference)
        expected_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
        loss.backward()
        torch.testing.assert_close(logits.grad, expected_gradient)
        assert torch.count_nonzero(logits.grad[0, [1, 4]]).item() == 0
        assert torch.count_nonzero(logits.grad[0, [0, 2, 3]]).item() > 0

    @pytest.mark.parametrize("collater", [torch.utils.data.default_collate, default_collater])
    @pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
    def test_attention_reset_is_effective_in_gpt2(self, collater, attention_implementation):
        from transformers import GPT2Config, GPT2LMHeadModel

        sample = self._dataset(
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
            create_attention_mask=True,
        )[0]
        batch = collater([sample, sample])
        assert batch["attention_mask"].shape == (2, 1, 5, 5)
        if "padding_mask" in batch:
            assert batch["padding_mask"].shape == (2, 5)
        config = GPT2Config(
            vocab_size=100,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = attention_implementation
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            model = GPT2LMHeadModel(config).eval()
        inputs = {key: batch[key] for key in ("input_ids", "position_ids", "attention_mask")}
        with torch.no_grad():
            output = model(**inputs).logits
            inputs["input_ids"] = inputs["input_ids"].clone()
            inputs["input_ids"][:, 0] = 6
            changed = model(**inputs).logits
            isolated = model(input_ids=torch.tensor([[7, 8, EOD]])).logits
        torch.testing.assert_close(output[:, 2:], changed[:, 2:], rtol=0, atol=0)
        torch.testing.assert_close(output[0, 2:], isolated[0], rtol=1e-5, atol=1e-6)

        # The output head is untied so this row can receive gradients only through
        # the input token's participation in attention, not as an output class.
        embedding = model.get_input_embeddings().weight
        assert embedding.data_ptr() != model.get_output_embeddings().weight.data_ptr()
        inputs["input_ids"] = batch["input_ids"]
        for reset_attention in (False, True):
            model.zero_grad(set_to_none=True)
            inputs["attention_mask"] = batch["attention_mask"] if reset_attention else None
            logits = model(**inputs).logits[:, 2:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 100), batch["labels"][:, 2:].reshape(-1))
            loss.backward()
            assert torch.count_nonzero(embedding.grad[7:9]).item() > 0
            # Token 5 occurs exclusively in document one. The causal control
            # must detect cross-document flow; resetting attention must stop it.
            assert (torch.count_nonzero(embedding.grad[5]).item() == 0) is reset_attention

    def test_labels_untouched_when_eod_mask_loss_disabled(self):
        sample = self._dataset()[0]
        assert sample["labels"].tolist() == [EOD, 7, 8, EOD, 9]

    def test_reset_attention_mask_requires_create_attention_mask(self):
        with pytest.raises(ValueError, match="create_attention_mask"):
            GPTDatasetConfig(
                random_seed=0,
                sequence_length=8,
                tokenizer=SimpleNamespace(eos_token_id=EOD, pad_token_id=0),
                reset_position_ids=False,
                reset_attention_mask=True,
                eod_mask_loss=False,
                create_attention_mask=False,
            )
