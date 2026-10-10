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

import torch

from nemo_automodel.components.datasets.llm import megatron_dataset
from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import _get_ltor_masks_and_position_ids
from nemo_automodel.components.datasets.llm.megatron_dataset import MegatronPretraining, MegatronPretrainingConfig

EOD = 99


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

    def test_init_stores_eod_flags(self, monkeypatch, tmp_path):
        monkeypatch.setattr(megatron_dataset, "compile_helper", lambda: None)
        monkeypatch.setattr(megatron_dataset, "get_blend_from_list", lambda p: (["dummy"], None))
        monkeypatch.setattr(megatron_dataset, "validate_dataset_asset_accessibility", lambda *a, **k: None)
        mp = MegatronPretraining(
            paths=[str(tmp_path)],
            reset_position_ids=True,
            reset_attention_mask=True,
            eod_mask_loss=True,
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
        mp.create_attention_mask = False
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
