# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.datasets.datum import (
    CROSS_ENTROPY_IGNORE_IDX,
    Datum,
    collate_datums,
)


def _toy_datums():
    return [
        Datum(
            input_ids=torch.tensor([10, 11, 12]),
            loss_inputs={
                "target_tokens": torch.tensor([11, 12, 13]),
                "weights": torch.tensor([1.0, 1.0, 0.0]),
                "advantages": torch.tensor([0.5, 0.5, 0.5]),
            },
        ),
        Datum(
            input_ids=torch.tensor([20, 21]),
            loss_inputs={
                "target_tokens": torch.tensor([21, 22]),
                "weights": torch.tensor([1.0, 1.0]),
                "advantages": torch.tensor([0.9, 0.9]),
            },
        ),
    ]


# ── Datum ─────────────────────────────────────────────────────────────────


def test_datum_coerces_and_validates():
    d = Datum(input_ids=[1, 2, 3], loss_inputs={"weights": [1, 1, 0]})
    assert isinstance(d.input_ids, torch.Tensor)
    assert d.input_ids.dtype == torch.long
    assert d.seq_len == 3
    assert isinstance(d.loss_inputs["weights"], torch.Tensor)


def test_datum_rejects_non_1d_input_ids():
    with pytest.raises(ValueError, match="must be 1-D"):
        Datum(input_ids=torch.zeros(2, 3, dtype=torch.long))


def test_datum_to_device_is_a_copy():
    d = Datum(input_ids=torch.tensor([1, 2]), loss_inputs={"weights": torch.tensor([1.0, 1.0])})
    moved = d.to("cpu")
    assert moved is not d
    assert torch.equal(moved.input_ids, d.input_ids)


def test_datum_dict_roundtrip():
    d = Datum(input_ids=torch.tensor([5, 6, 7]), loss_inputs={"advantages": torch.tensor([0.1, 0.2, 0.3])})
    d2 = Datum.from_dict(d.to_dict())
    assert torch.equal(d.input_ids, d2.input_ids)
    assert torch.equal(d.loss_inputs["advantages"], d2.loss_inputs["advantages"])


# ── Datum.to_features ───────────────────────────────────────────────────────


def test_to_features_applies_masking_convention():
    feats = _toy_datums()[0].to_features()
    assert feats["input_ids"] == [10, 11, 12]
    # weights==0 at position 2 -> ignore_index in labels.
    assert feats["labels"] == [11, 12, CROSS_ENTROPY_IGNORE_IDX]


def test_to_features_omits_labels_without_targets():
    feats = Datum(input_ids=torch.tensor([1, 2, 3])).to_features()
    assert feats == {"input_ids": [1, 2, 3]}


def test_to_features_native_python_ints():
    # Collaters call torch.LongTensor(...) on these, so they must be plain ints.
    feats = _toy_datums()[0].to_features()
    assert all(isinstance(x, int) for x in feats["input_ids"])
    assert all(isinstance(x, int) for x in feats["labels"])


# ── collate_datums delegates to the canonical collaters ─────────────────────


def test_collate_padded_uses_default_collater_schema():
    batch = collate_datums(_toy_datums())
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [20, 21, 0]  # right-pad
    assert batch["labels"][0].tolist() == [11, 12, CROSS_ENTROPY_IGNORE_IDX]
    assert batch["labels"][1].tolist() == [21, 22, CROSS_ENTROPY_IGNORE_IDX]
    # padding_mask is produced by default_collater, not by us.
    assert "padding_mask" in batch


def test_collate_packed_uses_thd_collater_schema():
    batch = collate_datums(_toy_datums(), packed=True)
    # qkv_format + seq_lens come straight from packed_sequence_thd_collater.
    assert batch["qkv_format"] == "thd"
    assert batch["input_ids"].shape == (2, 3)
    assert "seq_lens" in batch and "seq_lens_padded" in batch
    assert batch["labels"][0].tolist() == [11, 12, CROSS_ENTROPY_IGNORE_IDX]


def test_collate_carries_per_token_float_side_inputs():
    # advantages is float per-token -> padded to collated width and stacked.
    batch = collate_datums(_toy_datums())
    assert "advantages" in batch
    assert batch["advantages"].dtype == torch.float
    assert batch["advantages"].shape == (2, 3)
    assert batch["advantages"][1].tolist() == pytest.approx([0.9, 0.9, 0.0])  # right-pad with 0


def test_collate_per_sample_scalar_side_input():
    datums = [
        Datum(input_ids=torch.tensor([1, 2]), loss_inputs={"advantages": torch.tensor([0.5])}),
        Datum(input_ids=torch.tensor([3, 4, 5]), loss_inputs={"advantages": torch.tensor([0.9])}),
    ]
    batch = collate_datums(datums)
    # length-1 != seq_len -> treated as per-sample, one value per datum.
    assert batch["advantages"].shape == (2,)
    assert batch["advantages"].tolist() == pytest.approx([0.5, 0.9])


def test_collate_pad_seq_len_divisible():
    batch = collate_datums(_toy_datums(), pad_seq_len_divisible=8)
    assert batch["input_ids"].shape == (2, 8)
    # float side-inputs follow the collated width.
    assert batch["advantages"].shape == (2, 8)


def test_collate_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        collate_datums([])
