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

from __future__ import annotations

import pytest
import torch

from nemo_automodel import LossFnOutputBatch as PublicLossFnOutputBatch
from nemo_automodel import PerTokenOutput as PublicPerTokenOutput
from nemo_automodel.engine.outputs import LossFnOutputBatch, PerTokenOutput


def test_output_types_are_public_lazy_exports() -> None:
    assert PublicPerTokenOutput is PerTokenOutput
    assert PublicLossFnOutputBatch is LossFnOutputBatch


def test_loss_fn_output_batch_copies_containers_but_retains_values() -> None:
    logprobs = torch.randn(5, requires_grad=True)
    sample_id = torch.tensor(7)
    token_output = PerTokenOutput(logprobs, fill_value=-1.0)
    token_fields = {"logprobs": token_output}
    record = {"sample_id": sample_id}
    records = [record]

    output = LossFnOutputBatch(per_token=token_fields, per_datum=records)

    token_fields["entropy"] = PerTokenOutput(torch.ones(5))
    record["new_field"] = 3
    records.append({"sample_id": torch.tensor(8)})

    assert tuple(output.per_token) == ("logprobs",)
    assert output.per_token["logprobs"] is token_output
    assert output.per_token["logprobs"].tensor is logprobs
    assert output.per_token["logprobs"].fill_value == -1.0
    assert output.per_datum is not None
    assert len(output.per_datum) == 1
    assert dict(output.per_datum[0]) == {"sample_id": sample_id}
    assert output.per_datum[0]["sample_id"] is sample_id

    with pytest.raises(TypeError):
        output.per_token["another"] = PerTokenOutput(torch.ones(5))  # type: ignore[index]
    with pytest.raises(TypeError):
        output.per_datum[0]["another"] = 1  # type: ignore[index]


def test_loss_fn_output_batch_preserves_none_per_datum() -> None:
    output = LossFnOutputBatch(per_token={"logprobs": PerTokenOutput(torch.ones(3))})

    assert output.per_datum is None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"tensor": [1.0]}, TypeError, "tensor must be a torch.Tensor"),
        ({"tensor": torch.tensor(1.0)}, ValueError, "at least one dimension"),
        ({"tensor": torch.ones(1), "fill_value": "zero"}, TypeError, "fill_value must be an int or float"),
    ],
)
def test_per_token_output_rejects_invalid_values(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        PerTokenOutput(**kwargs)


@pytest.mark.parametrize(
    ("per_token", "error", "match"),
    [
        ([], TypeError, "per_token must be a mapping"),
        ({}, ValueError, "per_token cannot be empty"),
        ({"": PerTokenOutput(torch.ones(1))}, TypeError, "field names must be non-empty strings"),
        ({1: PerTokenOutput(torch.ones(1))}, TypeError, "field names must be non-empty strings"),
        ({"logprobs": torch.ones(1)}, TypeError, "must be a PerTokenOutput"),
    ],
)
def test_loss_fn_output_batch_rejects_invalid_token_fields(per_token, error, match) -> None:
    with pytest.raises(error, match=match):
        LossFnOutputBatch(per_token=per_token)


@pytest.mark.parametrize(
    ("per_datum", "error", "match"),
    [
        ({"sample_id": 1}, TypeError, "per_datum must be a sequence of mappings"),
        ("record", TypeError, "per_datum must be a sequence of mappings"),
        ([1], TypeError, "record 0 must be a mapping"),
        ([{"": 1}], TypeError, "field names must be non-empty strings"),
        ([{1: "sample"}], TypeError, "field names must be non-empty strings"),
        ([{"logprobs": torch.ones(1)}], ValueError, "conflicts with per_token fields"),
    ],
)
def test_loss_fn_output_batch_rejects_invalid_per_datum_records(per_datum, error, match) -> None:
    with pytest.raises(error, match=match):
        LossFnOutputBatch(
            per_token={"logprobs": PerTokenOutput(torch.ones(1))},
            per_datum=per_datum,
        )
