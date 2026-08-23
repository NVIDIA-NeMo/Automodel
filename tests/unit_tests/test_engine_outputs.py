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

from nemo_automodel import LossOutput as PublicLossOutput
from nemo_automodel.engine.outputs import LossOutput


def test_loss_output_is_a_public_lazy_export() -> None:
    assert PublicLossOutput is LossOutput


def test_loss_output_copies_containers_but_retains_values() -> None:
    logprobs = torch.randn(5, requires_grad=True)
    token_fields = {"logprobs": logprobs}

    output = LossOutput(
        loss_sum=torch.tensor(1.0),
        token_outputs=token_fields,
    )

    token_fields["entropy"] = torch.ones(5)

    assert dict(output.token_outputs) == {"logprobs": logprobs}
    assert output.token_outputs["logprobs"] is logprobs
    with pytest.raises(TypeError):
        output.token_outputs["another"] = torch.ones(5)  # type: ignore[index]


def test_loss_output_requires_an_output_channel() -> None:
    with pytest.raises(ValueError, match="requires token_outputs or batch_output"):
        LossOutput(loss_sum=torch.tensor(1.0))


def test_loss_output_batch_channel_is_read_only_and_separate() -> None:
    metric = torch.tensor(2.0, requires_grad=True)
    batch_output = {"metrics": {"reward": metric}}

    output = LossOutput(loss_sum=torch.tensor(1.0), batch_output=batch_output)
    batch_output["model_output"] = torch.ones(1)

    assert dict(output.batch_output) == {"metrics": {"reward": metric}}
    with pytest.raises(TypeError):
        output.batch_output["another"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"loss_sum": torch.ones(2)}, TypeError, "scalar torch.Tensor"),
        ({"loss_sum": 1.0}, TypeError, "scalar torch.Tensor"),
        ({"loss_sum": torch.tensor(1.0), "token_outputs": {"x": torch.tensor(1.0)}}, TypeError, "non-scalar"),
    ],
)
def test_loss_output_rejects_invalid_tensors(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        LossOutput(**kwargs)


@pytest.mark.parametrize(
    ("token_outputs", "error", "match"),
    [
        ({"": torch.ones(1)}, TypeError, "field names must be non-empty strings"),
        ({1: torch.ones(1)}, TypeError, "field names must be non-empty strings"),
        ({"logprobs": [1.0]}, TypeError, "non-scalar torch.Tensor"),
    ],
)
def test_loss_output_rejects_invalid_token_fields(token_outputs, error, match) -> None:
    with pytest.raises(error, match=match):
        LossOutput(loss_sum=torch.tensor(1.0), token_outputs=token_outputs)
