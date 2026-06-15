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
"""Pure unit tests for the built-in LossFns (no model / no Engine)."""

import pytest
import torch

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.model_output import ModelOutput
from nemo_automodel.loss_fns import BUILTIN_LOSSES, cross_entropy


def test_registry_keys():
    # Only role-agnostic losses ship here; RL objectives (PPO, IS) are caller-supplied.
    assert set(BUILTIN_LOSSES) == {"cross_entropy"}


def test_cross_entropy_is_negative_logprob():
    mo = ModelOutput(logprobs=[torch.tensor([-0.5, -1.0]), torch.tensor([-2.0])])
    datums = [Datum(torch.tensor([1, 2])), Datum(torch.tensor([3]))]
    out = cross_entropy(mo, datums)
    assert torch.equal(out[0], torch.tensor([0.5, 1.0]))
    assert torch.equal(out[1], torch.tensor([2.0]))


def test_loss_requires_logprobs():
    with pytest.raises(ValueError, match="logprobs"):
        cross_entropy(ModelOutput(logprobs=None), [Datum(torch.tensor([1]))])
