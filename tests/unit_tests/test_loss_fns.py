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
from nemo_automodel.loss_fns import BUILTIN_LOSSES, cross_entropy, importance_sampling, ppo


def test_registry_keys():
    assert set(BUILTIN_LOSSES) == {"cross_entropy", "importance_sampling", "ppo"}


def test_cross_entropy_is_negative_logprob():
    mo = ModelOutput(logprobs=[torch.tensor([-0.5, -1.0]), torch.tensor([-2.0])])
    datums = [Datum(torch.tensor([1, 2])), Datum(torch.tensor([3]))]
    out = cross_entropy(mo, datums)
    assert torch.equal(out[0], torch.tensor([0.5, 1.0]))
    assert torch.equal(out[1], torch.tensor([2.0]))


def test_importance_sampling_ratio_times_advantage():
    lp = torch.tensor([-0.5, -1.0])
    old = torch.tensor([-0.5, -0.5])  # ratio = exp(0), exp(-0.5)
    adv = torch.tensor([2.0, 3.0])
    mo = ModelOutput(logprobs=[lp])
    datums = [Datum(torch.tensor([1, 2]), loss_inputs={"logprobs": old, "advantages": adv})]
    out = importance_sampling(mo, datums)[0]
    expected = -(torch.exp(lp - old) * adv)
    assert torch.allclose(out, expected)


def test_ppo_clips_ratio():
    # ratio = exp(lp - old); with a large positive gap the unclipped ratio is huge,
    # so for positive advantage the clipped (1+eps)*adv is the min -> clipping active.
    lp = torch.tensor([2.0])
    old = torch.tensor([0.0])  # ratio = e^2 ≈ 7.39
    adv = torch.tensor([1.0])
    mo = ModelOutput(logprobs=[lp])
    datums = [Datum(torch.tensor([1]), loss_inputs={"logprobs": old, "advantages": adv})]
    out = ppo(mo, datums, clip_eps=0.2)[0]
    # min(7.39*1, 1.2*1) = 1.2 -> loss = -1.2
    assert float(out) == pytest.approx(-1.2, rel=1e-4)


def test_loss_requires_logprobs():
    with pytest.raises(ValueError, match="logprobs"):
        cross_entropy(ModelOutput(logprobs=None), [Datum(torch.tensor([1]))])
