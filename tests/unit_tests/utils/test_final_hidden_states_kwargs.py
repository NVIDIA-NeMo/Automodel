# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn

from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import final_hidden_states_kwargs


class _HFStyle(nn.Module):
    def forward(self, input_ids, output_hidden_states: bool = False):
        return {"hidden_states": (torch.zeros(1), torch.ones(1)) if output_hidden_states else None}


class _V41Style(nn.Module):
    """``output_hidden_states`` captures residual streams; the loss wants ``return_hidden_states``."""

    def forward(self, input_ids, return_hidden_states: bool = False, output_hidden_states: bool = False):
        if return_hidden_states:
            return {"hidden_states": torch.full((1, 4, 16), 2.0)}
        if output_hidden_states:
            return {"hidden_states": (torch.zeros(1, 4, 4, 16),)}
        return {"hidden_states": None}


def test_hf_style_models_keep_output_hidden_states() -> None:
    model = _HFStyle()
    kwargs = final_hidden_states_kwargs(model)
    assert kwargs == {"output_hidden_states": True}
    torch.testing.assert_close(get_final_hidden_states(model(None, **kwargs)), torch.ones(1))


def test_models_with_return_hidden_states_get_the_final_tensor() -> None:
    model = _V41Style()
    kwargs = final_hidden_states_kwargs(model)
    assert kwargs == {"return_hidden_states": True}
    final = get_final_hidden_states(model(None, **kwargs))
    assert final.shape == (1, 4, 16)
    # the HF kwarg would have handed the loss a [batch, sequence, streams, hidden] residual stream
    assert get_final_hidden_states(model(None, output_hidden_states=True)).shape == (1, 4, 4, 16)


def test_wrapped_models_are_inspected_through_the_wrapper() -> None:
    wrapped = nn.parallel.DistributedDataParallel.__new__(nn.parallel.DistributedDataParallel)
    nn.Module.__init__(wrapped)
    wrapped.module = _V41Style()
    assert final_hidden_states_kwargs(wrapped) == {"return_hidden_states": True}
