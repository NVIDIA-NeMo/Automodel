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

import pytest
import torch

from nemo_automodel.components.loss.token_normalization import normalize_token_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("count", [0, 1_000_003])
def test_device_token_normalization_preserves_loss_and_gradient(dtype, count):
    loss = torch.tensor(17.0, dtype=dtype, requires_grad=True)
    expected = normalize_token_loss(loss, count)
    actual = normalize_token_loss(loss, torch.tensor(count))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (expected_grad,) = torch.autograd.grad(expected, loss)
    (actual_grad,) = torch.autograd.grad(actual, loss)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
