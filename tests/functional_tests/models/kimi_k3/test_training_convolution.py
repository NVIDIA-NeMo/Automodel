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

"""CUDA regression for singleton KDA training and noncontiguous cotangents."""

import pytest
import torch


@pytest.mark.parametrize("channels", [4, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires the FLA CUDA training kernel")
def test_singleton_kda_convolution_matches_torch_forward_and_backward(channels: int) -> None:
    from fla.modules import ShortConvolution

    from nemo_automodel.components.models.kimi_k3.model import KimiTrainingShortConvolution

    torch.manual_seed(901)
    convolution = KimiTrainingShortConvolution(channels, 4).cuda()
    broken = ShortConvolution(channels, 4, activation="silu", dtype=torch.float32).cuda()
    broken.load_state_dict(convolution.state_dict())
    value = torch.randn(1, 1, channels, device="cuda", requires_grad=True)
    boundaries = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    actual, _ = convolution(value, cu_seqlens=boundaries)
    expected = torch.nn.functional.silu(
        torch.nn.functional.conv1d(value.transpose(1, 2), convolution.weight, padding=3, groups=channels)[
            :, :, :1
        ].transpose(1, 2)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual_grads = torch.autograd.grad(actual.sum(), (value, convolution.weight))
    expected_grads = torch.autograd.grad(expected.sum(), (value, convolution.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        defective, _ = broken(value.detach(), cu_seqlens=boundaries)
    assert not torch.allclose(defective, expected), "The old inference shortcut must fail this fixture"
