# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA regression for MoE gate accounting under activation checkpointing."""

import pytest
import torch

from tests.unit_tests.moe.test_gate_activation_checkpointing import _assert_gate_load_and_gradient_parity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_gate_load_is_accumulated_once_during_activation_checkpointing() -> None:
    _assert_gate_load_and_gradient_parity(torch.device("cuda:0"), {})
