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

from nemo_automodel.shared.import_utils import safe_import
HAS_TE, transformer_engine = safe_import("transformer_engine")
from nemo_automodel.components._peft.module_matcher import _is_linear_module
import pytest
import torch.nn as nn

@pytest.mark.parametrize(("module", "expected"), 
    [(nn.Linear(10, 10), True),
    (nn.Conv1d(10, 10, 1), False),
    (nn.Conv2d(10, 10, 1), False),
    (nn.Conv3d(10, 10, 1), False),
    (nn.ConvTranspose1d(10, 10, 1), False),
    (nn.ConvTranspose2d(10, 10, 1), False),
    (nn.ConvTranspose3d(10, 10, 1), False),
])
def test_is_linear_module(module, expected):
    assert _is_linear_module(module) == expected

@pytest.mark.skipif(not HAS_TE, reason="transformer_engine is not installed")
def test_is_linear_module_transformer_engine():
    assert _is_linear_module(transformer_engine.pytorch.Linear(10, 10))