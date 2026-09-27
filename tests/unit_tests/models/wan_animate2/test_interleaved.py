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

"""Model type validation for Wan-Animate-2 training setup."""

import pytest
import torch

from nemo_automodel.components.models.wan_animate2.interleaved import install_forward_origin


def test_unsupported_model_reports_required_diffusers_class():
    with pytest.raises(TypeError, match="diffusers>=0.40.0"):
        install_forward_origin(torch.nn.Linear(2, 2))
