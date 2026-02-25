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

"""
Model adapters for FlowMatching Pipeline.

This module provides model-specific adapters that decouple the flow matching
logic from model-specific implementation details.

Available Adapters:
- ModelAdapter: Abstract base class for all adapters
- HunyuanAdapter: For HunyuanVideo 1.5 style models
- SimpleAdapter: For simple transformer models (e.g., Wan)
- FluxAdapter: For FLUX.1 text-to-image models

Usage:
    from automodel.flow_matching.adapters import HunyuanAdapter, SimpleAdapter, FluxAdapter

    # Or import the base class to create custom adapters
    from automodel.flow_matching.adapters import ModelAdapter
"""

from .base import FlowMatchingContext, ModelAdapter
from .flux import FluxAdapter
from .hunyuan import HunyuanAdapter
from .simple import SimpleAdapter


__all__ = [
    "FlowMatchingContext",
    "ModelAdapter",
    "FluxAdapter",
    "HunyuanAdapter",
    "SimpleAdapter",
]
