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

"""Compatibility imports for shared training and data-construction RNG utilities.

Keep this module path available for existing callers and legacy RNGState pickles.
"""

from nemo_automodel.shared.rng import RNGState, ScopedRNG, StatefulRNG, init_all_rng

__all__ = ["RNGState", "ScopedRNG", "StatefulRNG", "init_all_rng"]
