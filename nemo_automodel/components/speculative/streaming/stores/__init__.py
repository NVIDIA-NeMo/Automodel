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

"""Pluggable feature-store backends for the streaming data plane.

PR 1 ships one backend -- :class:`LocalFeatureStore` -- as a build-and-test
target. ``SharedDirFeatureStore`` (PR 3) and ``NcclFeatureStore`` (PR 4) plug in
behind the same :class:`~nemo_automodel.components.speculative.streaming.store.FeatureStore`
contract later.
"""

from nemo_automodel.components.speculative.streaming.stores.local import LocalFeatureStore

__all__ = ["LocalFeatureStore"]
