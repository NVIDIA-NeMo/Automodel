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

"""Pluggable feature-store backends for the streaming data plane.

:class:`LocalFeatureStore` is the in-process reference backend.
:class:`SharedDirFeatureStore` adds cross-process rendezvous on a shared
POSIX mount. A future :class:`NcclFeatureStore` can plug in behind the same
:class:`~nemo_automodel.components.speculative.streaming.store.FeatureStore`
contract.
"""

from nemo_automodel.components.speculative.streaming.stores.local import LocalFeatureStore
from nemo_automodel.components.speculative.streaming.stores.shared_dir import SharedDirFeatureStore

__all__ = ["LocalFeatureStore", "SharedDirFeatureStore"]
