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

"""Tensor-free control-plane contracts for the speculative-training stream.

The streaming pipeline (per the EAGLE-3 / DFlash / DSpark train-inference
disaggregation RFC, issue #3062 PR 1) splits every transferred sample into a
:mod:`refs` (control-plane, no tensors) and a :mod:`store` (data-plane, holds
the supervision tensors). The queue carries refs only; tensors live in the
store and are referenced by :class:`nemo_automodel.components.speculative.streaming.refs.SampleRef.feature_keys`.

This package owns:

- :class:`~nemo_automodel.components.speculative.streaming.refs.SampleRef` and
  :class:`~nemo_automodel.components.speculative.streaming.refs.FeatureSpec`
  -- the frozen, tensor-free reference carried on every control-plane hop.
- :func:`~nemo_automodel.components.speculative.streaming.refs.assert_no_tensors`
  -- the guard that enforces the no-tensor invariant on the control plane.
- :class:`~nemo_automodel.components.speculative.streaming.store.FeatureStore`
  -- the pluggable data-plane transport (local dict, shared POSIX mount, NCCL,
  ...); :class:`~nemo_automodel.components.speculative.streaming.stores.local.LocalFeatureStore`
  -- the in-process implementation land-tested by PR 1.
- :class:`~nemo_automodel.components.speculative.streaming.queue.SampleRefQueue`
  -- the metadata-only lease/ack/fail queue between producers and consumers
  with visibility-timeout reclaim and watermark-based backpressure.
"""

from nemo_automodel.components.speculative.streaming.queue import (
    Lease,
    SampleRefQueue,
    VisibilityTimeout,
)
from nemo_automodel.components.speculative.streaming.refs import (
    FeatureAlgorithm,
    FeatureSpec,
    SampleRef,
    assert_no_tensors,
)
from nemo_automodel.components.speculative.streaming.store import (
    FeatureStore,
    StoreHandle,
    StoreHealth,
)
from nemo_automodel.components.speculative.streaming.stores.local import LocalFeatureStore

__all__ = [
    "FeatureAlgorithm",
    "FeatureSpec",
    "FeatureStore",
    "Lease",
    "LocalFeatureStore",
    "SampleRef",
    "SampleRefQueue",
    "StoreHandle",
    "StoreHealth",
    "VisibilityTimeout",
    "assert_no_tensors",
]
