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

"""Unit tests for the P-EAGLE (parallel_drafting) unsupported-combination gates.

P-EAGLE's ``PEagleTrainerModule.forward`` consumes the live colocated target's
full-vocab ``target_logits`` only, so the remote / offline-cache backends
(precomputed draft-vocab supervision) and sequence packing (per-document
metadata kwargs) must be rejected up front instead of crashing mid-setup.
"""

import pytest

from nemo_automodel.recipes.llm.train_eagle3 import _validate_peagle_gates


def test_peagle_gate_allows_colocated_nonpacked():
    # The only supported combination must not raise.
    _validate_peagle_gates(backend="colocated", cached_target_path=None, packed_sequence_size=0)


def test_peagle_gate_rejects_remote_backend():
    with pytest.raises(NotImplementedError, match="remote backend supplies precomputed"):
        _validate_peagle_gates(backend="remote", cached_target_path=None, packed_sequence_size=0)


def test_peagle_gate_rejects_cached_target():
    with pytest.raises(NotImplementedError, match="offline cached target"):
        _validate_peagle_gates(backend="colocated", cached_target_path="/some/cache", packed_sequence_size=0)


def test_peagle_gate_rejects_sequence_packing():
    with pytest.raises(NotImplementedError, match="sequence packing"):
        _validate_peagle_gates(backend="colocated", cached_target_path=None, packed_sequence_size=4096)
