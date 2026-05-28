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

"""Dispatcher mapping capability names to their standardized tests.

To add a new capability test, drop a module under ``standardized_tests/`` that
exposes a :class:`CapabilityTest`-compatible class, and register it here.
"""

from __future__ import annotations

from tests.capability_registry.standardized_tests._base import CapabilityTest
from tests.capability_registry.standardized_tests.test_cp import CPTest
from tests.capability_registry.standardized_tests.test_ep import EPTest
from tests.capability_registry.standardized_tests.test_pp import PPTest
from tests.capability_registry.standardized_tests.test_tp import TPTest


CAPABILITY_TESTS: dict[str, CapabilityTest] = {
    "tp": TPTest(),
    "cp": CPTest(),
    "pp": PPTest(),
    "ep": EPTest(),
}
