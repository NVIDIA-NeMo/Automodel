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

# TEMPORARY: intentional failure to validate the "Failed tests" CI log section (PR #2381).
# This file must be removed before the PR is merged.


def test_ci_failed_section_demo_intentional_failure():
    expected = "device_mesh"
    actual = None
    assert actual == expected, "intentional failure to validate the Failed tests CI section (PR #2381 demo)"
