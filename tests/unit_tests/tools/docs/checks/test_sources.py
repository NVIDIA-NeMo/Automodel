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


# Over the default 5s budget on purpose: this module drives git through subprocesses over throwaway repositories.
# Shrink the work or the process count before raising this further.

from pathlib import Path

import pytest

from tools.docs.checks.sources import (
    _validate_generated_coverage,
    _validate_generated_tables,
    _validate_overview_tables,
)

pytestmark = pytest.mark.timeout(70)


def test_generated_model_coverage_tables_are_not_committed():
    _validate_generated_tables(Path(__file__).resolve().parents[5])


def test_model_type_overviews_use_one_dated_supported_models_table():
    _validate_overview_tables(Path(__file__).resolve().parents[5])


def test_typed_support_tables_include_every_documented_model_family():
    _validate_generated_coverage(Path(__file__).resolve().parents[5])
