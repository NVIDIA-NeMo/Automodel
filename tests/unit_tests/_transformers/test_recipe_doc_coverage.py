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

"""Regression tests for recipe documentation tools."""

import os
from pathlib import Path

import pytest

from tools.docs.recipe_coverage import (
    _collect_recipe_model_ids,
    validate_recipe_architectures,
    validate_recipe_publisher_aliases,
    validate_recipe_publishers,
)

pytestmark = pytest.mark.timeout(70)


def test_recipe_archs_have_doc_coverage():
    if os.environ.get("NEMO_DOC_COVERAGE_RESOLVE_ARCHS", "0") != "1":
        pytest.skip("Set NEMO_DOC_COVERAGE_RESOLVE_ARCHS=1 for optional HF config resolution")
    validate_recipe_architectures(Path(__file__).resolve().parents[3])


def test_yaml_recipe_scan_finds_model_ids():
    assert len(_collect_recipe_model_ids(Path(__file__).resolve().parents[3] / "examples")) >= 10


def test_recipe_model_ids_live_under_publishing_org_dir():
    validate_recipe_publishers(Path(__file__).resolve().parents[3])


def test_hf_org_to_doc_slug_targets_exist():
    validate_recipe_publisher_aliases(Path(__file__).resolve().parents[3])
