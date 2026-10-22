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

from tools.docs.checks.navigation import (
    _validate_internal_links,
    _validate_provider_sections,
    _validate_published_training_routes,
    _validate_recipe_routes,
    _validate_redirects,
    _validate_release_navigation,
)

pytestmark = pytest.mark.timeout(70)


def test_regrouped_training_pages_preserve_published_urls():
    _validate_published_training_routes(Path(__file__).resolve().parents[5])


def test_model_release_docs_pages_exist_in_nightly_navigation():
    _validate_release_navigation(Path(__file__).resolve().parents[5])


def test_model_coverage_pages_use_provider_sections_and_checkpoint_slugs():
    _validate_provider_sections(Path(__file__).resolve().parents[5])


def test_recipe_backed_model_sizes_have_exact_index_routes_and_one_card():
    _validate_recipe_routes(Path(__file__).resolve().parents[5])


def test_retired_model_routes_redirect_to_provider_indexes():
    _validate_redirects(Path(__file__).resolve().parents[5])


def test_internal_links_from_model_coverage_resolve_to_nightly_routes():
    _validate_internal_links(Path(__file__).resolve().parents[5])
