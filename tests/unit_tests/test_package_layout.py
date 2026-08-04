# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Tests that keep the unit-test package layout aligned with the source tree."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_COMPONENTS = _REPO_ROOT / "nemo_automodel" / "components"
_UNIT_TESTS = Path(__file__).resolve().parent
_COMPONENT_UNIT_TESTS = _UNIT_TESTS / "components"


def _directory_names(path: Path) -> set[str]:
    """Return the non-cache child directory names for ``path``."""
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir()
        and child.name != "__pycache__"
        and any(item.name != "__pycache__" for item in child.iterdir())
    }


def _relative_directory_names(path: Path) -> set[str]:
    """Return non-cache descendant directories relative to ``path``."""
    return {
        child.relative_to(path).as_posix()
        for child in path.rglob("*")
        if child.is_dir()
        and child.name != "__pycache__"
        and "__pycache__" not in child.parts
        and any(item.name != "__pycache__" for item in child.iterdir())
    }


def test_component_unit_tests_mirror_source_layout() -> None:
    """Component test suites should use the same first-level layout as their source packages."""
    source_components = _directory_names(_SOURCE_COMPONENTS)
    component_test_suites = _directory_names(_COMPONENT_UNIT_TESTS)
    misplaced_test_suites = _directory_names(_UNIT_TESTS) & source_components

    assert component_test_suites == source_components
    assert not misplaced_test_suites


def test_nested_unit_test_directories_mirror_source_packages() -> None:
    """Nested test directories should correspond to real source package directories."""
    for package in ("_diffusers", "_transformers", "cli", "components", "recipes", "shared"):
        source_directories = _relative_directory_names(_REPO_ROOT / "nemo_automodel" / package)
        test_directories = _relative_directory_names(_UNIT_TESTS / package)

        assert test_directories <= source_directories


def test_recipe_unit_test_directories_cover_recipe_packages() -> None:
    """Every recipe package should have a corresponding unit-test directory."""
    source_directories = _relative_directory_names(_REPO_ROOT / "nemo_automodel" / "recipes")
    test_directories = _relative_directory_names(_UNIT_TESTS / "recipes")

    assert test_directories == source_directories
