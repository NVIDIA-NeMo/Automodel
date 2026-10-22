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

"""Regression coverage for architecture documentation validators."""

from pathlib import Path

from tools.docs.architecture import (
    validate_architecture_aliases,
    validate_architecture_reference_tables,
    validate_registered_architectures,
)


def test_every_model_card_has_architecture_reference_table():
    validate_architecture_reference_tables(Path(__file__).resolve().parents[3])


def test_every_registered_arch_has_model_coverage_doc():
    validate_registered_architectures(Path(__file__).resolve().parents[3])


def test_doc_arch_aliases_target_strings_appear_in_docs():
    validate_architecture_aliases(Path(__file__).resolve().parents[3])
