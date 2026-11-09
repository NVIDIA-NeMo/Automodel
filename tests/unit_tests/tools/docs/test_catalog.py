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


"""Regression coverage for literal catalog metadata."""

from pathlib import Path

import pytest

from tools.docs.build.coverage_tables import _parse_doc_arch_aliases
from tools.docs.catalog import _DOC_ARCH_ALIASES


def test_doc_arch_aliases_are_parsed_from_the_docs_tool():
    source = '_DOC_ARCH_ALIASES = {"NativeArchitecture": "DocumentedArchitecture"}\n'

    assert _parse_doc_arch_aliases(source) == {"NativeArchitecture": "DocumentedArchitecture"}


def test_catalog_architecture_aliases_can_be_read_as_literal_data():
    catalog = Path(__file__).resolve().parents[4] / "tools/docs/catalog.py"
    assert _parse_doc_arch_aliases(catalog.read_text()) == _DOC_ARCH_ALIASES


def test_catalog_parser_rejects_computed_aliases_without_executing_them(tmp_path):
    marker = tmp_path / "executed"
    source = f"_DOC_ARCH_ALIASES = __import__('pathlib').Path({str(marker)!r}).touch()"

    with pytest.raises(ValueError):
        _parse_doc_arch_aliases(source)
    assert not marker.exists()
