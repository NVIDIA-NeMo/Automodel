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

from pathlib import Path
from unittest.mock import Mock

from tools.component_imports import ComponentImport, find_component_imports
from tools.import_linter_contracts import ComponentInterfaceContract

COMPONENTS = {"sample.components.alpha", "sample.components.beta"}


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_component_imports_allow_symbols_exported_from_package(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from sample.components.beta import PublicName\n",
    )
    _write(
        tmp_path / "sample/components/beta/__init__.py",
        '_LAZY_ATTRS = {"PublicName": (".implementation", "PublicName")}\n__all__ = sorted(_LAZY_ATTRS.keys())\n',
    )

    component_imports = find_component_imports(tmp_path, COMPONENTS)

    assert len(component_imports) == 1
    assert component_imports[0].is_public


def test_component_imports_check_consumers_outside_components(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')
    _write(tmp_path / "sample/recipes/train.py", "from sample.components.beta import PublicName\n")

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.importer == "sample.recipes.train"
    assert component_import.is_public


def test_component_imports_reject_private_modules_outside_components(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')
    _write(tmp_path / "sample/components/beta/implementation.py", "PublicName = object()\n")
    _write(
        tmp_path / "sample/recipes/train.py",
        "from sample.components.beta.implementation import PublicName\n",
    )

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.importer == "sample.recipes.train"
    assert component_import.violation == (
        "sample.components.beta.implementation is private; import exported symbols from sample.components.beta"
    )


def test_component_imports_reject_non_exported_symbols_outside_components(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')
    _write(tmp_path / "sample/recipes/train.py", "from sample.components.beta import PrivateName\n")

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.importer == "sample.recipes.train"
    assert component_import.violation == "PrivateName not exported by sample.components.beta.__all__"


def test_component_imports_reject_non_exported_symbols(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from sample.components.beta import PrivateName\n",
    )
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.violation == "PrivateName not exported by sample.components.beta.__all__"


def test_component_imports_reject_private_module_even_when_symbol_is_exported(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from sample.components.beta.implementation import PublicName\n",
    )
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')
    _write(tmp_path / "sample/components/beta/implementation.py", '__all__ = ["PublicName"]\n')

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.violation == (
        "sample.components.beta.implementation is private; import exported symbols from sample.components.beta"
    )


def test_component_imports_reject_component_module_imports(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "import sample.components.beta\nfrom sample.components import beta\n",
    )
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')

    component_imports = find_component_imports(tmp_path, COMPONENTS)

    assert [component_import.line_number for component_import in component_imports] == [1, 2]
    assert all(not component_import.is_public for component_import in component_imports)


def test_component_imports_resolve_relative_public_imports(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from ..beta import PublicName\n",
    )
    _write(tmp_path / "sample/components/beta/__init__.py", '__all__ = ["PublicName"]\n')

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.is_public
    assert component_import.imported_module == "sample.components.beta"


def test_component_imports_ignore_same_component_and_type_checking_imports(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(tmp_path / "sample/components/alpha/local.py", "LocalName = object()\n")
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from typing import TYPE_CHECKING\n"
        "from sample.components.alpha.local import LocalName\n"
        "if TYPE_CHECKING:\n"
        "    from sample.components.beta.implementation import PrivateName\n",
    )
    _write(tmp_path / "sample/components/beta/__init__.py", "")
    _write(tmp_path / "sample/components/beta/implementation.py", "PrivateName = object()\n")

    assert find_component_imports(tmp_path, COMPONENTS) == []


def test_component_contract_skips_importers_absent_from_import_graph():
    graph = Mock()
    graph.modules = set()
    component_import = ComponentImport(
        path=Path("sample/recipes/train.py"),
        line_number=1,
        importer="sample.recipes.train",
        target_component="sample.components.beta",
        imported_module="sample.components.beta",
        imported_names=("PublicName",),
        violation=None,
    )

    ComponentInterfaceContract._remove_direct_import(graph, component_import)

    graph.find_modules_directly_imported_by.assert_not_called()
