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

import pytest

from tools.component_imports import find_component_imports

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
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .implementation import PublicName\n"
        '_LAZY_ATTRS = {"PublicName": (".implementation", "PublicName")}\n'
        "__all__ = sorted(_LAZY_ATTRS.keys())\n",
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


def test_component_imports_require_types_for_lazy_exports(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/beta/__init__.py",
        '_LAZY_ATTRS = {"PublicName": (".implementation", "PublicName")}\n__all__ = sorted(_LAZY_ATTRS.keys())\n',
    )

    with pytest.raises(ValueError, match=r"TYPE_CHECKING exports.*missing: PublicName"):
        find_component_imports(tmp_path, COMPONENTS)


def test_component_exports_follow_reset_order(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/beta/__init__.py",
        '__all__ = ["Old"]\n__all__.append("AlsoOld")\n__all__ = ["PublicName"]\n',
    )
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from sample.components.beta import AlsoOld, Old, PublicName\n",
    )

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.violation == "AlsoOld, Old not exported by sample.components.beta.__all__"


def test_component_exports_keep_augmented_values_through_later_expressions(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/beta/__init__.py",
        'EXTRA = ["Second"]\n__all__ = ["First"]\n__all__ += EXTRA\n__all__ = sorted(__all__)\n',
    )
    _write(
        tmp_path / "sample/components/alpha/consumer.py",
        "from sample.components.beta import First, Second\n",
    )

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.is_public


def test_component_exports_ignore_function_local_mutations(tmp_path):
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(
        tmp_path / "sample/components/beta/__init__.py",
        '__all__ = ["PublicName"]\ndef mutate():\n    __all__.append("Nested")\n',
    )
    _write(tmp_path / "sample/components/alpha/consumer.py", "from sample.components.beta import Nested\n")

    [component_import] = find_component_imports(tmp_path, COMPONENTS)

    assert component_import.violation == "Nested not exported by sample.components.beta.__all__"


def test_component_exports_reject_conditional_mutations(tmp_path):
    init_path = tmp_path / "sample/components/beta/__init__.py"
    _write(tmp_path / "sample/components/alpha/__init__.py", "")
    _write(init_path, '__all__ = []\nif enabled:\n    __all__.append("Conditional")\n')

    with pytest.raises(ValueError, match=rf"{init_path}:2: unsupported __all__ conditional or nested mutation"):
        find_component_imports(tmp_path, COMPONENTS)
