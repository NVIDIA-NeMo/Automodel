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

"""Regression tests for explicit NumPy pickle opt-out enforcement."""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.lint_numpy_pickle import lint_source, main


@pytest.mark.parametrize(
    "statement,call",
    [
        ("import numpy", "numpy.load(path"),
        ("import numpy", "numpy.save(path, array"),
        ("import numpy as np", "np.load(path"),
        ("import numpy as numeric", "numeric.save(path, array"),
        ("from numpy import load", "load(path"),
        ("from numpy import save as write_array", "write_array(path, array"),
    ],
)
@pytest.mark.parametrize("argument", ["", ", allow_pickle=True", ", allow_pickle=0", ", allow_pickle=flag", ", **options"])
def test_rejects_missing_or_nonliteral_false(statement, call, argument):
    errors = lint_source(f"{statement}\n{call}{argument})\n", Path("sample.py"))
    assert len(errors) == 1
    assert errors[0].startswith("sample.py:2:1:")
    assert "requires explicit allow_pickle=False" in errors[0]


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nnp.load(path, allow_pickle=False)",
        "import numpy\nnumpy.save(path, array, allow_pickle=False)",
        "from numpy import load as read\nread(path, allow_pickle=False, mmap_mode='r')",
        "from numpy import save\nsave(path, array, allow_pickle=False)",
        "import numpy as np\nnp.load(\n path,\n allow_pickle=False,\n)",
        "import torch\ntorch.load(path)",
        "def load(path): pass\nload(path)",
        "# numpy.load(path)\ntext = 'np.save(path, array)'",
    ],
)
def test_accepts_explicit_false_and_unrelated_calls(source):
    assert lint_source(source, Path("sample.py")) == []


@pytest.mark.parametrize("call", ["np.load(path, None, False)", "np.save(path, array, False)"])
def test_requires_keyword_even_for_positional_false(call):
    assert lint_source("import numpy as np\n" + call, Path("sample.py"))


def test_checks_function_local_imports():
    errors = lint_source("def read():\n    import numpy as n\n    return n.load(path)", Path("sample.py"))
    assert len(errors) == 1
    assert "numpy.load requires" in errors[0]


def test_rejects_star_import():
    assert "use explicit NumPy imports" in lint_source("from numpy import *", Path("sample.py"))[0]


def test_syntax_errors_fail_closed():
    assert "cannot parse file" in lint_source("np.load(", Path("sample.py"))[0]


def test_cli_checks_directories_and_reports_locations(tmp_path, capsys):
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "sample.py"
    source.write_text("import numpy as np\nnp.save(path, array)\n")
    assert main([str(tmp_path)]) == 1
    assert f"{source}:2:1:" in capsys.readouterr().err
    source.write_text("import numpy as np\nnp.save(path, array, allow_pickle=False)\n")
    assert main([str(tmp_path)]) == 0


def test_cli_rejects_missing_or_empty_paths(tmp_path):
    assert main([str(tmp_path / "missing.py")]) == 2
    assert main([str(tmp_path)]) == 2


def test_script_exit_status(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text("import numpy as np\nnp.load(path, allow_pickle=True)\n")
    script = Path(__file__).resolve().parents[3] / "tools" / "lint_numpy_pickle.py"
    result = subprocess.run([sys.executable, str(script), str(source)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "numpy.load requires explicit allow_pickle=False" in result.stderr
