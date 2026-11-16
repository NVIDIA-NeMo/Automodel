# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import tomllib
from pathlib import Path

_PYPROJECT = Path("pyproject.toml")
_TE_PACKAGE = "transformer-engine"


def _strip_te(text: str) -> str:
    """Apply NeMo-CI's STRIP_TE step: ``sed -i '/transformer-engine/{/^name = /!d;}' pyproject.toml``."""
    lines = text.splitlines(keepends=True)
    return "".join(line for line in lines if _TE_PACKAGE not in line or line.startswith("name = "))


def test_te_strip_removes_only_te_build_variables():
    """NeMo-CI's Rubin builds strip TE line by line; other packages' build variables must survive intact."""
    text = _PYPROJECT.read_text()
    build_variables = tomllib.loads(text)["tool"]["uv"]["extra-build-variables"]

    stripped_variables = tomllib.loads(_strip_te(text))["tool"]["uv"]["extra-build-variables"]

    assert stripped_variables == {name: env for name, env in build_variables.items() if _TE_PACKAGE not in name}
