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

"""Import-hygiene guards: the package and the parallelizer must stay cheap to import."""

import json
import subprocess
import sys

import pytest

_PROBE = """
import json, sys
import {module}
prefixes = {prefixes!r}
print(json.dumps({{p: sorted(m for m in sys.modules if m == p or m.startswith(p + ".")) for p in prefixes}}))
"""


def _modules_loaded_by(module: str, *prefixes: str) -> dict[str, list[str]]:
    """Import module in a fresh interpreter and report which prefixes ended up in sys.modules."""
    code = _PROBE.format(module=module, prefixes=list(prefixes))
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(120)
def test_package_import_does_not_load_torch():
    """import nemo_automodel installs the transformers import hook (stdlib only) and nothing heavier."""
    assert _modules_loaded_by("nemo_automodel", "torch") == {"torch": []}


@pytest.mark.timeout(120)
def test_parallelizer_import_stays_off_the_model_zoo_and_compiler():
    """components.distributed reads contracts off the model class and defers pipelining and
    selective-AC machinery, so importing the parallelizer must load none of these (see
    components/distributed/AGENTS.md)."""
    loaded = _modules_loaded_by(
        "nemo_automodel.components.distributed.parallelizer",
        "transformers.models",
        "torch._dynamo",
        "torch.distributed.pipelining",
        "networkx",
    )
    assert all(not mods for mods in loaded.values()), loaded
