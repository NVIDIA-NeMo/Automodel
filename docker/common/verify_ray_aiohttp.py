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

"""Verify the Ray runtime-env agent uses the project's patched aiohttp."""

import importlib.util
import logging
import runpy
import sys
from importlib.metadata import distribution
from pathlib import Path

from packaging.version import Version


def main() -> None:
    """Import the actual agent without starting it and reject a stale HTTP bundle."""
    if "aiohttp" in sys.modules:
        raise RuntimeError("Run this check in a fresh process without pre-importing aiohttp")
    installed = distribution("aiohttp")
    if Version(installed.version) < Version("3.14.3"):
        raise RuntimeError(f"aiohttp>=3.14.3 is required, found {installed.version}")
    ray_spec = importlib.util.find_spec("ray")
    if ray_spec is None or ray_spec.origin is None:
        raise RuntimeError("Ray must be installed to verify its runtime-env agent")
    agent = Path(ray_spec.origin).parent / "_private/runtime_env/agent/main.py"
    namespace = runpy.run_path(str(agent), run_name="automodel_ray_agent_probe")
    aiohttp = namespace["aiohttp"]
    expected = Path(installed.locate_file("aiohttp/__init__.py")).resolve()
    if aiohttp.__version__ != installed.version or Path(aiohttp.__file__).resolve() != expected:
        raise RuntimeError(f"Ray imports a shadowed aiohttp: {aiohttp.__version__} at {aiohttp.__file__}")
    namespace["web"].Application()
    logging.getLogger(__name__).info("Ray uses aiohttp %s from %s", aiohttp.__version__, expected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
