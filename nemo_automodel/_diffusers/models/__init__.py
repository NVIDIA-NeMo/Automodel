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
"""Models trained through the flow-matching / diffusion bridge.

Each sub-package provides a ``ModelAdapter`` in ``<arch>/adapter.py``. The
architectures themselves come from upstream ``diffusers`` pipelines, which
:class:`NeMoAutoDiffusionPipeline` constructs -- so most adapters here never
import ``diffusers`` directly.

Models trained through the ``transformers`` bridge live in
:mod:`nemo_automodel._transformers.models`.

Kept import-light on purpose: importing this package must not pull in
``diffusers`` or ``torch``. Sub-packages import them lazily.
"""

import importlib.abc
import pathlib
import sys

from nemo_automodel._model_locations import TRANSFORMERS_MODELS_PACKAGE

__all__: list[str] = []

_MODELS_DIR = pathlib.Path(__file__).parent
_PACKAGE_PREFIX = __name__ + "."


def _available_model_submodules() -> set[str]:
    """Return the set of model sub-package names shipped with this installation."""
    return {
        p.name
        for p in _MODELS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "__init__.py").exists()
    }


def _make_upgrade_message(name: str) -> str:
    return (
        f"Module '{__name__}' has no submodule '{name}'. "
        f"Available diffusion model submodules in this installation: "
        f"{sorted(_available_model_submodules())}. "
        f"Models trained through the transformers bridge live in '{TRANSFORMERS_MODELS_PACKAGE}' -- "
        f"try '{TRANSFORMERS_MODELS_PACKAGE}.{name}'. "
        f"If '{name}' is a newly added model, your installed version of "
        f"nemo_automodel may be too old.  Upgrade with:\n"
        f"  pip install --upgrade nemo_automodel\n"
        f"or install from source:\n"
        f"  pip install git+https://github.com/NVIDIA-NeMo/Automodel.git"
    )


def __getattr__(name: str):
    if name in _available_model_submodules():
        # A real submodule that just has not been imported yet. Raise
        # AttributeError so ``hasattr`` in the import machinery falls through to
        # importing ``<pkg>.<name>``; raising ModuleNotFoundError here would
        # break ``from <pkg> import <model>``.
        raise AttributeError(name)
    raise ModuleNotFoundError(_make_upgrade_message(name))


class _MissingModelFinder(importlib.abc.MetaPathFinder):
    """Produces a helpful error when importing a non-existent model subpackage.

    Installed at the *end* of ``sys.meta_path`` so it is only consulted after
    all real finders have already returned ``None``.  For any import of the form
    ``nemo_automodel._diffusers.models.<name>`` (direct child only), it raises
    a ``ModuleNotFoundError`` pointing at the transformers-side package instead
    of the default unhelpful message.
    """

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_PACKAGE_PREFIX):
            return None
        child = fullname[len(_PACKAGE_PREFIX) :]
        if "." in child:
            return None
        raise ModuleNotFoundError(_make_upgrade_message(child))


sys.meta_path.append(_MissingModelFinder())
