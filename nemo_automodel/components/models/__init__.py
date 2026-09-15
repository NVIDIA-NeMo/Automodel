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
"""Convenience model builders for NeMo Automodel.

Currently includes:
    • build_gpt2_model – returns a GPT-2 causal language model (Flash-Attention-2 by default).
"""

from __future__ import annotations

import importlib
import importlib.abc
import pathlib
import re
import sys
from typing import TYPE_CHECKING

from .gpt2 import build_gpt2_model  # noqa: F401

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

__all__ = [
    "build_gpt2_model",
    "declared_parallel_spec",
    "model_family",
]

_MODELS_DIR = pathlib.Path(__file__).parent
_PACKAGE_PREFIX = __name__ + "."


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def model_family(model_class: type) -> str | None:
    """Model package under ``components/models`` that may declare contracts for an upstream class.

    transformers classes (stock or ``trust_remote_code``) map through the transformers module name
    of their config's ``model_type``: ``gemma3`` for both ``gemma3`` and ``gemma3_text``,
    ``nemotron_nas`` for a ``nemotron-nas`` checkpoint. diffusers models carry no ``model_type``;
    their class name is the identifier and the package is its snake_case stem before
    ``Transformer``: ``WanTransformer3DModel`` -> ``wan``, ``QwenImageTransformer2DModel`` ->
    ``qwen_image``.

    Returns:
        The package name, or ``None`` for a class that belongs to neither library.
    """
    model_type = getattr(getattr(model_class, "config_class", None), "model_type", None)
    if model_type:
        from transformers.models.auto.configuration_auto import model_type_to_module_name

        return model_type_to_module_name(model_type)
    if getattr(model_class, "config_name", None):  # diffusers ``ModelMixin`` marker
        return _CAMEL_BOUNDARY.sub("_", model_class.__name__.split("Transformer", 1)[0]).lower()
    return None


def declared_parallel_spec(model_class: type) -> ParallelSpec | None:
    """Return the ``ParallelSpec`` a model package declares for an upstream class it does not re-implement.

    Architectures the repository only wraps -- stock ``transformers`` classes, ``trust_remote_code``
    checkpoints, ``diffusers`` transformers -- keep their contract in
    ``components/models/<family>/parallelization.py`` (``<family>`` per :func:`model_family`) on a
    class named after the upstream architecture::

        class Gemma3ForConditionalGeneration:
            parallel_spec = ParallelSpec(tp_plan=gemma3_tp_plan, ...)

    Returns:
        The declared spec, or ``None`` when the package or the declaration does not exist.
    """
    family = model_family(model_class)
    if not family or not family.isidentifier() or not (_MODELS_DIR / family / "parallelization.py").is_file():
        return None
    module = importlib.import_module(f"{_PACKAGE_PREFIX}{family}.parallelization")
    return getattr(getattr(module, model_class.__name__, None), "parallel_spec", None)


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
        f"Available model submodules in this installation: "
        f"{sorted(_available_model_submodules())}. "
        f"If '{name}' is a newly added model, your installed version of "
        f"nemo_automodel may be too old.  Upgrade with:\n"
        f"  pip install --upgrade nemo_automodel\n"
        f"or install from source:\n"
        f"  pip install git+https://github.com/NVIDIA-NeMo/Automodel.git"
    )


def __getattr__(name: str):
    raise ModuleNotFoundError(_make_upgrade_message(name))


class _MissingModelFinder(importlib.abc.MetaPathFinder):
    """Produces a helpful error when importing a non-existent model subpackage.

    Installed at the *end* of ``sys.meta_path`` so it is only consulted after
    all real finders have already returned ``None``.  For any import of the form
    ``nemo_automodel.components.models.<name>`` (direct child only), it raises
    a ``ModuleNotFoundError`` with upgrade instructions instead of the default
    unhelpful message.
    """

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(_PACKAGE_PREFIX):
            return None
        child = fullname[len(_PACKAGE_PREFIX) :]
        if "." in child:
            return None
        raise ModuleNotFoundError(_make_upgrade_message(child))


sys.meta_path.append(_MissingModelFinder())
