#!/usr/bin/env python3
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
import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

_RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _discover_recipe_classes() -> dict[str, str]:
    """Scan ``nemo_automodel/recipes/`` for concrete recipe classes.

    Returns a mapping from bare class name to fully-qualified dotted path,
    e.g. ``{"TrainFinetuneRecipeForNextTokenPrediction":
    "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"}``.
    """
    registry: dict[str, str] = {}
    pkg_root = _RECIPES_DIR.parent.parent
    for py_file in _RECIPES_DIR.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        module_dotted = ".".join(py_file.relative_to(pkg_root).with_suffix("").parts)
        source = py_file.read_text()
        for m in re.finditer(r"^class\s+(\w*Recipe\w*)\s*[\(:]", source, re.MULTILINE):
            cls_name = m.group(1)
            if cls_name == "BaseRecipe":
                continue
            registry[cls_name] = f"{module_dotted}.{cls_name}"
    return registry


def resolve_recipe_name(raw: str) -> str:
    """Resolve a recipe name to its fully-qualified dotted path.

    Accepts:
      - Bare class name: ``"TrainFinetuneRecipeForNextTokenPrediction"``
      - Full FQN: ``"nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"``

    Raises ``ValueError`` when a bare name cannot be found.
    """
    if "." in raw:
        return raw
    registry = _discover_recipe_classes()
    if raw in registry:
        return registry[raw]
    available = "\n".join(f"  - {name}" for name in sorted(registry))
    raise ValueError(f"Unknown recipe class '{raw}'. Available short names:\n{available}")


def load_yaml(file_path):
    """Load and return a YAML file as a dict."""
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        logger.error("File '%s' was not found.", file_path)
        raise e
    except yaml.YAMLError as e:
        logger.error("parsing YAML file '%s' failed: %s", file_path, e)
        raise e
