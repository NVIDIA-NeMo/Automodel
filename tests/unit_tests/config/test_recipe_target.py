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
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from nemo_automodel.cli.app import resolve_recipe_name

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _PROJECT_ROOT / "examples"

_EXCLUDED_PREFIXES = (
    str(_EXAMPLES_DIR / "diffusion"),
    str(_EXAMPLES_DIR / "biencoder" / "mining_config.yaml"),
)


def _collect_example_yamls() -> list[Path]:
    """Return every example YAML that should declare a recipe target."""
    paths: list[Path] = []
    for p in sorted(_EXAMPLES_DIR.rglob("*.yaml")):
        s = str(p)
        if any(s.startswith(exc) for exc in _EXCLUDED_PREFIXES):
            continue
        with open(p) as fh:
            cfg = yaml.safe_load(fh)
        if cfg and isinstance(cfg, dict):
            paths.append(p)
    return paths


def _get_recipe_target(cfg: dict) -> str | None:
    """Extract the recipe target string from a parsed YAML config."""
    recipe = cfg.get("recipe")
    if isinstance(recipe, str) and recipe.strip():
        return recipe.strip()
    if isinstance(recipe, dict):
        target = recipe.get("_target_")
        if isinstance(target, str) and target.strip():
            return target.strip()
    return None


def _fqn_to_source_file(fqn: str) -> tuple[Path, str]:
    """Convert ``pkg.mod.Class`` to ``(pkg/mod.py, 'Class')``."""
    parts = fqn.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected 'module.ClassName', got: {fqn!r}")
    module_dotted, class_name = parts
    module_rel = module_dotted.replace(".", "/") + ".py"
    return _PROJECT_ROOT / module_rel, class_name


@pytest.mark.parametrize(
    "yaml_path",
    _collect_example_yamls(),
    ids=lambda p: str(p.relative_to(_PROJECT_ROOT)),
)
def test_example_config_has_recipe_target(yaml_path: Path):
    """Every example YAML must specify a recipe target pointing to an existing class."""
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    raw = _get_recipe_target(cfg)
    assert raw is not None, (
        "Missing recipe target. Add one of:\n"
        "  recipe: TrainFinetuneRecipeForNextTokenPrediction\n"
        "  recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction\n"
        "  recipe:\n"
        "    _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"
    )

    fqn = resolve_recipe_name(raw)

    source_file, class_name = _fqn_to_source_file(fqn)
    assert source_file.is_file(), f"Module not found: {source_file.relative_to(_PROJECT_ROOT)} (from target '{fqn}')"

    source = source_file.read_text()
    assert re.search(rf"^class\s+{re.escape(class_name)}\b", source, re.MULTILINE), (
        f"Class '{class_name}' not found in {source_file.relative_to(_PROJECT_ROOT)}"
    )
