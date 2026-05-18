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
from __future__ import annotations

from pathlib import Path

from tools.lint_example_yamls import lint_yaml_text


def test_linter_accepts_valid_example_recipe():
    errors = lint_yaml_text(
        "\n".join([
            "recipe: TrainFinetuneRecipeForNextTokenPrediction",
            "model:",
            "  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "ci:",
            "  recipe_owner: test",
            "",
        ]),
        Path("examples/llm_finetune/valid.yaml"),
        Path.cwd(),
    )

    assert errors == []


def test_linter_requires_recipe_target():
    errors = lint_yaml_text("model: {}\n", Path("examples/llm_finetune/missing_recipe.yaml"), Path.cwd())

    assert len(errors) == 1
    assert "Missing recipe target" in errors[0].message


def test_linter_requires_recipe_first():
    errors = lint_yaml_text(
        "step_scheduler: {}\nrecipe: TrainFinetuneRecipeForNextTokenPrediction\n",
        Path("examples/llm_finetune/recipe_order.yaml"),
        Path.cwd(),
    )

    assert any("`recipe` section must be the first" in error.message for error in errors)


def test_linter_requires_ci_last():
    errors = lint_yaml_text(
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nci: {}\nmodel: {}\n",
        Path("examples/llm_finetune/ci_order.yaml"),
        Path.cwd(),
    )

    assert any("`ci` section must be the last" in error.message for error in errors)


def test_linter_rejects_duplicate_top_level_keys():
    errors = lint_yaml_text(
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nmodel: {}\nmodel: {}\n",
        Path("examples/llm_finetune/duplicate.yaml"),
        Path.cwd(),
    )

    assert any("Duplicate top-level key `model`" in error.message for error in errors)
