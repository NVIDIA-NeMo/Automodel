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

import pytest

from tools.lint_example_yamls import lint_yaml_text


def test_linter_accepts_valid_example_recipe():
    errors = lint_yaml_text(
        "\n".join(
            [
                "recipe: TrainFinetuneRecipeForNextTokenPrediction",
                "model:",
                "  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
                "ci:",
                "  cluster_tag: h100",
                "  recipe_owner: test",
                "",
            ]
        ),
        Path("examples/llm_finetune/valid.yaml"),
        Path.cwd(),
    )

    assert errors == []


def test_linter_requires_recipe_target():
    errors = lint_yaml_text(
        "model: {}\nci:\n  cluster_tag: h100\n", Path("examples/llm_finetune/missing_recipe.yaml"), Path.cwd()
    )

    assert len(errors) == 1
    assert "Missing recipe target" in errors[0].message


@pytest.mark.parametrize("cluster_tag", ["h100", "gb200", "gb10", "h100|gb200"])
def test_linter_accepts_explicit_cluster_tag(cluster_tag: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: {cluster_tag}\n",
        Path("examples/llm_finetune/tagged.yaml"),
        Path.cwd(),
    )

    assert errors == []


@pytest.mark.parametrize("text", ["", "# Empty recipe\n"])
def test_linter_rejects_empty_recipe(text: str) -> None:
    errors = lint_yaml_text(text, Path("examples/llm_finetune/empty.yaml"), Path.cwd())

    assert len(errors) == 1
    assert "`ci.cluster_tag`" in errors[0].message


@pytest.mark.parametrize("ci_text", ["", "ci: null\n", "ci: []\n", "ci: h100\n", "ci: {}\n"])
def test_linter_requires_ci_cluster_metadata(ci_text: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\n{ci_text}",
        Path("examples/llm_finetune/untagged.yaml"),
        Path.cwd(),
    )

    assert len(errors) == 1
    assert "`ci.cluster_tag`" in errors[0].message
    assert "`ci.deprecated: true`" in errors[0].message
    assert errors[0].line == (2 if ci_text else 1)


@pytest.mark.parametrize("value", ["null", "''", "'   '", "100", "true", "[]", "{}"])
def test_linter_rejects_empty_or_nonstring_cluster_tag(value: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: {value}\n",
        Path("examples/llm_finetune/invalid_tag.yaml"),
        Path.cwd(),
    )

    assert len(errors) == 1
    assert "`ci.cluster_tag`" in errors[0].message


@pytest.mark.parametrize("deprecated", ["true", "false", "'true'", "1", "null"])
def test_linter_only_exempts_boolean_deprecated_true(deprecated: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  deprecated: {deprecated}\n",
        Path("examples/llm_finetune/deprecated.yaml"),
        Path.cwd(),
    )

    if deprecated == "true":
        assert errors == []
    else:
        assert len(errors) == 1
        assert "`ci.cluster_tag`" in errors[0].message


@pytest.mark.parametrize("folder", ["diffusion/finetune", "diffusion/generate/configs", "convergence/tulu3/models"])
def test_linter_checks_cluster_tag_when_recipe_target_is_exempt(folder: str) -> None:
    path = Path("examples") / folder / "untagged.yaml"
    errors = lint_yaml_text("model: {}\n", path, Path.cwd())

    assert len(errors) == 1
    assert "`ci.cluster_tag`" in errors[0].message
    assert lint_yaml_text("model: {}\nci:\n  cluster_tag: h100\n", path, Path.cwd()) == []


def test_linter_exempts_command_only_precompute_configs():
    """The dspark distributed-precompute configs are launched via `python -m`, not a recipe class."""
    for name in ("deepseek_v4_flash_dspark_precompute.yaml", "glm_5.2_dspark_precompute.yaml"):
        errors = lint_yaml_text(
            "recipe_args: {}\n",
            Path("examples/speculative/dspark") / name,
            Path.cwd(),
        )
        assert errors == []


def test_linter_exempts_command_only_retrieval_mining_configs():
    """Mining configs are consumed by their script and have no recipe-class target."""
    for name in ("mining_config.yaml", "mining_multimodal_config.yaml"):
        errors = lint_yaml_text(
            "mining: {}\n",
            Path("examples/retrieval/data_utils") / name,
            Path.cwd(),
        )
        assert errors == []


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


def test_linter_rejects_wandb_without_enable_false():
    errors = lint_yaml_text(
        "\n".join(
            [
                "recipe: TrainFinetuneRecipeForNextTokenPrediction",
                "model:",
                "  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
                "wandb:",
                "  project: my-project",
                "ci:",
                "  recipe_owner: test",
                "",
            ]
        ),
        Path("examples/llm_finetune/with_wandb.yaml"),
        Path.cwd(),
    )

    assert any("must set `enable: false`" in error.message for error in errors)


def test_linter_accepts_wandb_with_enable_false():
    errors = lint_yaml_text(
        "\n".join(
            [
                "recipe: TrainFinetuneRecipeForNextTokenPrediction",
                "model:",
                "  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
                "wandb:",
                "  enable: false",
                "  project: my-project",
                "ci:",
                "  recipe_owner: test",
                "",
            ]
        ),
        Path("examples/llm_finetune/with_wandb.yaml"),
        Path.cwd(),
    )

    assert not any("wandb" in error.message for error in errors)


def test_linter_rejects_duplicate_top_level_keys():
    errors = lint_yaml_text(
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nmodel: {}\nmodel: {}\n",
        Path("examples/llm_finetune/duplicate.yaml"),
        Path.cwd(),
    )

    assert any("Duplicate top-level key `model`" in error.message for error in errors)
