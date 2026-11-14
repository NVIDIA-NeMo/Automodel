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
                "  nemo_ci_schedule: release",
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
        "model: {}\nci:\n  cluster_tag: h100\n  nemo_ci_schedule: skipped\n",
        Path("examples/llm_finetune/missing_recipe.yaml"),
        Path.cwd(),
    )

    assert len(errors) == 1
    assert "Missing recipe target" in errors[0].message


@pytest.mark.parametrize("cluster_tag", ["h100", "gb200", "gb10", "h100|gb200"])
def test_linter_accepts_explicit_cluster_tag(cluster_tag: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: {cluster_tag}\n  nemo_ci_schedule: skipped\n",
        Path("examples/llm_finetune/tagged.yaml"),
        Path.cwd(),
    )

    assert errors == []


@pytest.mark.parametrize("text", ["", "# Empty recipe\n"])
def test_linter_rejects_empty_recipe(text: str) -> None:
    errors = lint_yaml_text(text, Path("examples/llm_finetune/empty.yaml"), Path.cwd())

    assert len(errors) == 2
    assert "`ci.cluster_tag`" in errors[0].message
    assert "`ci.nemo_ci_schedule` is missing" in errors[1].message


@pytest.mark.parametrize("ci_text", ["", "ci: null\n", "ci: []\n", "ci: h100\n", "ci: {}\n"])
def test_linter_requires_ci_cluster_metadata(ci_text: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\n{ci_text}",
        Path("examples/llm_finetune/untagged.yaml"),
        Path.cwd(),
    )

    assert [error.line for error in errors] == [2 if ci_text else 1] * 2
    assert "`ci.cluster_tag`" in errors[0].message
    assert "`ci.deprecated: true`" in errors[0].message
    assert "`ci.nemo_ci_schedule` is missing" in errors[1].message


@pytest.mark.parametrize("value", ["null", "''", "'   '", "100", "true", "[]", "{}"])
def test_linter_rejects_empty_or_nonstring_cluster_tag(value: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: {value}\n  nemo_ci_schedule: skipped\n",
        Path("examples/llm_finetune/invalid_tag.yaml"),
        Path.cwd(),
    )

    assert len(errors) == 1
    assert "`ci.cluster_tag`" in errors[0].message


@pytest.mark.parametrize("deprecated", ["true", "false", "'true'", "1", "null"])
def test_linter_only_exempts_boolean_deprecated_true(deprecated: str) -> None:
    errors = lint_yaml_text(
        f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  deprecated: {deprecated}\n  nemo_ci_schedule: skipped\n",
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

    assert len(errors) == 2
    assert "`ci.cluster_tag`" in errors[0].message
    assert "`ci.nemo_ci_schedule` is missing" in errors[1].message
    assert lint_yaml_text("model: {}\nci:\n  cluster_tag: h100\n  nemo_ci_schedule: skipped\n", path, Path.cwd()) == []


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


def _perf_text(perf_block: str) -> str:
    # `nemo_ci_schedule` follows the block so the asserted perf_testing line numbers stay put.
    return (
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: h100\n  perf_testing:\n"
        + perf_block
        + "  nemo_ci_schedule: skipped\n"
    )


def _perf_errors(block: str, *, default_schedule: bool = True):
    """Lint a ``perf_testing`` block, appending a valid ``schedule`` unless the block sets one."""
    if default_schedule and "    schedule:" not in block and not block.lstrip().startswith("-"):
        block += "    schedule: nightly\n"
    return lint_yaml_text(_perf_text(block), Path("examples/llm_finetune/perf.yaml"), Path.cwd())


def test_linter_accepts_valid_perf_testing():
    errors = _perf_errors('    time: "00:40:00"\n    schedule: weekly\n    fake_balanced_gate: false\n')

    assert errors == []


def test_linter_accepts_minimal_perf_testing():
    assert _perf_errors("    schedule: disabled\n") == []


def test_linter_requires_perf_testing_schedule():
    errors = _perf_errors('    time: "00:40:00"\n', default_schedule=False)

    assert len(errors) == 1
    assert "`schedule` is required" in errors[0].message


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("    schedule: true\n", "`schedule` must be one of"),
        ("    schedule: 1\n", "`schedule` must be one of"),
        ("    schedule: monthly\n", "`schedule` must be one of"),
        ("    schedule: off\n", "`schedule` must be one of"),  # YAML 1.1 parses unquoted off as False
        ("    time: 2400\n", "`time` must be"),
        ('    time: "40 minutes"\n', "`time` must be"),
        ('    time: "25:00"\n', "`time` must be"),
        ('    time: "24:00:00"\n', "`time` must be"),
        ("    config_overrides: {}\n", "unknown key `config_overrides`"),
        ("    fake_balanced_gate: 1\n", "`fake_balanced_gate` must be a boolean"),
        ("    fake_balanced_gate: 'true'\n", "`fake_balanced_gate` must be a boolean"),
        ("    fake_balanced_gate: null\n", "`fake_balanced_gate` must be a boolean"),
        ("    fake_balanced_gate: []\n", "`fake_balanced_gate` must be a boolean"),
        ("    fake_balanced_gate: {}\n", "`fake_balanced_gate` must be a boolean"),
        ("    parallelism: {}\n", "unknown key `parallelism`"),
        ("    nightly: true\n", "unknown key `nightly`"),
        ("    2: true\n", "key `2` must be a string"),
    ],
)
def test_linter_rejects_invalid_perf_testing(block: str, expected: str):
    errors = _perf_errors(block)

    assert len(errors) == 1
    assert expected in errors[0].message


def test_linter_rejects_non_mapping_perf_testing():
    errors = _perf_errors("    - a\n")

    assert len(errors) == 1
    assert "must be a mapping" in errors[0].message


def test_linter_rejects_duplicate_perf_testing_keys_with_line_numbers():
    errors = _perf_errors("    schedule: nightly\n    schedule: weekly\n")

    assert len(errors) == 1
    assert "Duplicate key `schedule`" in errors[0].message
    assert errors[0].line == 6


def test_linter_rejects_duplicate_fake_gate_keys():
    errors = _perf_errors("    fake_balanced_gate: true\n    fake_balanced_gate: false\n")

    assert len(errors) == 1
    assert "Duplicate key `fake_balanced_gate`" in errors[0].message
    assert errors[0].line == 6


@pytest.mark.parametrize("key", ["recipe", "dataset.seq_len", "step_scheduler.local_batch_size", "distributed.ep_size"])
def test_linter_rejects_workload_changes_in_perf_testing(key: str) -> None:
    errors = _perf_errors(f"    config_overrides:\n      {key}: 4\n")

    assert len(errors) == 1
    assert "unknown key `config_overrides`" in errors[0].message


def test_linter_reports_perf_testing_error_on_offending_line():
    errors = _perf_errors("    schedule: 0\n")

    assert errors[0].line == 5


def test_linter_still_validates_perf_testing_for_deprecated_recipes():
    text = (
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  deprecated: true\n  perf_testing:\n    time: nope\n"
    )

    errors = lint_yaml_text(text, Path("examples/llm_finetune/perf.yaml"), Path.cwd())

    messages = " ".join(e.message for e in errors)
    assert "`schedule` is required" in messages
    assert "`time` must be" in messages


@pytest.mark.parametrize(
    ("first", "second", "invalid_line"),
    [("bogus", "nightly", None), ("nightly", "bogus", 7)],
)
def test_linter_checks_the_last_duplicate_perf_testing_section(first: str, second: str, invalid_line: int | None):
    text = (
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: h100\n"
        f"  perf_testing:\n    schedule: {first}\n  perf_testing:\n    schedule: {second}\n  nemo_ci_schedule: skipped\n"
    )

    errors = lint_yaml_text(text, Path("examples/llm_finetune/perf.yaml"), Path.cwd())

    assert "Duplicate key `perf_testing` in `ci`" in errors[0].message
    assert errors[0].line == 6
    if invalid_line is None:
        assert len(errors) == 1
    else:
        assert len(errors) == 2
        assert "`schedule` must be one of" in errors[1].message
        assert errors[1].line == invalid_line


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_linter_accepts_optional_performance_validation(enabled: str) -> None:
    assert _perf_errors(f"    validation: {enabled}\n") == []


@pytest.mark.parametrize("value", ["null", "0", "1", "'false'", "[]", "{}"])
def test_linter_rejects_non_boolean_performance_validation(value: str) -> None:
    errors = _perf_errors(f"    validation: {value}\n")
    assert len(errors) == 1
    assert "`validation` must be a boolean" in errors[0].message
    assert errors[0].line == 5


def test_linter_rejects_duplicate_performance_validation() -> None:
    errors = _perf_errors("    validation: true\n    validation: false\n")
    assert len(errors) == 1
    assert "Duplicate key `validation`" in errors[0].message


def _schedule_errors(value: str, *, path: str = "examples/llm_finetune/recipe.yaml", extra: str = ""):
    """Lint a recipe whose ``ci.nemo_ci_schedule`` is ``value``, after any ``extra`` ``ci`` lines."""
    text = f"recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: h100\n{extra}  nemo_ci_schedule: {value}\n"
    return lint_yaml_text(text, Path(path), Path.cwd())


@pytest.mark.parametrize("value", ["nightly", "release", "nightly,release", "skipped"])
def test_linter_accepts_schedule_values(value: str) -> None:
    assert _schedule_errors(value) == []


def test_linter_requires_nemo_ci_schedule() -> None:
    errors = lint_yaml_text(
        "recipe: TrainFinetuneRecipeForNextTokenPrediction\nci:\n  cluster_tag: h100\n",
        Path("examples/llm_finetune/recipe.yaml"),
        Path.cwd(),
    )

    assert len(errors) == 1
    assert "`ci.nemo_ci_schedule` is missing" in errors[0].message
    assert errors[0].line == 2


@pytest.mark.parametrize(
    "value", ["release,nightly", "'nightly, release'", "nightly,skipped", "test", "true", "''", "null", "[nightly]"]
)
def test_linter_rejects_other_schedule_values(value: str) -> None:
    errors = _schedule_errors(value)

    assert len(errors) == 1
    assert "`ci.nemo_ci_schedule` is" in errors[0].message
    assert "one of: nightly, release, nightly,release, skipped" in errors[0].message
    assert errors[0].line == 4


@pytest.mark.parametrize("value", ["nightly", "release", "nightly,release"])
def test_linter_requires_skipped_schedule_for_deprecated_recipes(value: str) -> None:
    errors = _schedule_errors(value, extra="  deprecated: true\n")

    assert len(errors) == 1
    assert "A deprecated recipe runs no tests" in errors[0].message
    assert errors[0].line == 5
    assert _schedule_errors("skipped", extra="  deprecated: true\n") == []


@pytest.mark.parametrize(
    "path",
    [
        "examples/llm_benchmark/recipe.yaml",
        "examples/diffusion/generate/recipe.yaml",
        "examples/retrieval/nemotron_3_5/recipe.yaml",
    ],
)
def test_linter_requires_skipped_schedule_outside_schedule_folders(path: str) -> None:
    errors = _schedule_errors("nightly", path=path)

    assert len(errors) == 1
    assert "NeMo-CI schedules run only recipes under examples/llm_finetune" in errors[0].message
    assert _schedule_errors("skipped", path=path) == []
