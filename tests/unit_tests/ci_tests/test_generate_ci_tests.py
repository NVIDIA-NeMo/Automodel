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

from pathlib import Path

import pytest

from tests.ci_tests.utils.generate_ci_tests import generate_job, generate_pipeline


def _write_ci_recipe(
    automodel_dir: Path,
    name: str,
    *,
    time: str,
    nodes: int = 1,
    extra_ci: str = "",
) -> Path:
    recipe = automodel_dir / "examples" / "llm_finetune" / "family" / f"{name}.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(
        f'ci:\n  time: "{time}"\n  nodes: {nodes}\n  recipe_owner: test\n{extra_ci}',
        encoding="utf-8",
    )
    return recipe


def _write_ci_lists(automodel_dir: Path, nightly_names: list[str], *, exempt_configs: list[str] | None = None) -> None:
    config_dir = automodel_dir / "tests" / "ci_tests" / "configs" / "llm_finetune"
    config_dir.mkdir(parents=True)
    nightly_lines = "\n".join(f"  - family/{name}.yaml" for name in nightly_names)
    (config_dir / "nightly_recipes.yml").write_text(f"configs:\n{nightly_lines}\n", encoding="utf-8")
    exempt_lines = "\n".join(f"  {name}:\n    reason: test" for name in (exempt_configs or []))
    (config_dir / "override_recipes.yml").write_text(
        f"exempt_configs:\n{exempt_lines}\n" if exempt_lines else "exempt_configs: {}\n",
        encoding="utf-8",
    )


def _job_names(pipeline: dict) -> set[str]:
    return set(pipeline) - {"include"}


def test_generate_deepseek_v4_pretrain_nightly_job():
    pipeline = generate_pipeline(".", "nightly", "llm_pretrain")

    job = pipeline["deepseek_v4_flash_pretrain"]
    assert job["extends"] == ".llm_pretrain_test"
    assert job["stage"] == "pretrain"
    assert job["variables"]["CONFIG_PATH"] == ("examples/llm_pretrain/deepseek_v4/deepseek_v4_flash_pretrain.yaml")
    assert job["variables"]["REQUIRE_FINITE_METRICS"] == "true"
    assert job["variables"]["TEST_NODE_COUNT"] == 2


def test_generate_deepseek_v4_pretrain_release_job():
    pipeline = generate_pipeline(".", "release", "llm_pretrain")

    job = pipeline["deepseek_v4_flash_pretrain"]
    assert job["extends"] == ".llm_pretrain_test"
    assert job["variables"]["TEST_LEVEL"] == "release"


def test_generate_vllm_deploy_time_override(tmp_path):
    config = Path("model_peft.yaml")
    (tmp_path / config).write_text(
        """
ci:
  time: "00:25:00"
  vllm_deploy: true
  vllm_deploy_time: "00:30:00"
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "nightly", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["TIME"] == "00:25:00"
    assert jobs["_vllm_deploy"]["variables"]["TIME"] == "00:30:00"


def test_rolling_release_shards_preserve_nightly_and_cover_release_only(tmp_path):
    _write_ci_lists(tmp_path, ["core"], exempt_configs=["excluded"])
    _write_ci_recipe(tmp_path, "core", time="00:10:00")
    for index in range(5):
        _write_ci_recipe(tmp_path, f"release_{index}", time=f"00:{index + 11:02d}:00")
    _write_ci_recipe(tmp_path, "excluded", time="01:00:00")
    _write_ci_recipe(tmp_path, "known_issue", time="01:00:00", extra_ci="  known_issue_id: AM-123\n")

    rolling_jobs = []
    for shard_index in range(5):
        pipeline = generate_pipeline(
            str(tmp_path),
            "nightly",
            "llm_finetune",
            rolling_release_shards=5,
            rolling_release_shard=shard_index,
        )
        assert pipeline["core"]["variables"]["TEST_LEVEL"] == "nightly"
        rolling_jobs.append(_job_names(pipeline) - {"core"})

    assert set.union(*rolling_jobs) == {f"release_{index}" for index in range(5)}
    assert sum(len(jobs) for jobs in rolling_jobs) == 5


def test_rolling_release_shards_balance_declared_node_time(tmp_path):
    _write_ci_lists(tmp_path, ["core"])
    _write_ci_recipe(tmp_path, "core", time="00:10:00")
    recipe_minutes = {"r60": 60, "r50": 50, "r40": 40, "r30": 30, "r20": 20, "r10": 10}
    for name, minutes in recipe_minutes.items():
        _write_ci_recipe(tmp_path, name, time=f"{minutes // 60:02d}:{minutes % 60:02d}:00")

    shard_costs = []
    for shard_index in range(2):
        pipeline = generate_pipeline(
            str(tmp_path),
            "nightly",
            "llm_finetune",
            rolling_release_shards=2,
            rolling_release_shard=shard_index,
        )
        shard_costs.append(sum(recipe_minutes[name] for name in _job_names(pipeline) - {"core"}))

    assert sorted(shard_costs) == [100, 110]


def test_rolling_release_rejects_invalid_scope_or_shard(tmp_path):
    _write_ci_lists(tmp_path, ["core"])
    _write_ci_recipe(tmp_path, "core", time="00:10:00")

    with pytest.raises(ValueError, match="only be added to the nightly scope"):
        generate_pipeline(str(tmp_path), "release", "llm_finetune", rolling_release_shards=5)
    with pytest.raises(ValueError, match="requires a positive rolling release shard count"):
        generate_pipeline(str(tmp_path), "nightly", "llm_finetune", rolling_release_shard=1)
    with pytest.raises(ValueError, match=r"must be in \[0, 5\)"):
        generate_pipeline(
            str(tmp_path),
            "nightly",
            "llm_finetune",
            rolling_release_shards=5,
            rolling_release_shard=5,
        )
