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

from tests.ci_tests.utils.generate_ci_tests import generate_job, generate_pipeline


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
