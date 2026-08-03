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

from ruamel.yaml import YAML

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


def test_generate_checkpoint_robustness_process_isolation_derives_phases(tmp_path):
    config = Path("large_lora.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    process_isolation: true
    check_source_load_parity: true
    no_check_resume: true
    trust_remote_code: true
    hf_device_map_auto: true
    hf_device_map_max_memory_gib: 55
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["HAS_ROBUSTNESS"] == "true"
    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload"
    )


def test_generate_checkpoint_robustness_process_isolation_preserves_full_default(tmp_path):
    config = Path("full_robustness.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    process_isolation: true
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "train_and_save automodel_reload hf_reload resume_baseline resume"
    )


def test_generate_checkpoint_robustness_process_isolation_honors_skips(tmp_path):
    config = Path("reload_only.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    process_isolation: true
    skip_hf_reload: true
    no_check_resume: true
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == "train_and_save automodel_reload"


def test_generate_checkpoint_robustness_process_isolation_allows_phase_override(tmp_path):
    config = Path("custom_phases.yaml")
    (tmp_path / config).write_text(
        """
ci:
  env_vars:
    CHECKPOINT_ROBUSTNESS_PHASES: "train_and_save automodel_reload"
  checkpoint_robustness:
    process_isolation: true
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == "train_and_save automodel_reload"


def test_generate_qwen3_moe_lora_uses_isolated_reload_phases():
    config = Path("examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml")

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", "."))
    robustness = YAML(typ="safe").load(config)["ci"]["checkpoint_robustness"]

    variables = jobs[""]["variables"]
    assert variables["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert variables["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload"
    )
    assert robustness["check_source_load_parity"] is True
    assert robustness["skip_hf_logit_parity"] is True
    assert robustness["source_load_kl_threshold"] == 3e-2
    assert robustness["source_load_mean_kl_threshold"] == 6e-3
    assert robustness["source_load_cosine_threshold"] == 0.997


def test_generate_qwen3_moe_te_deepep_uses_isolated_source_and_reload_phases():
    config = Path("examples/llm_finetune/qwen/qwen3_moe_30b_te_deepep.yaml")

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", "."))
    robustness = YAML(typ="safe").load(config)["ci"]["checkpoint_robustness"]

    variables = jobs[""]["variables"]
    assert variables["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert variables["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload"
    )
    assert robustness["check_source_load_parity"] is True
    assert robustness["source_load_kl_threshold"] == 1e-1
    assert robustness["source_load_mean_kl_threshold"] == 1.5e-2
    assert robustness["source_load_cosine_threshold"] == 0.995
