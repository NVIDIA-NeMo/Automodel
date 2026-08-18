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


def test_generate_gpt_oss_120b_release_job_uses_ep64():
    config = Path("examples/llm_finetune/gpt_oss/gpt_oss_120b.yaml")

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", "."))
    recipe = YAML(typ="safe").load(config)

    variables = jobs[""]["variables"]
    world_size = variables["TEST_NODE_COUNT"] * 8
    assert variables["TEST_NODE_COUNT"] == 8
    assert variables["LOCAL_BATCH_SIZE"] == 8
    assert recipe["distributed"]["ep_size"] == world_size
    assert recipe["distributed"]["activation_checkpointing"] is False


def test_generate_gpt_oss_120b_benchmark_job_uses_ep64_without_activation_checkpointing():
    config = Path("examples/llm_benchmark/gpt_oss/gptoss_120b_te_deepep.yaml")

    jobs = dict(generate_job(config, {}, "performance", "llm_benchmark", "."))
    recipe = YAML(typ="safe").load(config)

    variables = jobs[""]["variables"]
    world_size = variables["TEST_NODE_COUNT"] * 8
    assert variables["TEST_NODE_COUNT"] == 8
    assert variables["EP_SIZE"] == world_size
    assert recipe["distributed"]["ep_size"] == world_size
    assert recipe["distributed"]["activation_checkpointing"] is False


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
  checkpoint_robustness: {}
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload resume"
    )


def test_generate_checkpoint_robustness_process_isolation_honors_skips(tmp_path):
    config = Path("reload_only.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    process_isolation: true
    skip_source_load_parity: true
    skip_hf_reload: true
    skip_resume: true
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == "train_and_save automodel_reload"


def test_generate_checkpoint_robustness_process_isolation_is_default_and_cross_tp_is_last(tmp_path):
    config = Path("dense_model.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    cross_tp_size: 2
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert jobs[""]["variables"]["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload resume cross_tp_reload"
    )


def test_generate_checkpoint_robustness_allows_single_process_fallback(tmp_path):
    config = Path("fallback_model.yaml")
    (tmp_path / config).write_text(
        """
ci:
  checkpoint_robustness:
    process_isolation: false
""",
        encoding="utf-8",
    )

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", str(tmp_path)))

    assert "CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION" not in jobs[""]["variables"]
    assert "CHECKPOINT_ROBUSTNESS_PHASES" not in jobs[""]["variables"]


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


def test_generate_qwen3_moe_lora_uses_all_isolated_checkpoint_phases():
    config = Path("examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml")

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", "."))
    ci_config = YAML(typ="safe").load(config)["ci"]
    robustness = ci_config["checkpoint_robustness"]

    variables = jobs[""]["variables"]
    assert "CHECKPOINT_ROBUSTNESS_PHASES" not in ci_config.get("env_vars", {})
    assert variables["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert variables["TIME"] == "01:00:00"
    assert variables["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert variables["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload resume"
    )
    assert "check_source_load_parity" not in robustness
    assert "skip_source_load_parity" not in robustness
    assert "skip_automodel_reload_logit_parity" not in robustness
    assert "skip_hf_reload_logit_parity" not in robustness
    assert "skip_resume" not in robustness
    for key in (
        "source_load_kl_threshold",
        "source_load_mean_kl_threshold",
        "source_load_cosine_threshold",
    ):
        assert key not in robustness


def test_generate_qwen3_moe_te_deepep_uses_isolated_source_and_reload_phases():
    config = Path("examples/llm_finetune/qwen/qwen3_moe_30b_te_deepep.yaml")

    jobs = dict(generate_job(config, {}, "release", "llm_finetune", "."))
    ci_config = YAML(typ="safe").load(config)["ci"]
    robustness = ci_config["checkpoint_robustness"]

    variables = jobs[""]["variables"]
    assert "CHECKPOINT_ROBUSTNESS_PHASES" not in ci_config.get("env_vars", {})
    assert variables["TIME"] == "00:30:00"
    assert variables["CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION"] == "true"
    assert variables["CHECKPOINT_ROBUSTNESS_PHASES"] == (
        "source_load_reference source_load_parity train_and_save automodel_reload hf_reload resume"
    )
    assert "check_source_load_parity" not in robustness
    assert "skip_source_load_parity" not in robustness
    assert "source_load_kl_threshold" not in robustness
    assert "source_load_mean_kl_threshold" not in robustness
    assert "source_load_cosine_threshold" not in robustness
    assert "skip_resume" not in robustness
    assert robustness["trust_remote_code"] is True
    assert robustness["hf_device_map_auto"] is True
