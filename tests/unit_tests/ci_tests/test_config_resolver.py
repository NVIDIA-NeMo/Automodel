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

import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from ruamel.yaml import YAML

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "tests" / "ci_tests" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import config_resolver  # noqa: E402

yaml = YAML()


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------


def test_set_dotted_creates_nested_path():
    d: dict = {}
    config_resolver._set_dotted(d, "a.b.c", 5)
    assert d == {"a": {"b": {"c": 5}}}


def test_set_dotted_preserves_siblings():
    d = {"a": {"b": 1, "c": 2}}
    config_resolver._set_dotted(d, "a.b", 99)
    assert d == {"a": {"b": 99, "c": 2}}


def test_set_dotted_replaces_non_dict_intermediate():
    d = {"a": "scalar"}
    config_resolver._set_dotted(d, "a.b", 1)
    assert d == {"a": {"b": 1}}


@pytest.mark.parametrize(
    "raw,expected",
    [("5", 5), ("0", 0), ("-3", -3), ("1.5", 1.5), ("hello", "hello"), ("", "")],
)
def test_coerce(raw, expected):
    assert config_resolver._coerce(raw) == expected


# ---------------------------------------------------------------------------
# _resolve_env_layer
# ---------------------------------------------------------------------------


ENV_SPEC = {
    "MAX_STEPS": {"target": "step_scheduler.max_steps", "phases": ["nightly", "convergence"]},
    "LOCAL_BATCH_SIZE": {"target": "step_scheduler.local_batch_size", "phases": ["nightly"]},
}


def test_env_layer_applies_when_set_and_phase_matches(monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "99")
    monkeypatch.delenv("LOCAL_BATCH_SIZE", raising=False)
    layer = config_resolver._resolve_env_layer(ENV_SPEC, phase="nightly")
    assert layer == {"step_scheduler.max_steps": 99}


def test_env_layer_skips_when_phase_excluded(monkeypatch):
    monkeypatch.setenv("MAX_STEPS", "99")
    layer = config_resolver._resolve_env_layer(ENV_SPEC, phase="checkpoint_robustness")
    assert layer == {}


def test_env_layer_skips_when_var_unset(monkeypatch):
    monkeypatch.delenv("MAX_STEPS", raising=False)
    layer = config_resolver._resolve_env_layer(ENV_SPEC, phase="nightly")
    assert layer == {}


# ---------------------------------------------------------------------------
# _resolve_computed_layer
# ---------------------------------------------------------------------------


def test_computed_layer_substitutes_env(monkeypatch):
    monkeypatch.setenv("PIPELINE_DIR", "/p")
    monkeypatch.setenv("TEST_NAME", "t1")
    entries = [{
        "target": "checkpoint.checkpoint_dir",
        "format": "{PIPELINE_DIR}/{TEST_NAME}/checkpoint",
        "phases": ["nightly"],
    }]
    assert config_resolver._resolve_computed_layer(entries, "nightly") == {
        "checkpoint.checkpoint_dir": "/p/t1/checkpoint",
    }


def test_computed_layer_substitutes_date(monkeypatch):
    entries = [{
        "target": "wandb.project",
        "format": "test-{date:%Y%m%d}",
        "phases": ["convergence"],
    }]
    result = config_resolver._resolve_computed_layer(entries, "convergence")
    today = datetime.now().strftime("%Y%m%d")
    assert result == {"wandb.project": f"test-{today}"}


def test_computed_layer_phase_filter():
    entries = [{
        "target": "wandb.name",
        "format": "x",
        "phases": ["convergence"],
    }]
    assert config_resolver._resolve_computed_layer(entries, "nightly") == {}


def test_computed_layer_missing_substitution_exits(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    entries = [{"target": "x.y", "format": "{DOES_NOT_EXIST}", "phases": ["nightly"]}]
    with pytest.raises(SystemExit, match="missing substitution"):
        config_resolver._resolve_computed_layer(entries, "nightly")


# ---------------------------------------------------------------------------
# _resolve_conditional_layer
# ---------------------------------------------------------------------------


CONDITIONAL_ENTRIES = [
    {
        "when_recipe_contains_all": ["customizer_"],
        "apply": {"dataset.path_or_dataset_id": "{NEMO_CI_PATH}/prompt_completion/train.jsonl"},
    },
    {
        "when_recipe_contains_all": ["customizer_", "chat"],
        "apply": {"dataset.path_or_dataset_id": "{NEMO_CI_PATH}/chat/train.jsonl"},
    },
    {
        "when_recipe_contains_any": ["peft", "lora"],
        "phases": ["checkpoint_robustness"],
        "apply": {"peft.use_triton": False},
    },
]


def test_conditional_layer_contains_all_matches(monkeypatch):
    monkeypatch.setenv("NEMO_CI_PATH", "/data")
    out = config_resolver._resolve_conditional_layer(CONDITIONAL_ENTRIES, "nightly", "customizer_foo")
    assert out == {"dataset.path_or_dataset_id": "/data/prompt_completion/train.jsonl"}


def test_conditional_layer_later_rule_wins(monkeypatch):
    """Chat-specific customizer rule shadows the catch-all customizer rule."""
    monkeypatch.setenv("NEMO_CI_PATH", "/data")
    out = config_resolver._resolve_conditional_layer(CONDITIONAL_ENTRIES, "nightly", "customizer_foo_chat")
    assert out == {"dataset.path_or_dataset_id": "/data/chat/train.jsonl"}


def test_conditional_layer_skips_non_matching_recipe(monkeypatch):
    monkeypatch.setenv("NEMO_CI_PATH", "/data")
    assert config_resolver._resolve_conditional_layer(CONDITIONAL_ENTRIES, "nightly", "llama3_squad") == {}


def test_conditional_layer_phase_filter_excludes_non_robustness(monkeypatch):
    """peft rule is phase-filtered to robustness; nightly should not see it."""
    monkeypatch.setenv("NEMO_CI_PATH", "/data")
    out = config_resolver._resolve_conditional_layer(CONDITIONAL_ENTRIES, "nightly", "llama_lora")
    assert out == {}


def test_conditional_layer_contains_any_match():
    out = config_resolver._resolve_conditional_layer(CONDITIONAL_ENTRIES, "checkpoint_robustness", "llama_lora")
    assert out == {"peft.use_triton": False}


def test_conditional_layer_passes_non_string_values_through():
    """Non-string apply values (e.g. bool False) must not go through str.format."""
    entries = [{"when_recipe_contains_any": ["x"], "apply": {"some.flag": False, "some.num": 5}}]
    out = config_resolver._resolve_conditional_layer(entries, "nightly", "xfoo")
    assert out == {"some.flag": False, "some.num": 5}


# ---------------------------------------------------------------------------
# End-to-end: subprocess invocation against the real ci_config.yaml
# ---------------------------------------------------------------------------


RESOLVER = str(SCRIPTS_DIR / "config_resolver.py")


@pytest.fixture
def synthetic_recipe(tmp_path: Path) -> Path:
    """A tiny recipe with a ci.nightly override, written to tmp_path."""
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "step_scheduler:\n"
        "  global_batch_size: 8\n"
        "  max_steps: 1000\n"
        "ci:\n"
        "  recipe_owner: tester\n"
        "  nightly:\n"
        "    step_scheduler.max_steps: 7   # per-recipe override of phase default 50\n"
    )
    return path


def _run_resolver(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, RESOLVER, *args],
        check=True, capture_output=True, text=True,
        env={**({} if env is None else env), "PATH": "/usr/bin:/bin"},
    )


def test_end_to_end_phase_defaults_and_ci_section(tmp_path, synthetic_recipe):
    """Phase defaults apply; recipe ci.<phase> overrides them; ci: block preserved for downstream consumers."""
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1"}
    _run_resolver(["--base", str(synthetic_recipe), "--phase", "nightly", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    # Phase default (ckpt/val_every_steps) survives
    assert resolved["step_scheduler"]["ckpt_every_steps"] == 50
    assert resolved["step_scheduler"]["val_every_steps"] == 50
    # Recipe ci.nightly wins over phase default for max_steps (7, not 50)
    assert resolved["step_scheduler"]["max_steps"] == 7
    # Computed override applied
    assert resolved["checkpoint"]["checkpoint_dir"] == f"{tmp_path}/t1/checkpoint"


def test_end_to_end_env_overrides_ci_section(tmp_path, synthetic_recipe):
    """Env overrides win over recipe ci.<phase> (explicit user override beats persisted recipe config)."""
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1", "MAX_STEPS": "999"}
    _run_resolver(["--base", str(synthetic_recipe), "--phase", "nightly", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    assert resolved["step_scheduler"]["max_steps"] == 999  # env wins over ci.nightly's 7


def test_end_to_end_robustness_ignores_max_steps_env(tmp_path, synthetic_recipe):
    """ci_config.yaml's env entry restricts MAX_STEPS to non-robustness phases, so it must not leak in."""
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1", "MAX_STEPS": "999"}
    _run_resolver(["--base", str(synthetic_recipe), "--phase", "checkpoint_robustness", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    assert resolved["step_scheduler"]["max_steps"] == 5  # robustness phase default holds
    assert resolved["checkpoint"]["checkpoint_dir"] == f"{tmp_path}/t1/robustness_checkpoint"


def test_end_to_end_customizer_chat_path_wins(tmp_path):
    """A recipe whose stem contains both 'customizer_' and 'chat' picks up the chat dataset paths."""
    recipe = tmp_path / "customizer_nano_chat.yaml"
    recipe.write_text("step_scheduler: {global_batch_size: 8}\nci: {recipe_owner: t}\n")
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1", "NEMO_CI_PATH": "/mnt/nci"}
    _run_resolver(["--base", str(recipe), "--phase", "nightly", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    assert resolved["dataset"]["path_or_dataset_id"] == "/mnt/nci/datasets/customizer/sample-datasets/chat/train.jsonl"
    assert resolved["validation_dataset"]["path_or_dataset_id"] == "/mnt/nci/datasets/customizer/sample-datasets/chat/validation.jsonl"


def test_end_to_end_robustness_peft_disables_triton(tmp_path):
    """A robustness-phase peft recipe gets peft.use_triton: false applied."""
    recipe = tmp_path / "llama_peft.yaml"
    recipe.write_text("step_scheduler: {global_batch_size: 8}\nci: {recipe_owner: t}\n")
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1"}
    _run_resolver(["--base", str(recipe), "--phase", "checkpoint_robustness", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    assert resolved["peft"]["use_triton"] is False


def test_end_to_end_fixture_keys_not_applied_as_overrides(tmp_path):
    """Non-config fixture-arg keys in ci.checkpoint_robustness must not leak into the top-level config."""
    recipe = tmp_path / "llama_squad.yaml"
    recipe.write_text(
        "step_scheduler: {global_batch_size: 8}\n"
        "ci:\n"
        "  checkpoint_robustness:\n"
        "    hf_kl_threshold: 5e-3                       # fixture arg, must NOT become top-level\n"
        "    tokenizer_name: nvidia/Test                 # fixture arg, must NOT become top-level\n"
        "    dataset.limit_dataset_samples: 500          # dotted -> applied as override\n"
    )
    out = tmp_path / "resolved.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1"}
    _run_resolver(["--base", str(recipe), "--phase", "checkpoint_robustness", "--output", str(out)], env=env)

    resolved = yaml.load(out.open())
    # Dotted key applied as a normal override
    assert resolved["dataset"]["limit_dataset_samples"] == 500
    # Fixture args stay under ci.checkpoint_robustness for the consumer (pytest) to read,
    # and do NOT pollute the top level.
    assert "hf_kl_threshold" not in resolved
    assert "tokenizer_name" not in resolved
    assert resolved["ci"]["checkpoint_robustness"]["hf_kl_threshold"] == 5e-3


def test_end_to_end_dry_run_does_not_write(tmp_path, synthetic_recipe):
    out = tmp_path / "should_not_exist.yaml"
    env = {"PIPELINE_DIR": str(tmp_path), "TEST_NAME": "t1"}
    result = _run_resolver(["--base", str(synthetic_recipe), "--phase", "nightly", "--dry-run"], env=env)

    assert not out.exists()
    assert "Resolution stack" in result.stdout
    assert "[phase_defaults]" in result.stdout
    assert "[recipe.ci.nightly]" in result.stdout
    assert "[env]" in result.stdout
    assert "[computed]" in result.stdout
    # Resolved YAML body included
    resolved = yaml.load(io.StringIO(result.stdout.split("--- resolved config ---", 1)[1]))
    assert resolved["step_scheduler"]["max_steps"] == 7
