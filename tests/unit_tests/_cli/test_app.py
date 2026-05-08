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

import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

sys.modules["nemo_run"] = mock.MagicMock()
sys.modules["torch.distributed.run"] = mock.MagicMock()

import nemo_automodel.cli.app as module
import nemo_automodel.cli.utils as utils

DATA_URL_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/lzTRVwAAAABJRU5ErkJggg=="
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_yaml_file():
    config_data = {"dummy_key": "dummy_value"}
    tmp_dir = tempfile.mkdtemp()
    tmp_file = Path(tmp_dir) / "config.yaml"
    with open(tmp_file, "w") as f:
        yaml.dump(config_data, f)
    yield tmp_file
    shutil.rmtree(tmp_dir)


@pytest.fixture
def recipe_yaml(tmp_path):
    """YAML with a recipe._target_ (no launcher section)."""
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "recipe": {
                    "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
                },
                "step_scheduler": {"num_epochs": 1},
            }
        )
    )
    return cfg


@pytest.fixture
def nemo_run_yaml(tmp_path):
    """YAML with recipe + nemo_run section."""
    cfg = tmp_path / "nemo_run.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "recipe": {
                    "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
                },
                "step_scheduler": {"num_epochs": 1},
                "nemo_run": {
                    "executor": "local",
                    "num_gpus_per_node": 1,
                },
            }
        )
    )
    return cfg


@pytest.fixture
def clear_recipe_discovery_cache():
    utils._discover_recipe_classes.cache_clear()
    yield
    utils._discover_recipe_classes.cache_clear()


# ---------------------------------------------------------------------------
# load_yaml tests
# ---------------------------------------------------------------------------


def test_load_yaml_valid(tmp_yaml_file):
    data = module.load_yaml(tmp_yaml_file)
    assert isinstance(data, dict)
    assert "dummy_key" in data


def test_load_yaml_missing():
    with pytest.raises(FileNotFoundError):
        module.load_yaml("non_existent.yaml")


def test_load_yaml_bad_format(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\n  -")
    with pytest.raises(Exception):
        module.load_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# recipe resolution tests
# ---------------------------------------------------------------------------


def test_discover_recipe_classes_scans_recipe_modules(monkeypatch, tmp_path, clear_recipe_discovery_cache):
    recipes_dir = tmp_path / "nemo_automodel" / "recipes"
    recipe_file = recipes_dir / "llm" / "train_ft.py"
    recipe_file.parent.mkdir(parents=True)
    recipe_file.write_text(
        "class BaseRecipe:\n"
        "    pass\n\n"
        "class HelperClass:\n"
        "    pass\n\n"
        "class TrainFinetuneRecipeForNextTokenPrediction(BaseRecipe):\n"
        "    pass\n\n"
        "class EvalRecipe:\n"
        "    pass\n"
    )
    private_file = recipes_dir / "_private.py"
    private_file.write_text("class HiddenRecipe:\n    pass\n")

    monkeypatch.setattr(utils, "_RECIPES_DIR", recipes_dir)

    assert utils._discover_recipe_classes() == {
        "EvalRecipe": "nemo_automodel.recipes.llm.train_ft.EvalRecipe",
        "TrainFinetuneRecipeForNextTokenPrediction": (
            "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"
        ),
    }


def test_resolve_recipe_name_resolves_bare_class_name(monkeypatch):
    expected = "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"
    monkeypatch.setattr(
        utils,
        "_discover_recipe_classes",
        lambda: {"TrainFinetuneRecipeForNextTokenPrediction": expected},
    )

    assert module.resolve_recipe_name("TrainFinetuneRecipeForNextTokenPrediction") == expected


def test_resolve_recipe_name_returns_fqn_unchanged(monkeypatch):
    raw = "nemo_automodel.recipes.llm.train_ft.Foo"
    discover = mock.MagicMock()
    monkeypatch.setattr(utils, "_discover_recipe_classes", discover)

    assert utils.resolve_recipe_name(raw) == raw
    discover.assert_not_called()


def test_resolve_recipe_name_raises_for_unknown_bare_name(monkeypatch):
    monkeypatch.setattr(
        utils,
        "_discover_recipe_classes",
        lambda: {"KnownRecipe": "nemo_automodel.recipes.llm.train_ft.KnownRecipe"},
    )

    with pytest.raises(ValueError, match="Unknown recipe class 'MissingRecipe'") as exc_info:
        utils.resolve_recipe_name("MissingRecipe")

    assert "  - KnownRecipe" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_parser tests
# ---------------------------------------------------------------------------


def test_build_parser_requires_config():
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_accepts_config(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("foo: bar")
    parser = module.build_parser()
    args, _ = parser.parse_known_args([str(cfg)])
    assert args.inputs == [str(cfg)]
    assert args.nproc_per_node is None


def test_build_parser_accepts_quick_lora_inputs(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": []}\n')
    parser = module.build_parser()
    args, _ = parser.parse_known_args(["Qwen/Qwen2.5-1.5B-Instruct", str(data)])
    assert args.inputs == ["Qwen/Qwen2.5-1.5B-Instruct", str(data)]


def test_build_parser_nproc_per_node(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("foo: bar")
    parser = module.build_parser()
    args, _ = parser.parse_known_args([str(cfg), "--nproc-per-node=4"])
    assert args.nproc_per_node == 4


def test_quick_lora_config_uses_openai_chat_dataset(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": []}\n')

    config = module.build_lora_config("Qwen/Qwen2.5-1.5B-Instruct", str(data), nproc_per_node=4)

    assert config["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"
    assert config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["dataset"]["_target_"] == "nemo_automodel.components.datasets.llm.chat_dataset.OpenAIChatDataset"
    assert config["dataset"]["path_or_dataset_id"] == str(data.resolve())
    assert config["peft"]["_target_"] == "nemo_automodel.components._peft.lora.PeftConfig"
    assert config["step_scheduler"]["global_batch_size"] % 4 == 0


def test_quick_lora_config_auto_detects_vlm_dataset(tmp_path):
    data = tmp_path / "vlm.jsonl"
    data.write_text(
        '{"messages": [{"role": "user", "content": ['
        '{"type": "text", "text": "What is shown?"}, '
        f'{{"type": "image_url", "image_url": {{"url": "{DATA_URL_IMAGE}"}}}}'
        ']}, {"role": "assistant", "content": "A tiny image."}]}\n'
    )

    config = module.build_lora_config("Qwen/Qwen2.5-VL-3B-Instruct", str(data), nproc_per_node=2)

    assert config["recipe"] == "FinetuneRecipeForVLM"
    assert config["model"]["_target_"] == "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
    assert config["processor"]["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert config["dataset"]["_target_"] == (
        "nemo_automodel.components.datasets.vlm.datasets.make_openai_vlm_chat_dataset"
    )
    assert config["dataset"]["path_or_dataset"] == str(data.resolve())
    assert config["peft"]["exclude_modules"]
    assert "shuffle" not in config["dataloader"]
    assert config["step_scheduler"]["global_batch_size"] % 2 == 0


def test_quick_lora_config_rejects_missing_local_jsonl():
    with pytest.raises(FileNotFoundError, match="Training data file was not found"):
        module.build_lora_config("Qwen/Qwen2.5-1.5B-Instruct", "missing.jsonl")


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------


def test_main_missing_recipe(monkeypatch, tmp_yaml_file):
    """YAML without recipe._target_ should exit with error."""
    monkeypatch.setattr("sys.argv", ["automodel", str(tmp_yaml_file)])
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 1


def test_main_dispatches_to_interactive(monkeypatch, recipe_yaml):
    """No launcher section -> InteractiveLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(recipe_yaml)])

    launched = {}

    class FakeInteractiveLauncher:
        def launch(self, config, config_path, recipe_target, nproc, extra):
            launched["recipe_target"] = recipe_target
            launched["nproc"] = nproc
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.interactive.InteractiveLauncher",
        FakeInteractiveLauncher,
    )
    result = module.main()
    assert result == 0
    assert "TrainFinetuneRecipeForNextTokenPrediction" in launched["recipe_target"]


def test_main_dispatches_to_nemo_run(monkeypatch, nemo_run_yaml):
    """nemo_run: section -> NemoRunLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(nemo_run_yaml)])

    launched = {}

    class FakeNemoRunLauncher:
        def launch(self, config, config_path, recipe_target, launcher_config, extra):
            launched["launcher_config"] = launcher_config
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.nemo_run.launcher.NemoRunLauncher",
        FakeNemoRunLauncher,
    )
    result = module.main()
    assert result == 0
    assert launched["launcher_config"]["executor"] == "local"


def test_main_resolves_skypilot_env_vars(monkeypatch, tmp_path):
    cfg = tmp_path / "skypilot.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "recipe": {
                    "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
                },
                "skypilot": {
                    "cloud": "kubernetes",
                    "hf_token": "${HF_TOKEN}",
                    "env_vars": {"WANDB_API_KEY": "${WANDB_API_KEY,}"},
                },
            }
        )
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_test_key")
    monkeypatch.setattr("sys.argv", ["automodel", str(cfg)])

    launched = {}

    class FakeSkyPilotLauncher:
        def launch(self, config, config_path, recipe_target, launcher_config, extra):
            launched["launcher_config"] = launcher_config
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.skypilot.launcher.SkyPilotLauncher",
        FakeSkyPilotLauncher,
    )
    result = module.main()
    assert result == 0
    assert launched["launcher_config"]["hf_token"] == "hf_test_token"
    assert launched["launcher_config"]["env_vars"]["WANDB_API_KEY"] == "wandb_test_key"


def test_main_passes_extra_args(monkeypatch, recipe_yaml):
    """Extra CLI args should be forwarded to the launcher."""
    monkeypatch.setattr(
        "sys.argv",
        ["automodel", str(recipe_yaml), "--model.pretrained_model_name_or_path=foo"],
    )

    launched = {}

    class FakeInteractiveLauncher:
        def launch(self, config, config_path, recipe_target, nproc, extra):
            launched["extra"] = extra
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.interactive.InteractiveLauncher",
        FakeInteractiveLauncher,
    )
    module.main()
    assert "--model.pretrained_model_name_or_path=foo" in launched["extra"]


def test_main_quick_lora_generates_config_and_dispatches(monkeypatch, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n')
    monkeypatch.setattr(
        "sys.argv",
        [
            "automodel",
            "Qwen/Qwen2.5-1.5B-Instruct",
            str(data),
            "--step_scheduler.max_steps=2",
        ],
    )
    monkeypatch.setattr(module, "_infer_local_worker_count", lambda nproc: 4)
    monkeypatch.setattr(
        module,
        "resolve_recipe_name",
        lambda raw: "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
    )

    launched = {}

    def fake_parse_args_and_load_config(config_path, argv=None):
        generated = yaml.safe_load(Path(config_path).read_text())
        launched["generated_config"] = generated
        launched["parse_argv"] = argv
        launched["generated_path"] = Path(config_path)
        return {"parsed": True}

    class FakeInteractiveLauncher:
        def launch(self, config, config_path, recipe_target, nproc, extra):
            launched["config"] = config
            launched["config_path_exists_during_launch"] = Path(config_path).exists()
            launched["recipe_target"] = recipe_target
            launched["nproc"] = nproc
            launched["extra"] = extra
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.config._arg_parser.parse_args_and_load_config",
        fake_parse_args_and_load_config,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.launcher.interactive.InteractiveLauncher",
        FakeInteractiveLauncher,
    )

    result = module.main()

    assert result == 0
    assert launched["config"] == {"parsed": True}
    assert launched["config_path_exists_during_launch"] is True
    assert not launched["generated_path"].exists()
    assert launched["generated_config"]["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert launched["generated_config"]["dataset"]["_target_"] == (
        "nemo_automodel.components.datasets.llm.chat_dataset.OpenAIChatDataset"
    )
    assert launched["generated_config"]["dataset"]["path_or_dataset_id"] == str(data.resolve())
    assert launched["generated_config"]["step_scheduler"]["global_batch_size"] == 8
    assert "--step_scheduler.max_steps=2" in launched["parse_argv"]
    assert "--step_scheduler.max_steps=2" in launched["extra"]
    assert launched["nproc"] is None


# ---------------------------------------------------------------------------
# Repo structure test (unchanged)
# ---------------------------------------------------------------------------


def test_repo_structure():
    repo_root = Path(__file__).parents[3]
    assert (repo_root / "nemo_automodel" / "cli" / "app.py").exists()
    assert (repo_root / "nemo_automodel").exists()
    assert (repo_root / "nemo_automodel" / "components").exists()
    assert (repo_root / "nemo_automodel" / "recipes").exists()
    assert (repo_root / "nemo_automodel" / "shared").exists()
    assert (repo_root / "docs").exists()
    assert (repo_root / "examples").exists()
