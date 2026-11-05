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

"""Exercise generated launcher commands through the real CLI worker dispatch."""

import importlib
import json
import runpy
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from nemo_automodel.cli import app
from nemo_automodel.components.launcher import interactive
from nemo_automodel.components.launcher.nemo_run import launcher as nemo_launcher
from nemo_automodel.components.launcher.skypilot import launcher as sky_launcher
from nemo_automodel.components.launcher.skypilot import utils as sky_utils

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("launcher", ["interactive", "interactive_explicit", "external", "skypilot", "nemo_run"])
@pytest.mark.parametrize("recipe_form", ["string", "target"])
@pytest.mark.parametrize("recipe_kind", ["training", "benchmark"])
def test_launcher_command_executes_recipe(
    launcher: str, recipe_form: str, recipe_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A recipe module without a main block reproduces the speculative recipe contract.
    # Only model work and process/job submission are replaced. CLI parsing, recipe
    # resolution, config packaging and worker dispatch run their production code.
    module_name = "launcher_dispatch_probe"
    recipe_file = tmp_path / f"{module_name}.py"
    base_import = ""
    base_class = "object"
    entry_method = "run_train_validation_loop"
    event = "train"
    if recipe_kind == "benchmark":
        # Retain the real benchmark class's CLI entry contract, replacing only
        # model setup and the GPU benchmark body.
        base_import = "from nemo_automodel.recipes.llm.benchmark import BenchmarkingRecipeForNextTokenPrediction\n"
        base_class = "BenchmarkingRecipeForNextTokenPrediction"
        entry_method = "run_benchmark"
        event = "benchmark"
    recipe_file.write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"{base_import}"
        f"class ProbeRecipe({base_class}):\n"
        "    def __init__(self, config):\n"
        "        self.partial_cuda_graph_manager = None\n"
        "        self.events = Path(config.get('events'))\n"
        "        self.events.write_text('init\\n')\n"
        "        observed = {key: config.get(key) for key in ('label', 'lr', 'restore_path')}\n"
        "        Path(config.get('observed')).write_text(json.dumps(observed))\n"
        "    def setup(self):\n"
        "        with self.events.open('a') as stream: stream.write('setup\\n')\n"
        f"    def {entry_method}(self):\n"
        f"        with self.events.open('a') as stream: stream.write('{event}\\n')\n"
        "        return 0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location(module_name, recipe_file)
    probe_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_module)
    monkeypatch.setitem(sys.modules, module_name, probe_module)
    monkeypatch.setattr(interactive, "_get_repo_root", lambda: tmp_path)
    for name in ("LOCAL_RANK", "RANK", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)

    # Other CLI tests install a global mock; use the real torchrun parser here.
    monkeypatch.delitem(sys.modules, "torch.distributed.run", raising=False)
    torch_run = importlib.import_module("torch.distributed.run")
    monkeypatch.setattr(torch_run, "determine_local_world_size", lambda nproc_per_node: 2)
    events = tmp_path / "events.txt"
    observed = tmp_path / "observed.json"
    target = f"{module_name}.ProbeRecipe"
    config = {
        "recipe": target if recipe_form == "string" else {"_target_": target},
        "events": str(events),
        "observed": str(observed),
        "label": "original",
        "lr": 1.0,
        "restore_path": "checkpoint with spaces",
    }
    config_file = tmp_path / "config with spaces.yaml"
    overrides = ["--label", "value with spaces and 'quotes'", "--lr=0.25"]
    worker_calls = []

    def run_worker(args: Namespace) -> int:
        worker_calls.append(args)
        assert len(worker_calls) == 1, "The worker recursively relaunched torchrun"
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "launcher-dispatch-test")
        monkeypatch.setattr(sys, "argv", [args.training_script, *args.training_script_args])
        if args.module:
            # Emulate a fresh worker import without shadowing the loaded CLI.
            monkeypatch.delitem(sys.modules, args.training_script, raising=False)
            runpy.run_module(args.training_script, run_name="__main__")
        else:
            # SkyPilot syncs the module into its remote workdir.
            script_path = recipe_file if launcher == "skypilot" else args.training_script
            runpy.run_path(str(script_path), run_name="__main__")
        return 0

    monkeypatch.setattr(torch_run, "run", run_worker)

    if launcher == "skypilot":
        config["skypilot"] = {"cloud": "aws", "accelerators": "A100:2", "job_dir": str(tmp_path / "sky jobs")}
        remote_config = tmp_path / "remote config.yaml"
        monkeypatch.setattr(sky_launcher, "REMOTE_CONFIG_PATH", str(remote_config))

        def submit_sky(sky_config, job_dir: str) -> int:
            # Emulate the config mount, then let bash expand the actual command.
            remote_config.write_text((Path(job_dir) / "job_config.yaml").read_text())
            assert "skypilot" not in yaml.safe_load(remote_config.read_text())
            command = 'torchrun() { printf "%s\\0" "$@"; }\n' + sky_config.command
            result = subprocess.run(["bash", "-c", command], capture_output=True, check=True)
            argv = result.stdout.decode().rstrip("\0").split("\0")
            return run_worker(torch_run.get_args_parser().parse_args(argv))

        monkeypatch.setattr(sky_utils, "submit_skypilot_job", submit_sky)
    elif launcher == "nemo_run":
        config["nemo_run"] = {"executor": "local", "job_dir": str(tmp_path / "nemo jobs")}
        remote_config = tmp_path / "packaged config.yaml"
        monkeypatch.setattr(nemo_launcher, "_REMOTE_CONFIG_PATH", str(remote_config))
        executor = MagicMock()
        executor.nproc_per_node.return_value = 2
        run = SimpleNamespace(LocalExecutor=lambda: executor, Script=SimpleNamespace, PatternPackager=SimpleNamespace)
        monkeypatch.setitem(sys.modules, "nemo_run", run)

        def submit_nemo(script, executor, **kwargs) -> int:
            remote_config.write_text(Path(executor.packager.include_pattern).read_text())
            assert "nemo_run" not in yaml.safe_load(remote_config.read_text())
            assert executor.launcher == "torchrun"
            argv = ["--nproc-per-node=2", *(["-m"] if script.m else []), script.path, *script.args]
            return run_worker(torch_run.get_args_parser().parse_args(argv))

        monkeypatch.setattr(nemo_launcher, "submit_nemo_run_job", submit_nemo)

    config_file.write_text(yaml.safe_dump(config))
    argv = ["automodel", str(config_file), *overrides]
    if launcher == "interactive_explicit":
        argv[2:2] = ["--nproc-per-node", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    if launcher == "external":
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "external-torchrun")
    expected_result = None if recipe_kind == "benchmark" and launcher == "external" else 0
    assert app.main() == expected_result
    assert events.exists(), "The selected worker exited without constructing the recipe"
    assert events.read_text().splitlines() == ["init", "setup", event]
    assert json.loads(observed.read_text()) == {
        "label": overrides[1],
        "lr": 0.25,
        "restore_path": "checkpoint with spaces",
    }
    assert len(worker_calls) == (0 if launcher == "external" else 1)
