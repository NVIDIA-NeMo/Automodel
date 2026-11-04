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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("stage", ["sft_vllm_deploy", "peft_vllm_deploy"])
@pytest.mark.parametrize("exit_code", [0, 13])
def test_deployment_uses_private_module_cache(tmp_path: Path, stage: str, exit_code: int) -> None:
    launcher = Path(__file__).resolve().parents[3] / "tests/ci_tests/scripts/vllm_launcher.sh"
    checkpoint = tmp_path / "recipe/robustness_checkpoint/epoch_0_step_7/model/consolidated"
    checkpoint.mkdir(parents=True)
    shared_cache = tmp_path / "hf_home/modules"
    shared_module = shared_cache / "transformers_modules/consolidated/configuration_model.py"
    shared_module.parent.mkdir(parents=True)
    shared_module.write_text("shared cached code\n")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'cache = Path(os.environ["HF_MODULES_CACHE"])\n'
        'Path(os.environ["TEST_RESULT"]).write_text(json.dumps({\n'
        '    "cache": str(cache), "hf_home": os.environ["HF_HOME"], "argv": sys.argv[1:]}))\n'
        "assert cache.is_dir() and not list(cache.iterdir())\n"
        '(cache / "configuration_model.py").write_text("private cached code\\n")\n'
        'sys.exit(int(os.environ["TEST_EXIT_CODE"]))\n'
    )
    stubs = tmp_path / "stubs.sh"
    stubs.write_text('cd() { return 0; }\npython3() { echo 0; }\npython() { "$TEST_PYTHON" "$TEST_PROBE" "$@"; }\n')
    result_path = tmp_path / "result.json"
    env = {
        **os.environ,
        "BASH_ENV": str(stubs),
        "TEST_PYTHON": sys.executable,
        "TEST_PROBE": str(probe),
        "TEST_RESULT": str(result_path),
        "TEST_EXIT_CODE": str(exit_code),
        "CONFIG_PATH": "recipe.yaml",
        "PIPELINE_DIR": str(tmp_path),
        "TEST_NAME": "recipe_vllm_deploy",
        "CI_JOB_STAGE": stage,
        "HF_HOME": str(shared_cache.parent),
        "HF_MODULES_CACHE": str(shared_cache),
        "TMPDIR": str(tmp_path),
    }
    caches = []
    for _ in range(2):
        process = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=10)
        assert process.returncode == exit_code, process.stderr
        result = json.loads(result_path.read_text())
        cache = Path(result["cache"])
        assert cache != shared_cache
        assert not cache.exists(), "The private module cache must be removed on success and failure"
        assert result["hf_home"] == str(shared_cache.parent)
        mode_index = result["argv"].index("--deploy_mode") + 1
        assert result["argv"][mode_index] == stage.split("_")[0]
        assert shared_module.read_text() == "shared cached code\n"
        caches.append(cache)
    assert caches[0] != caches[1]
