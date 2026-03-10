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

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from nemo_automodel.components.launcher.k8s.launcher import K8sLauncher, CONFIG_MOUNT_PATH


@pytest.fixture
def launcher_config():
    return {
        "num_nodes": 2,
        "gpus_per_node": 8,
        "image": "nvcr.io/nvidia/nemo-automodel:test",
        "namespace": "test-ns",
    }


@pytest.fixture
def recipe_config():
    return {"trainer": {"max_steps": 100}, "model": {"name": "llama"}}


RECIPE_TARGET = "nemo_automodel.recipes.llm.train_ft.TrainRecipe"


def _mock_subprocess(returncode=0, stdout="pytorchjob created", stderr=""):
    return mock.patch(
        "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
        return_value=SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )


class TestK8sLauncher:
    def test_launch_success(self, launcher_config, recipe_config):
        with _mock_subprocess(returncode=0) as mock_run:
            launcher = K8sLauncher()
            rc = launcher.launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert rc == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "kubectl"
        assert call_args[0][0][1] == "apply"

    def test_launch_failure(self, launcher_config, recipe_config):
        with _mock_subprocess(returncode=1, stderr="connection refused"):
            rc = K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert rc == 1

    def test_manifest_file_cleanup_on_success(self, launcher_config, recipe_config):
        created_files = []

        original_run = None

        def capture_and_run(cmd, **kwargs):
            manifest_path = cmd[-1]
            created_files.append(manifest_path)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert len(created_files) == 1
        assert not os.path.exists(created_files[0])

    def test_manifest_file_kept_on_failure(self, launcher_config, recipe_config):
        created_files = []

        def capture_and_run(cmd, **kwargs):
            manifest_path = cmd[-1]
            created_files.append(manifest_path)
            return SimpleNamespace(returncode=1, stdout="", stderr="error")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert len(created_files) == 1
        assert os.path.exists(created_files[0])
        os.unlink(created_files[0])

    def test_manifest_content(self, launcher_config, recipe_config):
        manifests = []

        def capture_and_run(cmd, **kwargs):
            manifest_path = cmd[-1]
            with open(manifest_path) as f:
                docs = list(yaml.safe_load_all(f))
            manifests.extend(docs)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert len(manifests) == 2
        cm, job = manifests
        assert cm["kind"] == "ConfigMap"
        assert cm["metadata"]["namespace"] == "test-ns"
        assert job["kind"] == "PyTorchJob"
        assert job["metadata"]["namespace"] == "test-ns"
        specs = job["spec"]["pytorchReplicaSpecs"]
        assert "Master" in specs
        assert "Worker" in specs
        assert specs["Worker"]["replicas"] == 1

    def test_extra_args_propagated(self, launcher_config, recipe_config):
        manifests = []

        def capture_and_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                docs = list(yaml.safe_load_all(f))
            manifests.extend(docs)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
                extra_args=["--lr=0.001"],
            )
        job = manifests[1]
        container = job["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
        assert "--lr=0.001" in container["command"]

    def test_config_mount_path_constant(self):
        assert CONFIG_MOUNT_PATH == "/etc/automodel/config.yaml"

    def test_job_name_contains_timestamp(self, launcher_config, recipe_config):
        job_names = []

        def capture_and_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                docs = list(yaml.safe_load_all(f))
            job_names.append(docs[1]["metadata"]["name"])
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=launcher_config,
            )
        assert len(job_names) == 1
        assert job_names[0].startswith("automodel-")
        timestamp_part = job_names[0].replace("automodel-", "")
        assert timestamp_part.isdigit()

    def test_single_node_no_worker(self, recipe_config):
        single_node_config = {
            "num_nodes": 1,
            "gpus_per_node": 8,
            "image": "nvcr.io/test:latest",
            "namespace": "default",
        }
        manifests = []

        def capture_and_run(cmd, **kwargs):
            with open(cmd[-1]) as f:
                docs = list(yaml.safe_load_all(f))
            manifests.extend(docs)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch(
            "nemo_automodel.components.launcher.k8s.launcher.subprocess.run",
            side_effect=capture_and_run,
        ):
            K8sLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=single_node_config,
            )
        job = manifests[1]
        specs = job["spec"]["pytorchReplicaSpecs"]
        assert "Master" in specs
        assert "Worker" not in specs
