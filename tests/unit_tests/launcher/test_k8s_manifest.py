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

import pytest

from nemo_automodel.components.launcher.k8s.config import K8sConfig
from nemo_automodel.components.launcher.k8s.manifest import (
    render_configmap,
    render_pytorchjob,
)


@pytest.fixture
def default_cfg():
    return K8sConfig()


def test_render_configmap():
    cm = render_configmap("test-cm", "default", "foo: bar\n")
    assert cm["apiVersion"] == "v1"
    assert cm["kind"] == "ConfigMap"
    assert cm["metadata"]["name"] == "test-cm"
    assert cm["metadata"]["namespace"] == "default"
    assert cm["data"]["config.yaml"] == "foo: bar\n"


def test_render_pytorchjob_single_node(default_cfg):
    job = render_pytorchjob(
        job_name="test-job",
        cfg=default_cfg,
        configmap_name="test-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
        extra_args=[],
    )
    assert job["apiVersion"] == "kubeflow.org/v1"
    assert job["kind"] == "PyTorchJob"
    assert job["metadata"]["name"] == "test-job"
    specs = job["spec"]["pytorchReplicaSpecs"]
    assert "Master" in specs
    assert specs["Master"]["replicas"] == 1
    assert "Worker" not in specs


def test_render_pytorchjob_multi_node():
    cfg = K8sConfig(num_nodes=4, gpus_per_node=8)
    job = render_pytorchjob(
        job_name="multi-job",
        cfg=cfg,
        configmap_name="multi-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=["--lr=0.001"],
    )
    specs = job["spec"]["pytorchReplicaSpecs"]
    assert specs["Master"]["replicas"] == 1
    assert specs["Worker"]["replicas"] == 3

    container = specs["Master"]["template"]["spec"]["containers"][0]
    assert container["name"] == "automodel"
    assert "--nproc_per_node=8" in container["command"]
    assert "--nnodes=4" in container["command"]
    assert "--lr=0.001" in container["command"]


def test_render_pytorchjob_pvc_mounts():
    cfg = K8sConfig(
        pvc_mounts=[{"claim": "data-pvc", "mount_path": "/data"}],
    )
    job = render_pytorchjob(
        job_name="pvc-job",
        cfg=cfg,
        configmap_name="pvc-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=[],
    )
    pod_spec = job["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]

    volume_names = [v["name"] for v in pod_spec["volumes"]]
    assert "config-volume" in volume_names
    assert "data-pvc" in volume_names

    mount_paths = [m["mountPath"] for m in pod_spec["containers"][0]["volumeMounts"]]
    assert "/etc/automodel" in mount_paths
    assert "/data" in mount_paths


def test_render_pytorchjob_env_vars():
    cfg = K8sConfig(env_vars={"HF_HOME": "/data/.hf", "WANDB_API_KEY": "secret"})
    job = render_pytorchjob(
        job_name="env-job",
        cfg=cfg,
        configmap_name="env-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=[],
    )
    container = job["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
    env_names = {e["name"] for e in container["env"]}
    assert "HF_HOME" in env_names
    assert "WANDB_API_KEY" in env_names


def test_render_pytorchjob_service_account_and_selectors():
    cfg = K8sConfig(
        service_account="my-sa",
        node_selector={"nvidia.com/gpu.product": "H100"},
        tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    )
    job = render_pytorchjob(
        job_name="sa-job",
        cfg=cfg,
        configmap_name="sa-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=[],
    )
    pod_spec = job["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "my-sa"
    assert pod_spec["nodeSelector"]["nvidia.com/gpu.product"] == "H100"
    assert len(pod_spec["tolerations"]) == 1


# ---------------------------------------------------------------------------
# _container_spec direct tests
# ---------------------------------------------------------------------------
from nemo_automodel.components.launcher.k8s.manifest import (
    _container_spec,
    _volumes,
    _pod_template,
)


def test_container_spec_default_resources():
    cfg = K8sConfig(gpus_per_node=4)
    spec = _container_spec(cfg, "/etc/automodel/config.yaml", "some.Recipe", [])
    assert spec["name"] == "automodel"
    assert spec["image"] == cfg.image
    assert spec["resources"] == {"limits": {"nvidia.com/gpu": "4"}}
    assert "--nproc_per_node=4" in spec["command"]


def test_container_spec_custom_resources():
    custom_res = {"limits": {"nvidia.com/gpu": "2"}, "requests": {"cpu": "4"}}
    cfg = K8sConfig(resources=custom_res)
    spec = _container_spec(cfg, "/etc/automodel/config.yaml", "some.Recipe", [])
    assert spec["resources"] == custom_res


def test_container_spec_env_vars():
    cfg = K8sConfig(env_vars={"MY_VAR": "val1"})
    spec = _container_spec(cfg, "/etc/automodel/config.yaml", "some.Recipe", [])
    env_names = {e["name"]: e["value"] for e in spec["env"]}
    assert "CUDA_DEVICE_MAX_CONNECTIONS" in env_names
    assert "TORCH_NCCL_AVOID_RECORD_STREAMS" in env_names
    assert "NCCL_NVLS_ENABLE" in env_names
    assert env_names["MY_VAR"] == "val1"


def test_container_spec_extra_args():
    cfg = K8sConfig()
    spec = _container_spec(
        cfg, "/etc/automodel/config.yaml", "some.Recipe", ["--lr=0.001", "--wd=0.01"]
    )
    assert "--lr=0.001" in spec["command"]
    assert "--wd=0.01" in spec["command"]


def test_container_spec_volume_mounts_with_pvc():
    cfg = K8sConfig(
        pvc_mounts=[
            {"claim": "data-pvc", "mount_path": "/data"},
            {"claim": "model-pvc", "mount_path": "/models"},
        ]
    )
    spec = _container_spec(cfg, "/etc/automodel/config.yaml", "some.Recipe", [])
    mount_paths = [m["mountPath"] for m in spec["volumeMounts"]]
    assert "/etc/automodel" in mount_paths
    assert "/data" in mount_paths
    assert "/models" in mount_paths


# ---------------------------------------------------------------------------
# _volumes direct tests
# ---------------------------------------------------------------------------
def test_volumes_no_pvcs():
    cfg = K8sConfig()
    vols = _volumes(cfg, "my-configmap")
    assert len(vols) == 1
    assert vols[0]["name"] == "config-volume"
    assert vols[0]["configMap"]["name"] == "my-configmap"


def test_volumes_with_pvcs():
    cfg = K8sConfig(
        pvc_mounts=[{"claim": "data-pvc", "mount_path": "/data"}]
    )
    vols = _volumes(cfg, "my-configmap")
    assert len(vols) == 2
    pvc_vol = vols[1]
    assert pvc_vol["name"] == "data-pvc"
    assert pvc_vol["persistentVolumeClaim"]["claimName"] == "data-pvc"


# ---------------------------------------------------------------------------
# _pod_template direct tests
# ---------------------------------------------------------------------------
def test_pod_template_no_optional_fields():
    cfg = K8sConfig()
    tmpl = _pod_template(cfg, "cm-name", "/etc/automodel/config.yaml", "some.Recipe", [])
    pod_spec = tmpl["spec"]
    assert len(pod_spec["containers"]) == 1
    assert pod_spec["restartPolicy"] == "Never"
    assert "serviceAccountName" not in pod_spec
    assert "nodeSelector" not in pod_spec
    assert "tolerations" not in pod_spec


def test_pod_template_all_optional_fields():
    cfg = K8sConfig(
        service_account="sa",
        node_selector={"gpu": "H100"},
        tolerations=[{"key": "gpu", "operator": "Exists"}],
    )
    tmpl = _pod_template(cfg, "cm-name", "/etc/automodel/config.yaml", "some.Recipe", [])
    pod_spec = tmpl["spec"]
    assert pod_spec["serviceAccountName"] == "sa"
    assert pod_spec["nodeSelector"] == {"gpu": "H100"}
    assert pod_spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]


def test_render_pytorchjob_namespace():
    cfg = K8sConfig(namespace="custom-ns")
    job = render_pytorchjob(
        job_name="ns-job",
        cfg=cfg,
        configmap_name="ns-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=[],
    )
    assert job["metadata"]["namespace"] == "custom-ns"


def test_render_configmap_custom_namespace():
    cm = render_configmap("my-cm", "production", "model: llama\n")
    assert cm["metadata"]["namespace"] == "production"
    assert cm["data"]["config.yaml"] == "model: llama\n"


def test_render_pytorchjob_master_port():
    cfg = K8sConfig(master_port=12345)
    job = render_pytorchjob(
        job_name="port-job",
        cfg=cfg,
        configmap_name="port-cm",
        config_mount_path="/etc/automodel/config.yaml",
        recipe_target="some.Recipe",
        extra_args=[],
    )
    container = job["spec"]["pytorchReplicaSpecs"]["Master"]["template"]["spec"]["containers"][0]
    assert any("12345" in str(c) for c in container["command"])
