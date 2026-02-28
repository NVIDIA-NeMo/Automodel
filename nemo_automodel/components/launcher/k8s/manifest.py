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

"""Render a Kubeflow PyTorchJob manifest from a :class:`K8sConfig`."""

from typing import Any, Dict, List

from nemo_automodel.components.launcher.k8s.config import K8sConfig


def _container_spec(
    cfg: K8sConfig,
    config_mount_path: str,
    recipe_target: str,
    extra_args: List[str],
) -> Dict[str, Any]:
    command = [
        "torchrun",
        f"--nproc_per_node={cfg.gpus_per_node}",
        f"--nnodes={cfg.num_nodes}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=$(MASTER_ADDR):{cfg.master_port}",
        "-m",
        "cli.app",
        config_mount_path,
    ]
    command.extend(extra_args)

    env = [
        {"name": "CUDA_DEVICE_MAX_CONNECTIONS", "value": "1"},
        {"name": "TORCH_NCCL_AVOID_RECORD_STREAMS", "value": "1"},
        {"name": "NCCL_NVLS_ENABLE", "value": "0"},
    ]
    for k, v in cfg.env_vars.items():
        env.append({"name": k, "value": str(v)})

    resources = cfg.resources or {
        "limits": {"nvidia.com/gpu": str(cfg.gpus_per_node)},
    }

    volume_mounts = [
        {"name": "config-volume", "mountPath": "/etc/automodel"},
    ]
    for pvc in cfg.pvc_mounts:
        volume_mounts.append(
            {"name": pvc["claim"], "mountPath": pvc["mount_path"]},
        )

    return {
        "name": "automodel",
        "image": cfg.image,
        "command": command,
        "env": env,
        "resources": resources,
        "volumeMounts": volume_mounts,
    }


def _volumes(cfg: K8sConfig, configmap_name: str) -> List[Dict[str, Any]]:
    volumes = [
        {
            "name": "config-volume",
            "configMap": {"name": configmap_name},
        },
    ]
    for pvc in cfg.pvc_mounts:
        volumes.append(
            {
                "name": pvc["claim"],
                "persistentVolumeClaim": {"claimName": pvc["claim"]},
            },
        )
    return volumes


def _pod_template(
    cfg: K8sConfig,
    configmap_name: str,
    config_mount_path: str,
    recipe_target: str,
    extra_args: List[str],
) -> Dict[str, Any]:
    template: Dict[str, Any] = {
        "spec": {
            "containers": [
                _container_spec(cfg, config_mount_path, recipe_target, extra_args),
            ],
            "volumes": _volumes(cfg, configmap_name),
            "restartPolicy": "Never",
        },
    }
    if cfg.service_account:
        template["spec"]["serviceAccountName"] = cfg.service_account
    if cfg.node_selector:
        template["spec"]["nodeSelector"] = cfg.node_selector
    if cfg.tolerations:
        template["spec"]["tolerations"] = cfg.tolerations
    return template


def render_pytorchjob(
    job_name: str,
    cfg: K8sConfig,
    configmap_name: str,
    config_mount_path: str,
    recipe_target: str,
    extra_args: List[str],
) -> Dict[str, Any]:
    """Return a dict representing a Kubeflow PyTorchJob manifest."""
    pod = _pod_template(cfg, configmap_name, config_mount_path, recipe_target, extra_args)

    manifest: Dict[str, Any] = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {
            "name": job_name,
            "namespace": cfg.namespace,
        },
        "spec": {
            "pytorchReplicaSpecs": {
                "Master": {
                    "replicas": 1,
                    "template": pod,
                },
            },
        },
    }
    if cfg.num_nodes > 1:
        manifest["spec"]["pytorchReplicaSpecs"]["Worker"] = {
            "replicas": cfg.num_nodes - 1,
            "template": pod,
        }
    return manifest


def render_configmap(
    name: str,
    namespace: str,
    config_yaml_str: str,
) -> Dict[str, Any]:
    """Return a dict representing a ConfigMap holding the job YAML."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "data": {
            "config.yaml": config_yaml_str,
        },
    }
