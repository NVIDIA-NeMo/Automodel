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

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_automodel.components.launcher.base import Launcher
from nemo_automodel.components.launcher.k8s.config import K8sConfig
from nemo_automodel.components.launcher.k8s.manifest import (
    render_configmap,
    render_pytorchjob,
)

logger = logging.getLogger(__name__)

CONFIG_MOUNT_PATH = "/etc/automodel/config.yaml"


class K8sLauncher(Launcher):
    """Submit a recipe as a Kubeflow PyTorchJob on Kubernetes."""

    def launch(
        self,
        config: Dict[str, Any],
        config_path: Path,
        recipe_target: str,
        launcher_config: Dict[str, Any],
        extra_args: Optional[List[str]] = None,
    ) -> int:
        extra_args = extra_args or []
        cfg = K8sConfig(**launcher_config)
        timestamp = str(int(time.time()))
        job_name = f"automodel-{timestamp}"
        configmap_name = f"{job_name}-config"

        config_yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)

        cm_manifest = render_configmap(configmap_name, cfg.namespace, config_yaml_str)
        job_manifest = render_pytorchjob(
            job_name=job_name,
            cfg=cfg,
            configmap_name=configmap_name,
            config_mount_path=CONFIG_MOUNT_PATH,
            recipe_target=recipe_target,
            extra_args=extra_args,
        )

        combined = [cm_manifest, job_manifest]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"automodel-k8s-{timestamp}-",
            delete=False,
        ) as f:
            yaml.dump_all(combined, f, default_flow_style=False, sort_keys=False)
            manifest_path = f.name

        logger.info("Generated k8s manifest -> %s", manifest_path)
        logger.info("Submitting PyTorchJob '%s' in namespace '%s'", job_name, cfg.namespace)

        proc = subprocess.run(
            ["kubectl", "apply", "-f", manifest_path],
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            logger.info(proc.stdout.strip())
        if proc.returncode != 0:
            logger.error("kubectl apply failed:\n%s", proc.stderr)
        else:
            logger.info(
                "Job submitted. Monitor with:\n"
                "  kubectl -n %s get pytorchjobs %s\n"
                "  kubectl -n %s logs -f -l training.kubeflow.org/job-name=%s",
                cfg.namespace,
                job_name,
                cfg.namespace,
                job_name,
            )

        # Clean up temp file on success; keep on failure for debugging
        if proc.returncode == 0:
            os.unlink(manifest_path)

        return proc.returncode
