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

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class K8sConfig:
    """Configuration for Kubernetes (Kubeflow PyTorchJob) launcher."""

    num_nodes: int = 1
    gpus_per_node: int = 8
    image: str = "nvcr.io/nvidia/nemo-automodel:latest"
    namespace: str = "default"
    service_account: str = ""
    pvc_mounts: List[Dict[str, str]] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Dict[str, str]] = field(default_factory=dict)
    tolerations: List[Dict[str, str]] = field(default_factory=list)
    node_selector: Dict[str, str] = field(default_factory=dict)
    master_port: int = 29500
