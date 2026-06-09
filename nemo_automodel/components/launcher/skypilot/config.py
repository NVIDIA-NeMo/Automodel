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

from __future__ import annotations

import os
from dataclasses import dataclass, field

SUPPORTED_CLOUDS = ("aws", "gcp", "azure", "lambda", "kubernetes")


@dataclass
class SkyPilotConfig:
    """Configuration for launching AutoModel jobs with SkyPilot."""

    # Required: cloud provider
    cloud: str = field(metadata=dict(help=f"Cloud provider. One of: {SUPPORTED_CLOUDS}"))

    # Compute resources
    accelerators: str = field(default="T4:1", metadata=dict(help="GPU type and count per node, e.g. 'T4:1', 'A100:8'"))
    num_nodes: int = field(default=1, metadata=dict(help="Number of nodes for distributed training"))
    use_spot: bool = field(default=True, metadata=dict(help="Use spot/preemptible instances for cost savings"))
    disk_size: int = field(default=100, metadata=dict(help="Disk size in GB"))
    instance_type: str | None = field(
        default=None, metadata=dict(help="Specific cloud instance type; auto-selected if None")
    )

    # Cloud location
    region: str | None = field(default=None, metadata=dict(help="Cloud region"))
    zone: str | None = field(default=None, metadata=dict(help="Availability zone within the region"))

    # Job identity
    job_name: str = field(default="", metadata=dict(help="Job and SkyPilot cluster name"))

    # Remote environment
    setup: str = field(default="", metadata=dict(help="Shell commands run on the remote VM before training starts"))
    hf_home: str = field(
        default="~/.cache/huggingface",
        metadata=dict(help="HuggingFace cache directory on the remote VM"),
    )

    # Credentials (sourced from env by default, never hard-coded)
    hf_token: str = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", ""),
        metadata=dict(help="HuggingFace token for gated model access"),
    )
    wandb_key: str = field(
        default_factory=lambda: os.environ.get("WANDB_API_KEY", ""),
        metadata=dict(help="Weights & Biases API key"),
    )
    env_vars: dict[str, str] = field(
        default_factory=dict,
        metadata=dict(help="Additional environment variables to set on the remote VM"),
    )

    # Training command (set programmatically by the launcher, not exposed in YAML)
    command: str = field(default="", metadata=dict(help="Training command executed on the remote VM"))

    def __post_init__(self) -> None:
        if self.cloud.lower() not in SUPPORTED_CLOUDS:
            raise ValueError(f"'cloud' must be one of {SUPPORTED_CLOUDS}, got: {self.cloud!r}")
        if self.num_nodes < 1:
            raise ValueError(f"'num_nodes' must be >= 1, got: {self.num_nodes}")
        if self.disk_size < 1:
            raise ValueError(f"'disk_size' must be >= 1 GB, got: {self.disk_size}")
