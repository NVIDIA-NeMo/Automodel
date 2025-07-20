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
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VolumeMapping:
    """Host-to-container mount specification."""

    source: Path = field(metadata={"help": "Absolute host path to mount"})
    dest: Path = field(metadata={"help": "Absolute container path"})

    def __post_init__(self):
        assert isinstance(self.source, Path)
        assert isinstance(self.dest, Path)
        if not self.source.exists():
            raise ValueError("Expected source to exist")
        if not self.source.is_absolute():
            raise ValueError(f"'source' must be absolute: {self.source}")
        if not self.dest.is_absolute():
            raise ValueError(f"'dest' must be absolute: {self.dest}")

    def to_str(self):
        return f"{self.source}:{self.dest}"


@dataclass
class SlurmConfig:
    # Slurm basics
    job_name: str = field(metadata=dict(help="Job name for Slurm (synonym: -J)"))
    nodes: int = field(default=1, metadata=dict(help="Number of nodes (synonym: -N)"))
    ntasks_per_node: int = field(default=8, metadata=dict(help="ntasks per node (synonym: --ntasks)"))
    time: str = field(default="00:05:00", metadata=dict(help="Wall-clock time limit. Default value: 00:05:00."))
    account: str = field(default=None, metadata=dict(help="Slurm account (-A)"))
    partition: str = field(default="batch", metadata=dict(help="Partition/queue (-p)"))

    # Container / mounts
    container_image: str = field(default="nvcr.io/nvidia/nemo:dev", metadata=dict(help="SquashFS / OCI image path"))
    nemo_mount: VolumeMapping = field(default=None, metadata=dict(help="Host directory to mount inside container"))
    hf_home: Path = field(default="~/.cache/huggingface", metadata=dict(help="Host HF cache directory"))
    extra_mounts: VolumeMapping = field(
        default=None, metadata=dict(help="Additional mounts host:container (comma-separated)")
    )

    # Misc env / training specifics
    master_port: int = field(default=13742, metadata=dict(help="Port for multinode"))
    gpus_per_node: int = field(default=8, metadata=dict(help="GPUs per node"))
    wandb_key: str = field(default=os.environ.get("WANDB_API_KEY", ""), metadata=dict(help="W&B key or env reference"))
    hf_token: str = field(
        default=os.environ.get("HF_TOKEN", ""),
        metadata=dict(help="HF-TOKEN key to use for retrieving gated assets from HuggingFace Hub."),
    )
    # User command
    command: str = field(default="", metadata=dict(help="Shell command(s) to run inside container"))
    chdir: str = field(default=None, metadata=dict(help="Working directory of the job"))

    def __post_init__(self):
        if isinstance(self.extra_mounts, list):
            for i, item in enumerate(self.extra_mounts):
                if isinstance(item, VolumeMapping):
                    continue
                elif isinstance(item, str):
                    parts = item.split(":")
                    assert len(parts) == 2, "Expected volume mapping to have format <src>:<dst>"
                    self.extra_mounts[i] = VolumeMapping(Path(parts[0]), Path(parts[1]))
                else:
                    raise ValueError("Expect mount to be VolumeMapping or str")
