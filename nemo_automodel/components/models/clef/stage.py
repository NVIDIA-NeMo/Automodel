# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Model-owned metadata contract shared by native CLEF stages and release export."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from nemo_automodel.components.models.clef.joint_schema_head import ClefHeadConfig


@dataclass(frozen=True)
class ClefStageManifest:
    """Define head architecture, state namespace, and adapter presence for native stages."""

    backbone_adapter: bool
    head_config: ClefHeadConfig
    stage: str
    format: str = "clef-native-stage-v1"
    HEAD_STATE_NAME: ClassVar[str] = "head"

    def __post_init__(self) -> None:
        if self.format != "clef-native-stage-v1":
            raise ValueError("a published native CLEF stage checkpoint is required")

    def write(self, path: Path) -> None:
        """Serialize stage metadata after the checkpoint lifecycle reserves its directory."""
        (path / "stage.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ClefStageManifest":
        """Read and validate the model-owned native stage metadata."""
        data = json.loads((path / "stage.json").read_text())
        return cls(
            backbone_adapter=data["backbone_adapter"],
            head_config=ClefHeadConfig(**data["head_config"]),
            stage=data["stage"],
            format=data.get("format", ""),
        )
