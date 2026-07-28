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

from pathlib import Path

import pytest
import torch

from nemo_automodel.components.checkpoint.checkpointing import safe_torch_load


def _touch_marker(marker_path: str) -> None:
    Path(marker_path).touch()


class _TouchPayload:
    def __init__(self, marker_path: str):
        self.marker_path = marker_path

    def __reduce__(self):
        return (_touch_marker, (self.marker_path,))


def test_safe_torch_load_rejects_pickle_payload_without_execution(tmp_path):
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "payload.pt"
    torch.save(_TouchPayload(str(marker)), checkpoint)

    with pytest.raises(RuntimeError, match="Refusing to load"):
        safe_torch_load(checkpoint)

    assert not marker.exists()


def test_safe_torch_load_allows_tensor_only_checkpoint(tmp_path):
    checkpoint = tmp_path / "state.pt"
    torch.save({"weight": torch.arange(3)}, checkpoint)

    loaded = safe_torch_load(checkpoint)

    torch.testing.assert_close(loaded["weight"], torch.arange(3))
