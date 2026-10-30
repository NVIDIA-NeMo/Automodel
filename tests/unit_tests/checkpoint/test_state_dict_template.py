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

"""Actual local DCP load destinations without live runtime snapshots."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader


def test_dcp_restores_opaque_loader_bytes_without_starting_destination_runtime(tmp_path):
    expected = b"opaque-zephon-checkpoint"
    source_pipeline = Mock(spec=["checkpoint", "restore", "close"])
    source_pipeline.checkpoint.return_value = expected
    destination_pipeline = Mock(spec=["checkpoint", "restore", "close"])
    destination_pipeline.checkpoint.side_effect = AssertionError("load initialized a fresh pipeline")
    source = ZephonDataLoader(
        pipeline=source_pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    destination = ZephonDataLoader(
        pipeline=destination_pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    checkpointer = CheckpointingConfig(enabled=False, save_consolidated=False).build(0, 0, 0)
    try:
        checkpointer.save_distributed_state(source, "loader", str(tmp_path))
        checkpointer.load_distributed_state(destination, "loader", str(tmp_path))
        source_pipeline.checkpoint.assert_called_once_with(as_bytes=True)
        destination_pipeline.checkpoint.assert_not_called()
        destination_pipeline.restore.assert_called_once_with(expected)
    finally:
        source.close()
        destination.close()
        checkpointer.close()


def test_dcp_load_falls_back_to_state_dict_for_ordinary_tensor_state(tmp_path):
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = SimpleNamespace(state_dict=lambda: {"weight": original})
    snapshot = Mock(return_value={"weight": torch.zeros_like(original)})
    restored = Mock()
    destination = SimpleNamespace(state_dict=snapshot, load_state_dict=restored)
    checkpointer = CheckpointingConfig(enabled=False, save_consolidated=False).build(0, 0, 0)
    try:
        checkpointer.save_distributed_state(source, "tensor", str(tmp_path))
        checkpointer.load_distributed_state(destination, "tensor", str(tmp_path))
        snapshot.assert_called_once()
        restored.assert_called_once()
        torch.testing.assert_close(restored.call_args.args[0]["weight"], original)
    finally:
        checkpointer.close()
