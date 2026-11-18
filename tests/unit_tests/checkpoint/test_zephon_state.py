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

"""Opaque loader state through the ordinary DCP state-dict interface."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader

pytestmark = pytest.mark.cpu_only


def test_dcp_restores_opaque_loader_bytes_without_starting_destination_runtime(tmp_path: Path) -> None:
    expected = b"opaque-zephon-checkpoint"
    source_pipeline = MagicMock(spec=["__iter__", "checkpoint", "restore", "close"])
    source_pipeline.__iter__.return_value = iter(())
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
        assert list(source) == []
        checkpointer.save_distributed_state(source, "loader", str(tmp_path))
        checkpointer.load_distributed_state(destination, "loader", str(tmp_path))
        source_pipeline.checkpoint.assert_called_once_with(as_bytes=True)
        destination_pipeline.checkpoint.assert_not_called()
        destination_pipeline.restore.assert_called_once_with(expected)
        # Saving a restored, unstarted loader must preserve its saved progress.
        checkpointer.save_distributed_state(destination, "resaved", str(tmp_path))
        destination_pipeline.checkpoint.assert_not_called()
        loaded = {"zephon": b""}
        torch.distributed.checkpoint.load(loaded, checkpoint_id=tmp_path / "resaved")
        assert loaded == {"zephon": expected}
    finally:
        source.close()
        destination.close()
        checkpointer.close()


def test_dcp_round_trips_initial_loader_state_without_starting_pipeline(tmp_path: Path) -> None:
    pipeline = Mock(spec=["checkpoint", "restore", "close"])
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    checkpointer = CheckpointingConfig(enabled=False, save_consolidated=False).build(0, 0, 0)
    try:
        checkpointer.save_distributed_state(loader, "initial", str(tmp_path))
        checkpointer.load_distributed_state(loader, "initial", str(tmp_path))
        assert loader.state_dict() == {"zephon": b""}
        assert pipeline.mock_calls == []
    finally:
        loader.close()
        checkpointer.close()


def test_dcp_load_uses_state_dict_for_ordinary_tensor_state(tmp_path: Path) -> None:
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


def _checkpoint_worker(rank: int, root: str) -> None:
    """Exercise real DCP on two ranks with a collective checkpoint producer."""
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )

    def checkpoint(*, as_bytes: bool) -> bytes:
        assert as_bytes
        dist.barrier()
        return b"aggregated-progress"

    source_pipeline = MagicMock(spec=["__iter__", "checkpoint", "restore", "close"])
    source_pipeline.__iter__.return_value = iter(())
    source_pipeline.checkpoint.side_effect = checkpoint
    destination_pipeline = Mock(spec=["checkpoint", "restore", "close"])
    destination_pipeline.checkpoint.side_effect = AssertionError("load initialized a fresh pipeline")
    source = ZephonDataLoader(
        pipeline=source_pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    destination = ZephonDataLoader(
        pipeline=destination_pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    checkpointer = CheckpointingConfig(enabled=False, save_consolidated=False).build(
        0, rank, 0, process_group=dist.group.WORLD
    )
    try:
        assert list(source) == []
        checkpointer.save_distributed_state(source, "loader", root)
        checkpointer.load_distributed_state(destination, "loader", root)
        checkpointer.save_distributed_state(destination, "resaved", root)
        source_pipeline.checkpoint.assert_called_once_with(as_bytes=True)
        destination_pipeline.checkpoint.assert_not_called()
        destination_pipeline.restore.assert_called_once_with(b"aggregated-progress")
        restored = {"zephon": b""}
        torch.distributed.checkpoint.load(restored, checkpoint_id=Path(root) / "resaved")
        assert restored == {"zephon": b"aggregated-progress"}
    finally:
        source.close()
        destination.close()
        checkpointer.close()
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=90,
    reason="Two CPU ranks exercise DCP save, restore and resave with collective snapshot collection.",
)
def test_every_rank_checkpoints_without_snapshotting_restore_destination(tmp_path: Path) -> None:
    mp.spawn(_checkpoint_worker, args=(str(tmp_path),), nprocs=2, join=True)
