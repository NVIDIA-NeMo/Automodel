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


"""Opt-in checkpoint metadata transport and process-group ownership."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.checkpoint.config import CheckpointingConfig


def test_load_metadata_transport_defaults_to_existing_group() -> None:
    assert CheckpointingConfig(save_consolidated=False).load_metadata_with_gloo is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_load_metadata_transport_requires_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="load_metadata_with_gloo must be a boolean"):
        CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=value)


@pytest.mark.parametrize("initialized,world_size,backend", [(False, 1, "nccl"), (True, 1, "nccl"), (True, 2, "gloo")])
def test_no_extra_group_without_multi_rank_nccl(initialized: bool, world_size: int, backend: str) -> None:
    original = object()
    with (
        patch("torch.distributed.is_initialized", return_value=initialized),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_backend", return_value=backend),
        patch("torch.distributed.new_group") as create,
        patch("torch.distributed.destroy_process_group") as destroy,
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        assert checkpointer.process_group is original
        assert checkpointer._get_load_process_group() is original
        assert checkpointer._load_process_group is None
        checkpointer.close()
    create.assert_not_called()
    destroy.assert_not_called()


def test_owned_load_group_preserves_membership_and_closes_once() -> None:
    original, owned = object(), object()
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.get_process_group_ranks", return_value=[2, 3]),
        patch("torch.distributed.new_group", return_value=owned) as create,
        patch("torch.distributed.destroy_process_group") as destroy,
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        assert checkpointer.process_group is original
        create.assert_not_called()
        assert checkpointer._get_load_process_group() is owned
        assert checkpointer._get_load_process_group() is owned
        assert checkpointer._load_process_group is owned
        create.assert_called_once_with(ranks=[2, 3], backend="gloo", use_local_synchronization=True)
        checkpointer.close()
        checkpointer.close()
        assert checkpointer._load_process_group is None
    destroy.assert_called_once_with(owned)


def test_default_does_not_create_metadata_group() -> None:
    original = object()
    with patch("torch.distributed.new_group") as create:
        checkpointer = CheckpointingConfig(save_consolidated=False).build(0, 0, 0, process_group=original)
        assert checkpointer._get_load_process_group() is original
    create.assert_not_called()


def test_load_uses_cpu_group_and_save_keeps_callers_group() -> None:
    original, owned = object(), object()
    checkpoint_module = "nemo_automodel.components.checkpoint.checkpointing"
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.get_backend", return_value="nccl"),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.distributed.new_group", return_value=owned),
        patch("torch.distributed.destroy_process_group") as destroy,
        patch(f"{checkpoint_module}.dcp.load") as load,
        patch(f"{checkpoint_module}.dcp.save") as save,
        patch(f"{checkpoint_module}._ensure_shared_dirs"),
    ):
        checkpointer = CheckpointingConfig(save_consolidated=False, load_metadata_with_gloo=True).build(
            0, 0, 0, process_group=original
        )
        tensors = {"weight": torch.zeros(2, 3)}
        state = SimpleNamespace(state_dict=lambda: tensors, load_state_dict=Mock())
        try:
            checkpointer._do_load(tensors, "/checkpoint/model")
            assert load.call_args.kwargs["process_group"] is owned
            checkpointer.load_distributed_state(state, "auxiliary", "/checkpoint")
            assert load.call_args.kwargs["process_group"] is owned
            state.load_state_dict.assert_called_once_with(tensors)
            checkpointer.save_distributed_state(state, "auxiliary", "/checkpoint")
            assert save.call_args.kwargs["process_group"] is original
        finally:
            checkpointer.close()
    destroy.assert_called_once_with(owned)
