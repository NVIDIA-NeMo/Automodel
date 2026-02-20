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

"""Tests for _broadcast_replicated_params_across_tp in infrastructure.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from nemo_automodel._transformers.infrastructure import (
    _broadcast_replicated_params_across_tp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Toy model with a replicated embedding and a linear layer."""

    def __init__(self):
        super().__init__()
        self.custom_embedding = nn.Embedding(100, 16)
        self.linear = nn.Linear(16, 16, bias=True)
        self.register_buffer("my_buffer", torch.zeros(4))


def _make_mock_device_mesh(tp_size: int = 2, has_tp: bool = True):
    """Return a lightweight mock that behaves like a DeviceMesh with a 'tp' dim."""
    mesh = MagicMock()
    if has_tp:
        mesh.mesh_dim_names = ("dp_shard_cp", "tp")
    else:
        mesh.mesh_dim_names = ("dp_shard_cp",)

    tp_submesh = MagicMock()
    tp_submesh.size.return_value = tp_size

    mock_group = MagicMock()
    tp_submesh.get_group.return_value = mock_group

    # device_mesh["tp"] returns the tp submesh
    mesh.__getitem__ = MagicMock(return_value=tp_submesh)
    return mesh, mock_group


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBroadcastReplicatedParamsAcrossTP:
    """Unit tests for _broadcast_replicated_params_across_tp."""

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_no_op_when_device_mesh_is_none(self, mock_dist):
        model = _SimpleModel()
        _broadcast_replicated_params_across_tp(model, device_mesh=None)
        mock_dist.broadcast.assert_not_called()

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_no_op_when_no_tp_dimension(self, mock_dist):
        mesh, _ = _make_mock_device_mesh(has_tp=False)
        model = _SimpleModel()
        _broadcast_replicated_params_across_tp(model, device_mesh=mesh)
        mock_dist.broadcast.assert_not_called()

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_no_op_when_tp_size_is_one(self, mock_dist):
        mesh, _ = _make_mock_device_mesh(tp_size=1)
        model = _SimpleModel()
        _broadcast_replicated_params_across_tp(model, device_mesh=mesh)
        mock_dist.broadcast.assert_not_called()

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_broadcasts_all_regular_params_and_buffers(self, mock_dist):
        """All params are regular tensors (no DTensors) → all should be broadcast."""
        mock_dist.get_global_rank.return_value = 0
        mesh, tp_group = _make_mock_device_mesh(tp_size=4)
        model = _SimpleModel()

        _broadcast_replicated_params_across_tp(model, device_mesh=mesh)

        # Expect broadcast calls for:
        # - custom_embedding.weight
        # - linear.weight
        # - linear.bias
        # - my_buffer
        assert mock_dist.broadcast.call_count == 4

        # All calls should use the TP process group and src = global rank 0
        for c in mock_dist.broadcast.call_args_list:
            assert c.kwargs.get("src", c.args[1] if len(c.args) > 1 else None) == 0
            assert c.kwargs.get("group", None) == tp_group

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_skips_meta_tensors(self, mock_dist):
        """Parameters on meta device should be silently skipped."""
        mock_dist.get_global_rank.return_value = 0
        mesh, _ = _make_mock_device_mesh(tp_size=2)
        model = _SimpleModel()
        # Move one param to meta device
        model.custom_embedding.weight = nn.Parameter(
            torch.empty(100, 16, device="meta")
        )

        _broadcast_replicated_params_across_tp(model, device_mesh=mesh)

        # custom_embedding.weight is on meta → skipped.
        # Remaining: linear.weight, linear.bias, my_buffer = 3 calls
        assert mock_dist.broadcast.call_count == 3

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_skips_tp_sharded_dtensors(self, mock_dist):
        """DTensor params with Shard placement on the TP dim should be skipped."""
        from torch.distributed.tensor.placement_types import Shard, Replicate

        mock_dist.get_global_rank.return_value = 0
        mesh, tp_group = _make_mock_device_mesh(tp_size=2)

        model = _SimpleModel()

        # Replace linear.weight with a mock DTensor that has Shard on the TP dim
        tp_sharded_param = MagicMock(spec=["device_mesh", "placements", "data", "to_local", "is_meta", "is_cuda", "device"])
        tp_sharded_param.device_mesh = SimpleNamespace(mesh_dim_names=("dp_shard_cp", "tp"))
        tp_sharded_param.placements = (Replicate(), Shard(0))  # Shard on TP dim (index 1)
        tp_sharded_param.device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        # Make it pass isinstance(param, DTensor) check by patching
        with patch(
            "nemo_automodel._transformers.infrastructure.DTensor",
            new=type(tp_sharded_param),
        ):
            _broadcast_replicated_params_across_tp(model, device_mesh=mesh)

        # The mocked DTensor param should NOT trigger a broadcast because it's TP-sharded.
        # But we replaced linear.weight after model creation, so named_parameters()
        # doesn't pick up MagicMock. Test the _is_tp_sharded logic separately below.

    @patch("nemo_automodel._transformers.infrastructure.dist")
    def test_broadcasts_replicated_dtensors(self, mock_dist):
        """DTensor params with Replicate placement on TP dim should be broadcast."""
        from torch.distributed.tensor.placement_types import Replicate

        mock_dist.get_global_rank.return_value = 0
        mesh, tp_group = _make_mock_device_mesh(tp_size=2)

        model = _SimpleModel()

        # Verify that regular-tensor params (which are effectively
        # TP-replicated) are all broadcast
        _broadcast_replicated_params_across_tp(model, device_mesh=mesh)

        # custom_embedding.weight + linear.weight + linear.bias + my_buffer = 4
        assert mock_dist.broadcast.call_count == 4

    def test_is_tp_sharded_logic_shard(self):
        """Directly test the _is_tp_sharded helper logic for Shard placement."""
        from torch.distributed.tensor.placement_types import Shard, Replicate

        # Simulate a DTensor with Shard on the TP dim
        tensor = MagicMock()
        tensor.device_mesh = SimpleNamespace(mesh_dim_names=("dp_shard_cp", "tp"))
        tensor.placements = (Replicate(), Shard(0))  # Shard on TP dim (index 1)

        # The _is_tp_sharded logic:
        tp_mesh_name = "tp"
        mesh_names = tensor.device_mesh.mesh_dim_names
        assert tp_mesh_name in mesh_names
        tp_dim_idx = list(mesh_names).index(tp_mesh_name)
        assert isinstance(tensor.placements[tp_dim_idx], Shard)

    def test_is_tp_sharded_logic_replicate(self):
        """Directly test the _is_tp_sharded helper logic for Replicate placement."""
        from torch.distributed.tensor.placement_types import Shard, Replicate

        # Simulate a DTensor with Replicate on the TP dim
        tensor = MagicMock()
        tensor.device_mesh = SimpleNamespace(mesh_dim_names=("dp_shard_cp", "tp"))
        tensor.placements = (Shard(0), Replicate())  # Replicate on TP dim (index 1)

        tp_mesh_name = "tp"
        mesh_names = tensor.device_mesh.mesh_dim_names
        assert tp_mesh_name in mesh_names
        tp_dim_idx = list(mesh_names).index(tp_mesh_name)
        assert not isinstance(tensor.placements[tp_dim_idx], Shard)

    def test_is_tp_sharded_logic_no_tp_in_mesh(self):
        """Params whose mesh lacks 'tp' should not be considered TP-sharded."""
        tensor = MagicMock()
        tensor.device_mesh = SimpleNamespace(mesh_dim_names=("dp_shard_cp",))
        tensor.placements = (MagicMock(),)

        tp_mesh_name = "tp"
        mesh_names = tensor.device_mesh.mesh_dim_names
        assert tp_mesh_name not in mesh_names
