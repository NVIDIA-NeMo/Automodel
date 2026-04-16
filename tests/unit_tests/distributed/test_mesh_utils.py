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

"""Tests for mesh_utils: get_flat_mesh, get_submesh, _unflatten_compat utilities."""

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.distributed.mesh_utils import (
    _unflatten_compat,
    get_flat_mesh,
    get_submesh,
)

# ---------------------------------------------------------------------------
# get_flat_mesh
# ---------------------------------------------------------------------------


class TestGetFlatMesh:
    def _make_mock_mesh(self, dim_names, flatten_mapping=None):
        """Create a mock DeviceMesh with given dim names and optional flatten mapping."""
        mesh = Mock()
        mesh.mesh_dim_names = dim_names

        # _get_root_mesh returns itself by default (root mesh)
        root = Mock()
        root._flatten_mapping = flatten_mapping or {}
        mesh._get_root_mesh = Mock(return_value=root)

        # __getitem__ returns a mock submesh
        def getitem(name):
            submesh = Mock()
            submesh._name = name
            return submesh

        mesh.__getitem__ = Mock(side_effect=getitem)
        return mesh

    def test_mesh_dim_returns_direct_slice(self):
        mesh = self._make_mock_mesh(("dp", "tp"))
        get_flat_mesh(mesh, "tp")
        mesh.__getitem__.assert_called_once_with("tp")
        # Should NOT call _get_root_mesh for direct mesh dims
        mesh._get_root_mesh.assert_not_called()

    def test_flattened_dim_returns_from_mapping(self):
        dp_flat = Mock()
        dp_flat.size = Mock(return_value=8)
        mesh = self._make_mock_mesh(
            ("dp_replicate", "dp_shard", "cp", "tp"),
            flatten_mapping={"dp": dp_flat, "dp_cp": Mock()},
        )
        result = get_flat_mesh(mesh, "dp")
        assert result is dp_flat
        # Should NOT go through __getitem__
        mesh.__getitem__.assert_not_called()

    def test_unknown_dim_raises_key_error(self):
        mesh = self._make_mock_mesh(("dp", "tp"), flatten_mapping={})
        with pytest.raises(KeyError, match="unknown"):
            get_flat_mesh(mesh, "unknown")

    def test_flattened_dim_checked_on_root_not_self(self):
        """When mesh is a submesh, flattened dims are looked up on the root."""
        dp_flat = Mock()
        submesh = Mock()
        submesh.mesh_dim_names = ("tp",)  # submesh only has "tp"

        root = Mock()
        root._flatten_mapping = {"dp": dp_flat}
        submesh._get_root_mesh = Mock(return_value=root)
        submesh.__getitem__ = Mock()

        result = get_flat_mesh(submesh, "dp")
        assert result is dp_flat
        submesh._get_root_mesh.assert_called_once()


# ---------------------------------------------------------------------------
# get_submesh
# ---------------------------------------------------------------------------


class TestGetSubmesh:
    def _make_mock_mesh(self, dim_names, flatten_mapping=None):
        mesh = Mock()
        mesh.mesh_dim_names = dim_names

        root = Mock()
        root._flatten_mapping = flatten_mapping or {}
        mesh._get_root_mesh = Mock(return_value=root)

        def getitem(names):
            submesh = Mock()
            submesh._names = names
            return submesh

        mesh.__getitem__ = Mock(side_effect=getitem)
        return mesh

    def test_single_physical_dim_delegates_to_get_flat_mesh(self):
        mesh = self._make_mock_mesh(("dp", "tp"))
        get_submesh(mesh, ("tp",))
        mesh.__getitem__.assert_called_once_with("tp")

    def test_single_flattened_dim_delegates_to_get_flat_mesh(self):
        dp_flat = Mock()
        mesh = self._make_mock_mesh(
            ("dp_replicate", "dp_shard"),
            flatten_mapping={"dp": dp_flat},
        )
        result = get_submesh(mesh, ("dp",))
        assert result is dp_flat

    def test_multi_dim_names_direct_slice(self):
        mesh = self._make_mock_mesh(("pp", "dp_replicate", "dp_shard", "cp", "tp"))
        get_submesh(mesh, ("dp_replicate", "dp_shard"))
        mesh.__getitem__.assert_called_once_with(("dp_replicate", "dp_shard"))

    def test_mixed_physical_flattened_uses_unflatten(self, monkeypatch):
        """Mixed physical + flattened tuple constructs submesh via _unflatten from parent."""
        # Shared group sentinel — validation checks groups match
        group_sentinel = Mock()

        dp_shard_cp_flat = Mock()
        dp_shard_cp_flat.size = Mock(return_value=4)
        dp_shard_cp_flat.get_group = Mock(return_value=group_sentinel)

        # dp_cp is the parent: dp_replicate(2) * dp_shard_cp(4) = dp_cp(8)
        dp_cp_flat = Mock()
        dp_cp_flat.size = Mock(return_value=8)

        # The unflatten result — both dims return matching groups
        unflatten_result = Mock()
        unflatten_result.__getitem__ = Mock(return_value=Mock(get_group=Mock(return_value=group_sentinel)))
        dp_cp_flat._unflatten = Mock(return_value=unflatten_result)

        # dp_replicate is a mesh dim with size 2
        dp_rep_submesh = Mock()
        dp_rep_submesh.size = Mock(return_value=2)
        dp_rep_submesh.get_group = Mock(return_value=group_sentinel)

        mesh = Mock()
        mesh.mesh_dim_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
        root = Mock()
        root._flatten_mapping = {"dp_shard_cp": dp_shard_cp_flat, "dp_cp": dp_cp_flat}
        mesh._get_root_mesh = Mock(return_value=root)
        mesh.__getitem__ = Mock(return_value=dp_rep_submesh)

        monkeypatch.setattr(
            "torch.distributed.get_process_group_ranks",
            lambda g: [0, 4],
        )

        result = get_submesh(mesh, ("dp_replicate", "dp_shard_cp"))

        dp_cp_flat._unflatten.assert_called_once_with(0, (2, 4), ("dp_replicate", "dp_shard_cp"))
        assert result is unflatten_result

    def test_all_physical_multi_dim(self):
        """All-physical multi-dim tuple slices directly."""
        mesh = self._make_mock_mesh(("pp", "dp", "tp"))
        get_submesh(mesh, ("dp", "tp"))
        mesh.__getitem__.assert_called_once_with(("dp", "tp"))


# ---------------------------------------------------------------------------
# _unflatten_compat  (PyTorch 2.9.x compatibility shim)
# ---------------------------------------------------------------------------


class TestUnflattenCompat:
    def test_uses_native_unflatten_when_available(self):
        """When flat_mesh has _unflatten(), delegates to it directly."""
        expected = Mock()
        flat_mesh = Mock()
        flat_mesh._unflatten = Mock(return_value=expected)

        result = _unflatten_compat(flat_mesh, 0, (2, 4), ("dp_replicate", "dp_shard_cp"))

        flat_mesh._unflatten.assert_called_once_with(0, (2, 4), ("dp_replicate", "dp_shard_cp"))
        assert result is expected

    def test_fallback_when_unflatten_missing(self):
        """PyTorch 2.9.x path: reshapes mesh tensor directly into a new DeviceMesh."""
        flat_mesh = Mock(spec=[])  # no _unflatten attribute
        flat_mesh.device_type = "cuda"
        flat_mesh.mesh = torch.arange(8)

        # DeviceMesh is imported locally inside _unflatten_compat, so patch at source.
        with patch("torch.distributed.device_mesh.DeviceMesh") as MockDM:
            mock_result = Mock()
            MockDM.return_value = mock_result

            result = _unflatten_compat(flat_mesh, 0, (2, 4), ("dp_replicate", "dp_shard_cp"))

            args, kwargs = MockDM.call_args
            assert args[0] == "cuda"
            assert args[1].shape == (2, 4)
            assert args[1].tolist() == torch.arange(8).reshape(2, 4).tolist()
            assert kwargs == {"mesh_dim_names": ("dp_replicate", "dp_shard_cp")}
            assert result is mock_result

    def test_fallback_preserves_values(self):
        """The reshaped tensor preserves all rank values from the flat mesh."""
        flat_mesh = Mock(spec=[])
        flat_mesh.device_type = "cpu"
        flat_mesh.mesh = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28])

        with patch("torch.distributed.device_mesh.DeviceMesh") as MockDM:
            MockDM.return_value = Mock()
            _unflatten_compat(flat_mesh, 0, (2, 4), ("a", "b"))

            reshaped = MockDM.call_args[0][1]
            assert reshaped.shape == (2, 4)
            assert reshaped.tolist() == [[0, 4, 8, 12], [16, 20, 24, 28]]


# ---------------------------------------------------------------------------
# get_flat_mesh — PyTorch 2.9.x fallback (no _get_root_mesh)
# ---------------------------------------------------------------------------


class TestGetFlatMeshPT29Compat:
    def test_no_get_root_mesh_falls_back_to_self(self):
        """When _get_root_mesh is absent, uses device_mesh itself as root."""
        dp_flat = Mock()
        mesh = Mock(spec=["mesh_dim_names", "_flatten_mapping", "__getitem__"])
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        mesh._flatten_mapping = {"dp": dp_flat}

        result = get_flat_mesh(mesh, "dp")
        assert result is dp_flat

    def test_no_get_root_mesh_direct_dim_still_works(self):
        """Direct mesh dim lookup bypasses _get_root_mesh entirely."""
        mesh = Mock(spec=["mesh_dim_names", "__getitem__"])
        mesh.mesh_dim_names = ("dp", "tp")
        tp_sub = Mock()
        mesh.__getitem__ = Mock(return_value=tp_sub)

        result = get_flat_mesh(mesh, "tp")
        assert result is tp_sub
        mesh.__getitem__.assert_called_once_with("tp")

    def test_no_get_root_mesh_missing_dim_raises(self):
        """KeyError raised when dim absent from both mesh_dim_names and _flatten_mapping."""
        mesh = Mock(spec=["mesh_dim_names", "_flatten_mapping"])
        mesh.mesh_dim_names = ("dp", "tp")
        mesh._flatten_mapping = {}

        with pytest.raises(KeyError, match="unknown"):
            get_flat_mesh(mesh, "unknown")


# ---------------------------------------------------------------------------
# get_submesh — PyTorch 2.9.x fallback (no _get_root_mesh)
# ---------------------------------------------------------------------------


class TestGetSubmeshPT29Compat:
    def test_no_get_root_mesh_uses_self_as_root(self, monkeypatch):
        """When _get_root_mesh is absent, root falls back to device_mesh itself."""
        group_sentinel = Mock()

        dp_shard_cp_flat = Mock()
        dp_shard_cp_flat.size = Mock(return_value=4)
        dp_shard_cp_flat.get_group = Mock(return_value=group_sentinel)

        dp_cp_flat = Mock()
        dp_cp_flat.size = Mock(return_value=8)

        unflatten_result = Mock()
        unflatten_result.__getitem__ = Mock(return_value=Mock(get_group=Mock(return_value=group_sentinel)))
        dp_cp_flat._unflatten = Mock(return_value=unflatten_result)

        dp_rep_submesh = Mock()
        dp_rep_submesh.size = Mock(return_value=2)
        dp_rep_submesh.get_group = Mock(return_value=group_sentinel)

        # mesh has no _get_root_mesh — simulates PyTorch 2.9.x
        mesh = Mock(spec=["mesh_dim_names", "_flatten_mapping", "__getitem__"])
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")
        mesh._flatten_mapping = {"dp_shard_cp": dp_shard_cp_flat, "dp_cp": dp_cp_flat}
        mesh.__getitem__ = Mock(return_value=dp_rep_submesh)

        monkeypatch.setattr(
            "torch.distributed.get_process_group_ranks",
            lambda g: [0, 4],
        )

        result = get_submesh(mesh, ("dp_replicate", "dp_shard_cp"))
        assert result is unflatten_result
