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

"""Tests for mesh_utils: get_flat_mesh, get_submesh utilities."""

from unittest.mock import Mock, PropertyMock

import pytest

from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh, get_submesh


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
        result = get_flat_mesh(mesh, "tp")
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
        result = get_submesh(mesh, ("tp",))
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
        result = get_submesh(mesh, ("dp_replicate", "dp_shard"))
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

        def getitem(key):
            if key == "dp_replicate":
                return dp_rep_submesh
            raise KeyError(key)

        mesh.__getitem__ = Mock(side_effect=getitem)

        monkeypatch.setattr(
            "torch.distributed.get_process_group_ranks",
            lambda g: [0, 4],
        )

        result = get_submesh(mesh, ("dp_replicate", "dp_shard_cp"))

        dp_cp_flat._unflatten.assert_called_once_with(
            0, (2, 4), ("dp_replicate", "dp_shard_cp")
        )
        assert result is unflatten_result

    def test_mixed_physical_flattened_prefers_direct_slice(self):
        """Direct slicing should win so the submesh stays attached to the root mesh."""
        direct_submesh = Mock()
        mesh = self._make_mock_mesh(("pp", "dp_replicate", "dp_shard", "cp", "tp"))
        mesh.__getitem__ = Mock(return_value=direct_submesh)

        dp_flat = Mock()
        dp_flat.size = Mock(return_value=16)
        dp_flat._unflatten = Mock()
        dp_shard_cp_flat = Mock()
        dp_shard_cp_flat.size = Mock(return_value=16)
        dp_shard_cp_flat._unflatten = Mock()
        dp_cp_flat = Mock()
        dp_cp_flat.size = Mock(return_value=16)
        dp_cp_flat._unflatten = Mock()
        mesh._get_root_mesh.return_value._flatten_mapping = {
            "dp": dp_flat,
            "dp_shard_cp": dp_shard_cp_flat,
            "dp_cp": dp_cp_flat,
        }

        result = get_submesh(mesh, ("dp_replicate", "dp_shard_cp"))

        mesh.__getitem__.assert_called_once_with(("dp_replicate", "dp_shard_cp"))
        dp_flat._unflatten.assert_not_called()
        dp_shard_cp_flat._unflatten.assert_not_called()
        dp_cp_flat._unflatten.assert_not_called()
        assert result is direct_submesh

    def test_mixed_physical_flattened_ambiguous_parent_prefers_superset_parent(self, monkeypatch):
        """Fallback should reject same-size flattened parents that do not cover all source dims."""
        group_sentinel = Mock()

        dp_flat = Mock()
        dp_flat.size = Mock(return_value=16)
        dp_flat._unflatten = Mock()

        dp_shard_cp_flat = Mock()
        dp_shard_cp_flat.size = Mock(return_value=16)
        dp_shard_cp_flat._unflatten = Mock()
        dp_shard_cp_flat.get_group = Mock(return_value=group_sentinel)

        dp_cp_flat = Mock()
        dp_cp_flat.size = Mock(return_value=16)
        unflatten_result = Mock()
        unflatten_result.__getitem__ = Mock(return_value=Mock(get_group=Mock(return_value=group_sentinel)))
        dp_cp_flat._unflatten = Mock(return_value=unflatten_result)

        dp_rep_submesh = Mock()
        dp_rep_submesh.size = Mock(return_value=1)
        dp_rep_submesh.get_group = Mock(return_value=group_sentinel)

        mesh = Mock()
        mesh.mesh_dim_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
        root = Mock()
        root._flatten_mapping = {"dp": dp_flat, "dp_shard_cp": dp_shard_cp_flat, "dp_cp": dp_cp_flat}
        root.__getitem__ = Mock(side_effect=KeyError(("dp_replicate", "dp_shard_cp")))
        mesh._get_root_mesh = Mock(return_value=root)

        def getitem(key):
            if key == "dp_replicate":
                return dp_rep_submesh
            raise KeyError(key)

        mesh.__getitem__ = Mock(side_effect=getitem)

        monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda g: [0])

        result = get_submesh(mesh, ("dp_replicate", "dp_shard_cp"))

        dp_flat._unflatten.assert_not_called()
        dp_shard_cp_flat._unflatten.assert_not_called()
        dp_cp_flat._unflatten.assert_called_once_with(0, (1, 16), ("dp_replicate", "dp_shard_cp"))
        assert result is unflatten_result

    def test_all_physical_multi_dim(self):
        """All-physical multi-dim tuple slices directly."""
        mesh = self._make_mock_mesh(("pp", "dp", "tp"))
        result = get_submesh(mesh, ("dp", "tp"))
        mesh.__getitem__.assert_called_once_with(("dp", "tp"))
