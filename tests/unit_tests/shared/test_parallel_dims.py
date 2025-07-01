# tests/test_parallel.py
import math
import os
import types
import pytest
import torch

import nemo_automodel.shared.parallel_dims as parallel
from nemo_automodel.shared.parallel_dims import ParallelDims


# ---------------------------------------------------------------------------
# Fixture 1 – stub the COMPLETE distributed stack that might be consulted
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_distributed(monkeypatch):
    """
    Replaces every init/destroy/group query entry-point that may be reached
    from *any* of

      • torch.distributed (top-level)
      • torch.distributed.distributed_c10d
      • torch.distributed.device_mesh.DeviceMesh._get_or_create_default_group

    with a minimal in-memory fake backend so that *no real rendez-vous*
    is attempted and the tests run in a single CPU process.
    """
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d

    # mutable global state
    _state = dict(initialized=False, world_size=1, rank=0)

    # helpers ----------------------------------------------------------------
    class _FakeProcessGroup:
        def __init__(self, size, rank=0):
            """ """
            self._size = size
            self._rank = rank

        def size(self):
            """ """
            return self._size

        def rank(self):
            """ """
            return self._rank

    # generic stubs -----------------------------------------------------------
    def _fake_init_process_group(backend="gloo", **kwargs):
        """
        Minimal replacement for torch.distributed.init_process_group.
        • Marks the backend as initialised.
        • Sets rank / world_size from kwargs **or** the usual torchrun
        environment variables so that size is always correct.
        """
        _state["initialized"] = True

        # world-size precedence: explicit kwarg > env > existing
        _state["world_size"] = kwargs.get(
            "world_size",
            int(os.environ.get("WORLD_SIZE", _state["world_size"]))
        )
        _state["rank"] = kwargs.get(
            "rank",
            int(os.environ.get("RANK", _state["rank"]))
        )

    def _fake_destroy():
        """ """
        _state["initialized"] = False

    # top-level torch.distributed
    monkeypatch.setattr(dist, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: _state["initialized"], raising=False)
    monkeypatch.setattr(dist, "init_process_group", _fake_init_process_group, raising=False)
    monkeypatch.setattr(dist, "destroy_process_group", _fake_destroy, raising=False)
    monkeypatch.setattr(dist, "get_world_size", lambda: _state["world_size"], raising=False)
    monkeypatch.setattr(dist, "get_rank", lambda: _state["rank"], raising=False)
    dist.group = types.SimpleNamespace(WORLD=_FakeProcessGroup(size=lambda: _state["world_size"]))

    # low-level alias used by DeviceMesh
    monkeypatch.setattr(c10d, "init_process_group", _fake_init_process_group, raising=False)

    # provide 'new_group' so user code could create additional PGs
    monkeypatch.setattr(
        dist, "new_group",
        lambda ranks=None,
        **kw: _FakeProcessGroup(size=len(ranks) if ranks else _state["world_size"]),
        raising=False)

    yield


# ---------------------------------------------------------------------------
# Fixture 2 – replace real DDP with a pass-through wrapper
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_ddp(monkeypatch):
    class _IdentityDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            """ """
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            """  """
            return self.module(*args, **kwargs)

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _IdentityDDP, raising=False)
    yield


# ---------------------------------------------------------------------------
# Fixture 3 – stub init_device_mesh to avoid the heavy implementation
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_device_mesh(monkeypatch):
    """
    Provide a very small substitute for `torch.distributed.device_mesh.init_device_mesh`
    that supports the attributes / indexing patterns used by the production code.
    """
    class _FakeDeviceMesh:
        def __init__(self, device_type, dims, mesh_dim_names=None):
            """ """
            self.device_type = device_type
            self.mesh = torch.empty(*dims) if dims else torch.empty(0)
            # store names so that `mesh[dim_name]` works
            self.mesh_dim_names = list(mesh_dim_names or [])

            # trivial pg for every dim
            self._pg = types.SimpleNamespace(size=lambda: self.mesh.numel() if self.mesh.numel() else 1)

        # support slicing like mesh["dp_replicate"]
        def __getitem__(self, key):
            """ """
            return self

        # public flatten in 2.2+, or private in earlier versions
        def flatten(self, mesh_dim_name):
            """ """
            # just pretend we did something
            self.mesh_dim_names = [mesh_dim_name]
            return self

        _flatten = flatten

        # mimick access helpers used in code
        def get_group(self):  # 2.2+
            """ """
            return self._pg

        @property
        def _process_group(self):  # <2.2
            """ """
            return self._pg

    def _fake_init_device_mesh(device_type, dims, mesh_dim_names=None):
        """ """
        return _FakeDeviceMesh(device_type, dims, mesh_dim_names)

    # patch both references: the public helper and the symbol captured
    # inside the module under test.
    monkeypatch.setattr("torch.distributed.device_mesh.init_device_mesh",
                        _fake_init_device_mesh, raising=True)
    monkeypatch.setattr(parallel, "init_device_mesh",
                        _fake_init_device_mesh, raising=True)
    yield


# =========================================================================== #
#                               TEST MATRIX                                   #
# =========================================================================== #

# ---------- valid configurations -------------------------------------------
VALID_CFGS = [
    # (dp_rep, dp_shard, cp, tp, pp, ep, world)
    (1,  1, 1, 1, 1, 1, 1),
    (2,  1, 1, 1, 1, 1, 2),
    (1,  2, 1, 1, 1, 1, 2),
    (1, -1, 1, 1, 1, 1, 2),           # infer dp_shard
    (2,  2, 1, 1, 1, 1, 4),
    (1,  1, 2, 2, 1, 1, 4),
    (2, -1, 2, 1, 1, 1, 4),
    (1, -1, 2, 2, 1, 1, 4),
    (1,  1, 1, 1, 2, 2, 4),
]

@pytest.mark.parametrize("dp_rep,dp_sh,cp,tp,pp,ep,ws", VALID_CFGS)
def test_paralleldims_valid(dp_rep, dp_sh, cp, tp, pp, ep, ws):
    """ """
    dims = ParallelDims(
        dp_replicate=dp_rep,
        dp_shard=dp_sh,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        world_size=ws,
    )

    # product must match world_size
    expected = dp_rep * dims.dp_shard * cp * tp * pp * ep
    assert expected == ws

    # mesh creation must succeed with our stub
    mesh = dims.build_mesh(device_type="cpu")
    # For ws==1 empty mesh (0-D); else prod(shape)==ws
    if ws == 1:
        assert math.prod(mesh.mesh.shape) == 0
    else:
        assert math.prod(mesh.mesh.shape) == ws


# ---------- invalid configurations -----------------------------------------
INVALID_CFGS = [
    dict(dp_replicate=2, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1),
    dict(dp_replicate=1, dp_shard=3, cp=1, tp=1, pp=1, ep=1, world_size=2),
    dict(dp_replicate=0, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1),
    dict(dp_replicate=1, dp_shard=-1, cp=3, tp=1, pp=1, ep=1, world_size=4),
]

@pytest.mark.parametrize("kwargs", INVALID_CFGS)
def test_paralleldims_invalid(kwargs):
    """ """
    with pytest.raises((ValueError, TypeError)):
        ParallelDims(**kwargs)

