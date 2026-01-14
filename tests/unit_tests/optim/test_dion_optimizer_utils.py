import pytest
import torch
import torch.nn as nn


class DummyCfgOpt:
    '''Minimal config shim compatible with build_dion_optimizer().'''

    def __init__(self, target, d: dict):
        self._target_ = target
        self._d = dict(d)

    def to_dict(self):
        return dict(self._d)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 10, bias=False)


class FakeMesh:
    '''Simple stand-in for a named DeviceMesh-like object.'''

    def __init__(self, mapping: dict, ndim: int = 1):
        self._mapping = dict(mapping)
        self.ndim = ndim

    def __getitem__(self, key):
        return self._mapping[key]


def test_build_dion_optimizer_passes_distributed_mesh_when_supported(monkeypatch):
    # Import inside test so we can monkeypatch module globals safely.
    from nemo_automodel.components.optim import utils as optim_utils

    # Avoid requiring real dion installation.
    monkeypatch.setattr(optim_utils, '_import_error', None, raising=False)

    captured = {}

    class Target:
        def __init__(self, param_groups, distributed_mesh=None, lr=None):
            captured['param_groups'] = param_groups
            captured['distributed_mesh'] = distributed_mesh
            captured['lr'] = lr

    model = TinyModel()
    mesh = FakeMesh({'dp_replicate': object(), 'dp_shard_cp': object(), 'tp': object()}, ndim=1)
    cfg = DummyCfgOpt(Target, {'lr': 1e-3, 'foo': 'ignored'})

    opt = optim_utils.build_dion_optimizer(cfg_opt=cfg, model=model, distributed_mesh=mesh)

    assert isinstance(opt, Target)
    assert captured['distributed_mesh'] is mesh
    assert captured['lr'] == pytest.approx(1e-3)
    assert isinstance(captured['param_groups'], list)
    assert len(captured['param_groups']) >= 2


def test_build_dion_optimizer_passes_named_submeshes_when_supported(monkeypatch):
    from nemo_automodel.components.optim import utils as optim_utils

    monkeypatch.setattr(optim_utils, '_import_error', None, raising=False)

    captured = {}

    class Target:
        def __init__(
            self,
            param_groups,
            replicate_mesh=None,
            outer_shard_mesh=None,
            inner_shard_mesh=None,
            lr=None,
            weight_decay=None,
        ):
            captured['param_groups'] = param_groups
            captured['replicate_mesh'] = replicate_mesh
            captured['outer_shard_mesh'] = outer_shard_mesh
            captured['inner_shard_mesh'] = inner_shard_mesh
            captured['lr'] = lr
            captured['weight_decay'] = weight_decay

    rep = object()
    outer = object()
    inner = object()
    mesh = FakeMesh({'dp_replicate': rep, 'dp_shard_cp': outer, 'tp': inner}, ndim=1)
    model = TinyModel()
    cfg = DummyCfgOpt(Target, {'lr': 2e-4, 'weight_decay': 0.1, 'unused': 123})

    opt = optim_utils.build_dion_optimizer(cfg_opt=cfg, model=model, distributed_mesh=mesh)

    assert isinstance(opt, Target)
    assert captured['replicate_mesh'] is rep
    assert captured['outer_shard_mesh'] is outer
    assert captured['inner_shard_mesh'] is inner
    assert captured['lr'] == pytest.approx(2e-4)
    assert captured['weight_decay'] == pytest.approx(0.1)
