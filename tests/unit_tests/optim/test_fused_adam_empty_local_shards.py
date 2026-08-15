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

"""TE FusedAdam construction must drop zero-numel local shards.

FSDP2 shards every parameter along dim-0 across the shard group, so any
parameter with dim-0 smaller than the group (e.g. small vision-tower biases or
norm weights on a wide mesh) leaves zero-numel local shards on the tail ranks.
TransformerEngine FusedAdam's ``multi_tensor_apply`` has no empty-tensor guard
and faults at the first optimizer step.  Both TE FusedAdam construction paths
(the typed :class:`FusedAdamConfig` and the YAML ``_target_`` factory escape
hatch) must filter locally-empty shards out before TE sees them — but never a
whole param group (rank-asymmetric ``param_groups`` desynchronize positional
LR/WD scheduling) or the whole param list: those cases must raise.
"""

import functools
import logging
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.optim.optimizer import (
    FusedAdamConfig,
    OptimizerFromFactoryConfig,
)


class _RecordingFusedAdam:
    """Stand-in for TE FusedAdam that records its constructor arguments.

    Args:
        params: Flat list of parameters or list of param-group dicts, recorded
            as-is on the class (parameters keep their original tensor/DTensor
            types and shapes; no tensor is read or modified).
    """

    last_params = None
    last_kwargs = None

    def __init__(self, params, **kwargs):
        type(self).last_params = list(params)
        type(self).last_kwargs = kwargs
        self.param_groups = []


def _te_stub_modules() -> dict[str, types.ModuleType]:
    """Module stubs exposing ``_RecordingFusedAdam`` as TE's ``FusedAdam``."""
    optimizers_mod = types.ModuleType("transformer_engine.pytorch.optimizers")
    optimizers_mod.FusedAdam = _RecordingFusedAdam
    pytorch_mod = types.ModuleType("transformer_engine.pytorch")
    pytorch_mod.optimizers = optimizers_mod
    te_mod = types.ModuleType("transformer_engine")
    te_mod.pytorch = pytorch_mod
    return {
        "transformer_engine": te_mod,
        "transformer_engine.pytorch": pytorch_mod,
        "transformer_engine.pytorch.optimizers": optimizers_mod,
    }


@pytest.fixture
def stub_te_fused_adam(monkeypatch):
    """Install a fake ``transformer_engine.pytorch.optimizers.FusedAdam``.

    Makes the CPU tests independent of a TransformerEngine installation and
    lets them observe exactly which parameters reach the TE constructor.
    """
    for name, module in _te_stub_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    _RecordingFusedAdam.last_params = None
    _RecordingFusedAdam.last_kwargs = None
    return _RecordingFusedAdam


@pytest.fixture
def single_rank_pg():
    """Single-rank gloo process group, enough to build CPU DTensors."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    yield
    dist.destroy_process_group()


class TestFusedAdamConfigDropsEmptyLocalShards:
    def test_flat_params_drop_empty_and_warn(self, stub_te_fused_adam, caplog):
        empty = nn.Parameter(torch.empty(0))
        kept = nn.Parameter(torch.ones(3))

        with caplog.at_level(logging.WARNING, logger="nemo_automodel.components.optim.optimizer"):
            FusedAdamConfig(lr=1e-3)._build_optimizer([empty, kept])

        assert stub_te_fused_adam.last_params == [kept]
        assert stub_te_fused_adam.last_kwargs["lr"] == 1e-3
        assert "zero-numel local shards" in caplog.text

    def test_flat_params_all_empty_raises(self, stub_te_fused_adam):
        with pytest.raises(ValueError, match="zero-numel local shard"):
            FusedAdamConfig()._build_optimizer([nn.Parameter(torch.empty(0))])

        assert stub_te_fused_adam.last_params is None

    def test_param_groups_drop_empty_params_and_keep_options(self, stub_te_fused_adam):
        empty = nn.Parameter(torch.empty(0))
        kept_a = nn.Parameter(torch.ones(3))
        kept_b = nn.Parameter(torch.ones(2))

        FusedAdamConfig().build_from_param_groups(
            [
                {"params": [empty, kept_a], "lr": 0.25},
                {"params": [kept_b], "lr": 0.5},
            ]
        )

        assert stub_te_fused_adam.last_params == [
            {"params": [kept_a], "lr": 0.25},
            {"params": [kept_b], "lr": 0.5},
        ]

    def test_param_group_with_only_empty_shards_raises(self, stub_te_fused_adam):
        # A whole-group drop would leave param_groups rank-asymmetric (LR/WD
        # schedulers address groups positionally, so ranks would silently schedule
        # different values), and keeping the group empty breaks torch DCP's
        # flattened optimizer-state load.  Construction must fail loudly instead.
        with pytest.raises(ValueError, match="param group 1"):
            FusedAdamConfig().build_from_param_groups(
                [
                    {"params": [nn.Parameter(torch.ones(3))], "lr": 0.25},
                    {"params": [nn.Parameter(torch.empty(0))], "lr": 0.5},
                ]
            )

        assert stub_te_fused_adam.last_params is None

    def test_dtensor_uses_local_not_global_numel(self, stub_te_fused_adam, single_rank_pg):
        mesh = init_device_mesh("cpu", (1,))
        # Globally non-empty (4, 3) parameter whose local dim-0 shard on this rank is empty —
        # the shape FSDP2 leaves on tail ranks when dim-0 < shard-group size.
        locally_empty = nn.Parameter(
            DTensor.from_local(torch.empty(0, 3), mesh, [Shard(0)], run_check=False, shape=(4, 3), stride=(3, 1))
        )
        locally_present = nn.Parameter(DTensor.from_local(torch.ones(2, 3), mesh, [Shard(0)], run_check=False))
        assert locally_empty.numel() == 12  # global numel would NOT catch this shard

        FusedAdamConfig()._build_optimizer([locally_empty, locally_present])

        assert stub_te_fused_adam.last_params == [locally_present]


class _TinyModel(nn.Module):
    """Linear probe plus a zero-numel trainable parameter."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.empty = nn.Parameter(torch.empty(0))


class _OnlyEmptyTrainableModel(nn.Module):
    """Frozen linear; the sole trainable parameter has a zero-numel (local) shard."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.linear.requires_grad_(False)
        self.empty = nn.Parameter(torch.empty(0))


class TestFactoryEscapeHatchDropsEmptyLocalShards:
    def test_te_fused_adam_factory_drops_empty(self, stub_te_fused_adam):
        cfg = OptimizerFromFactoryConfig(factory=stub_te_fused_adam, kwargs={"lr": 1e-3})

        cfg.build(_TinyModel())

        assert all(p.numel() > 0 for p in stub_te_fused_adam.last_params)
        assert len(stub_te_fused_adam.last_params) == 2  # linear weight + bias

    def test_partial_wrapped_te_fused_adam_factory_drops_empty(self, stub_te_fused_adam):
        # functools.partial wrappers must be unwrapped by the TE FusedAdam identity
        # check; without unwrapping, the empty shard would reach the constructor.
        factory = functools.partial(stub_te_fused_adam, bias_correction=True)
        cfg = OptimizerFromFactoryConfig(factory=factory, kwargs={"lr": 1e-3})

        cfg.build(_TinyModel())

        assert all(p.numel() > 0 for p in stub_te_fused_adam.last_params)
        assert len(stub_te_fused_adam.last_params) == 2  # linear weight + bias
        assert stub_te_fused_adam.last_kwargs["bias_correction"] is True

    def test_te_fused_adam_factory_all_params_empty_raises(self, stub_te_fused_adam):
        # The pre-filter `len(trainable_params) > 0` assert passes (one trainable
        # param), so the post-filter empty list must raise the specific error, not
        # torch's generic "optimizer got an empty parameter list".
        cfg = OptimizerFromFactoryConfig(factory=stub_te_fused_adam, kwargs={"lr": 1e-3})

        with pytest.raises(ValueError, match="zero-numel local shard"):
            cfg.build(_OnlyEmptyTrainableModel())

        assert stub_te_fused_adam.last_params is None

    def test_te_fused_adam_factory_drops_empty_param_groups(self, stub_te_fused_adam):
        cfg = OptimizerFromFactoryConfig(factory=stub_te_fused_adam, kwargs={"lr": 1e-3})
        kept = nn.Parameter(torch.ones(3))

        cfg.build_from_param_groups(
            [
                {"params": [nn.Parameter(torch.empty(0)), kept], "weight_decay": 0.1},
            ]
        )

        assert stub_te_fused_adam.last_params == [{"params": [kept], "weight_decay": 0.1}]

    def test_non_te_factory_is_not_filtered(self):
        # torch.optim handles empty tensors; the guard must not change other factories.
        cfg = OptimizerFromFactoryConfig(factory=torch.optim.SGD, kwargs={"lr": 0.01})

        opt = cfg.build(_TinyModel())[0]

        assert len(opt.param_groups[0]["params"]) == 3  # linear weight + bias + empty


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_adam_cuda_step_ignores_locally_empty_parameter():
    """Real TE FusedAdam step succeeds when a zero-numel parameter is present.

    Without the guard, TE's multi_tensor_apply faults with a CUDA misaligned
    address / illegal memory access on the empty tensor.
    """
    pytest.importorskip("transformer_engine")

    device = torch.device("cuda", 0)
    # bf16 params with fp32 master weights: the configuration wide-mesh runs use.
    empty = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16))
    kept = nn.Parameter(torch.tensor([1.0, -2.0, 3.0], device=device, dtype=torch.bfloat16))
    empty.grad = torch.empty_like(empty)
    kept.grad = torch.tensor([0.5, -0.25, 1.0], device=device, dtype=torch.bfloat16)
    before = kept.detach().clone()

    opt = FusedAdamConfig(lr=1e-2, master_weight_dtype="torch.float32")._build_optimizer([empty, kept])
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["params"] == [kept]

    opt.step()
    torch.cuda.synchronize(device)

    assert torch.isfinite(kept).all()
    assert not torch.equal(kept, before)
