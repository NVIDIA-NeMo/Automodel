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

import pytest
import torch
from torch import nn

import nemo_automodel.components.training.utils as training_utils
from nemo_automodel.components.optim.gradient_sketch import GradientSketchOptimizer, SketchSettings
from nemo_automodel.components.training.utils import SketchClippingOptimizer, clip_grads_for_optimizers


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _sketch(net):
    return GradientSketchOptimizer(
        net, torch.optim.AdamW(net.parameters(), lr=0.1), SketchSettings(rank=2, target_modules=("0",))
    )


def test_dispatches_to_sketch_optimizer():
    net = _net()
    opt = _sketch(net)
    net(torch.randn(3, 4)).square().sum().backward()
    norm = clip_grads_for_optimizers([opt], 0.5, [net])
    assert torch.is_tensor(norm) and norm.item() > 0 and opt.ready


def test_plain_optimizers_use_dense_path(monkeypatch):
    net = _net()
    opt = torch.optim.AdamW(net.parameters(), lr=0.1)
    sentinel = torch.tensor(3.0)
    calls = []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(training_utils, "scale_grads_and_clip_grad_norm", stub)
    out = clip_grads_for_optimizers(
        [opt], 1.0, [net], dp_group="ignored", pp_enabled=False, foreach=True, grad_norm_backend="triton"
    )
    assert out is sentinel
    assert calls == [((1.0, [net]), {"pp_enabled": False, "foreach": True, "grad_norm_backend": "triton"})]


def test_mixed_optimizers_rejected():
    net = _net()
    sketch = _sketch(net)
    with pytest.raises(ValueError, match="mix"):
        clip_grads_for_optimizers([sketch, torch.optim.SGD(net[1].parameters(), lr=0.1)], 1.0, [net])


def test_protocol_matches_only_sketch_optimizers():
    net = _net()
    assert isinstance(_sketch(net), SketchClippingOptimizer)
    for plain in (torch.optim.AdamW(_net().parameters()), torch.optim.SGD(_net().parameters(), lr=0.1)):
        assert not isinstance(plain, SketchClippingOptimizer)


def test_rejects_owner_scaled_gradients():
    net = _net()
    opt = _sketch(net)
    net[1].bias._nemo_model_owned_grad_divisor = 2.0
    net(torch.randn(3, 4)).square().sum().backward()
    with pytest.raises(RuntimeError, match="owner-scaled"):
        clip_grads_for_optimizers([opt], 1.0, [net])
    assert not opt.ready


class _FakeMesh:
    def __init__(self, sizes):
        self.sizes = sizes
        self.mesh_dim_names = tuple(sizes)

    def __getitem__(self, name):
        size = self.sizes[name]
        return type("_Dim", (), {"size": lambda self: size})()


@pytest.mark.parametrize("ep, rejected", [(2, True), (1, False)])
def test_rejects_expert_parallelism(ep, rejected):
    net = _net()
    opt = _sketch(net)
    net(torch.randn(3, 4)).square().sum().backward()
    kwargs = {"moe_mesh": _FakeMesh({"ep_shard": 1, "ep": ep}), "ep_axis_name": "ep"}
    if rejected:
        with pytest.raises(RuntimeError, match="expert parallelism"):
            clip_grads_for_optimizers([opt], 1.0, [net], **kwargs)
    else:
        clip_grads_for_optimizers([opt], 1.0, [net], **kwargs)
        assert opt.ready
