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

"""Shared-expert overlap in the shared ``MoE`` class (``BackendConfig.shared_expert_overlap``)."""

import pytest
import torch

import nemo_automodel.components.moe.layers as layers_mod
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


def _tiny_moe_config(n_shared_experts: int = 1) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=n_shared_experts,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=8,
        inter_dim=16,
        moe_inter_dim=16,
        norm_topk_prob=False,
        dtype=torch.float32,
    )


def _build_moe(overlap: bool, device: str, n_shared_experts: int = 1) -> MoE:
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        fake_balanced_gate=True,
        shared_expert_overlap=overlap,
    )
    torch.manual_seed(0)
    moe = MoE(_tiny_moe_config(n_shared_experts), backend).to(device)
    # The expert parameters are allocated with torch.empty and only filled by init_weights (the model
    # builder calls it under no_grad); without it the test would run on whatever the allocator hands out.
    with torch.no_grad():
        moe.init_weights(torch.device(device), init_std=0.02)
    return moe


def _run(moe: MoE, x: torch.Tensor):
    moe.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out = moe(x)
    out.float().square().sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
    return out.detach().clone(), x.grad.detach().clone(), grads


def _assert_same(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    # Same ops on the same values; only fp32 reduction order may differ between the two streams.
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=lambda m: f"{what}: {m}")


def test_backend_flag_defaults_off():
    assert BackendConfig().shared_expert_overlap is False


def test_cpu_path_ignores_overlap_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(layers_mod, "_shared_expert_stream", lambda device: calls.append(device) or None)
    moe_ref = _build_moe(False, "cpu")
    moe_ovl = _build_moe(True, "cpu")
    moe_ovl.load_state_dict(moe_ref.state_dict())
    x = torch.randn(2, 6, 8)
    out_ref, gx_ref, g_ref = _run(moe_ref, x)
    out_ovl, gx_ovl, g_ovl = _run(moe_ovl, x)
    assert calls == []  # CPU tensors never take the side-stream path
    _assert_same(out_ref, out_ovl, "output")
    _assert_same(gx_ref, gx_ovl, "input grad")
    assert set(g_ref) == set(g_ovl)
    for name in g_ref:
        _assert_same(g_ref[name], g_ovl[name], name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stream overlap needs CUDA")
def test_gpu_overlap_matches_sequential(monkeypatch):
    real_stream = layers_mod._shared_expert_stream
    calls = []

    def counted(device):
        calls.append(device)
        return real_stream(device)

    monkeypatch.setattr(layers_mod, "_shared_expert_stream", counted)
    moe_ref = _build_moe(False, "cuda")
    moe_ovl = _build_moe(True, "cuda")
    moe_ovl.load_state_dict(moe_ref.state_dict())
    x = torch.randn(3, 8, 8, device="cuda")
    for _ in range(2):  # the second pass reuses the cached side stream
        out_ref, gx_ref, g_ref = _run(moe_ref, x)
        out_ovl, gx_ovl, g_ovl = _run(moe_ovl, x)
        torch.cuda.synchronize()
        _assert_same(out_ref, out_ovl, "output")
        _assert_same(gx_ref, gx_ovl, "input grad")
        assert set(g_ref) == set(g_ovl)
        for name in g_ref:
            _assert_same(g_ref[name], g_ovl[name], name)
    assert len(calls) == 2  # one side-stream fetch per overlapped forward, none for the reference
    stream = real_stream(x.device)
    assert stream is real_stream(x.device)
    assert stream != torch.cuda.current_stream()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stream overlap needs CUDA")
def test_gpu_overlap_without_shared_experts_is_noop():
    moe = _build_moe(True, "cuda", n_shared_experts=0)
    assert moe.shared_experts is None
    x = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    moe(x).sum().backward()
    assert x.grad is not None
