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

import pytest
import torch

import nemo_automodel.components.models.kimi_k3.model as model_mod
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiK3MoE, _build_moe_config


def _tiny_config(num_shared_experts: int = 2) -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=num_shared_experts,
        first_k_dense_replace=0,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=1,
    )


def _build_moe(overlap: bool, device: str, num_shared_experts: int = 2) -> KimiK3MoE:
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        fake_balanced_gate=True,
        shared_expert_overlap=overlap,
    )
    config = _tiny_config(num_shared_experts)
    moe_config = _build_moe_config(config, torch.float32, None)
    torch.manual_seed(0)
    moe = KimiK3MoE(config, moe_config, backend).to(device)
    # The expert parameters are allocated with torch.empty and only filled by init_weights (the model
    # builder calls it); without it the test would run on whatever the allocator hands out — zeros on
    # some machines, NaN-producing garbage on others.
    moe.init_weights(torch.device(device), init_std=0.02)
    return moe


def _run(moe: KimiK3MoE, x: torch.Tensor):
    moe.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    out = moe(x)
    out.float().square().sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in moe.named_parameters() if p.grad is not None}
    return out.detach().clone(), x.grad.detach().clone(), grads


def test_backend_flag_defaults_off():
    assert BackendConfig().shared_expert_overlap is False


def _assert_same(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    # The two models run the same ops on the same values; only fp32 reduction order may differ
    # (CPU GEMM paths depend on buffer alignment, CUDA index_add accumulates atomically), so
    # compare at fp32 resolution rather than bitwise.
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=lambda m: f"{what}: {m}")


def test_cpu_path_ignores_overlap_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(model_mod, "_shared_expert_stream", lambda device: calls.append(device) or None)
    moe_ref = _build_moe(False, "cpu")
    moe_ovl = _build_moe(True, "cpu")
    moe_ovl.load_state_dict(moe_ref.state_dict())
    x = torch.randn(2, 6, 32)
    out_ref, gx_ref, g_ref = _run(moe_ref, x)
    out_ovl, gx_ovl, g_ovl = _run(moe_ovl, x)
    assert calls == []  # CPU tensors never take the side-stream path
    _assert_same(out_ref, out_ovl, "output")
    _assert_same(gx_ref, gx_ovl, "input grad")
    for name in g_ref:
        _assert_same(g_ref[name], g_ovl[name], name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stream overlap needs CUDA")
def test_gpu_overlap_matches_sequential(monkeypatch):
    real_stream = model_mod._shared_expert_stream
    calls = []

    def counted(device):
        calls.append(device)
        return real_stream(device)

    monkeypatch.setattr(model_mod, "_shared_expert_stream", counted)
    moe_ref = _build_moe(False, "cuda")
    moe_ovl = _build_moe(True, "cuda")
    moe_ovl.load_state_dict(moe_ref.state_dict())
    x = torch.randn(3, 8, 32, device="cuda")
    for _ in range(2):  # second pass reuses the cached side stream
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
    moe = _build_moe(True, "cuda", num_shared_experts=0)
    assert moe.shared_experts is None
    x = torch.randn(2, 4, 32, device="cuda", requires_grad=True)
    moe(x).sum().backward()
    assert x.grad is not None
