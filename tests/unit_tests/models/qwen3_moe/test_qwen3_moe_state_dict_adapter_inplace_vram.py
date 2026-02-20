# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import gc
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _skip_if_low_free_cuda_mem(required_bytes: int, safety_factor: float = 2.0) -> None:
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * safety_factor):
        pytest.skip(
            f"Not enough free CUDA memory for this test. "
            f"Need ~{required_bytes * safety_factor / (1024**3):.2f} GiB, have {free / (1024**3):.2f} GiB."
        )


def _reset_cuda_mem_stats() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _make_adapter(dim: int, moe_inter_dim: int, n_experts: int = 4, dtype: torch.dtype = torch.float16):
    # Qwen3MoeStateDictAdapter only needs moe_config/backend for conversion; the HF config is not used.
    cfg = SimpleNamespace()
    moe_config = MoEConfig(
        n_routed_experts=n_experts,
        n_shared_experts=0,
        n_activated_experts=min(2, n_experts),
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=dim,
        inter_dim=dim * 2,
        moe_inter_dim=moe_inter_dim,
        norm_topk_prob=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        dtype=dtype,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    return Qwen3MoeStateDictAdapter(config=cfg, moe_config=moe_config, backend=backend, dtype=dtype)


def test_qwen3_moe_to_hf_inplace_reduces_live_vram():
    """Regression: inplace=True should drop the large merged expert tensors after splitting.

    In non-inplace mode, the merged tensors remain referenced by the input dict,
    so after conversion you hold both merged + split expert weights.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float16
    n_experts = 4
    dim = 4096
    moe_inter_dim = 1024

    adapter = _make_adapter(dim=dim, moe_inter_dim=moe_inter_dim, n_experts=n_experts, dtype=dtype)

    def make_native_state_dict():
        sd = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.empty(
                (n_experts, dim, 2 * moe_inter_dim), device=device, dtype=dtype
            ),
            "model.layers.0.mlp.experts.down_projs": torch.empty((n_experts, moe_inter_dim, dim), device=device, dtype=dtype),
        }
        expected_saved = _tensor_bytes(sd["model.layers.0.mlp.experts.gate_and_up_projs"]) + _tensor_bytes(
            sd["model.layers.0.mlp.experts.down_projs"]
        )
        return sd, expected_saved

    tmp_sd, expected_saved_bytes = make_native_state_dict()
    required = expected_saved_bytes * 2  # merged + split
    del tmp_sd
    _skip_if_low_free_cuda_mem(required_bytes=required)

    # Non-inplace
    _reset_cuda_mem_stats()
    native_sd, expected_saved_bytes = make_native_state_dict()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    out = adapter.to_hf(native_sd, inplace=False)
    torch.cuda.synchronize()
    delta_non_inplace = torch.cuda.memory_allocated() - baseline
    del out, native_sd

    # Inplace
    _reset_cuda_mem_stats()
    native_sd2, _ = make_native_state_dict()
    torch.cuda.synchronize()
    baseline2 = torch.cuda.memory_allocated()
    out2 = adapter.to_hf(native_sd2, inplace=True)
    torch.cuda.synchronize()
    delta_inplace = torch.cuda.memory_allocated() - baseline2
    del out2, native_sd2

    assert (delta_non_inplace - delta_inplace) >= int(expected_saved_bytes * 0.70), (
        f"Expected inplace conversion to free most of merged expert tensor VRAM.\n"
        f"expected_saved_bytes={expected_saved_bytes}\n"
        f"delta_non_inplace={delta_non_inplace}\n"
        f"delta_inplace={delta_inplace}\n"
    )


def test_qwen3_moe_from_hf_inplace_reduces_live_vram():
    """Regression: inplace=True should drop the split per-expert tensors after merging.

    In non-inplace mode, the split per-expert tensors remain referenced by the input dict,
    so after conversion you hold both split + merged expert weights.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float16
    n_experts = 4
    dim = 4096
    moe_inter_dim = 1024

    adapter = _make_adapter(dim=dim, moe_inter_dim=moe_inter_dim, n_experts=n_experts, dtype=dtype)

    def make_hf_state_dict():
        sd = {}
        for e in range(n_experts):
            sd[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.empty(
                (moe_inter_dim, dim), device=device, dtype=dtype
            )
            sd[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.empty((moe_inter_dim, dim), device=device, dtype=dtype)
            sd[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.empty((dim, moe_inter_dim), device=device, dtype=dtype)
        expected_saved = sum(_tensor_bytes(t) for t in sd.values())
        return sd, expected_saved

    tmp_sd, expected_saved_bytes = make_hf_state_dict()
    required = expected_saved_bytes * 2  # split + merged
    del tmp_sd
    _skip_if_low_free_cuda_mem(required_bytes=required)

    # Non-inplace
    _reset_cuda_mem_stats()
    hf_sd, expected_saved_bytes = make_hf_state_dict()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    out = adapter.from_hf(hf_sd, inplace=False)
    torch.cuda.synchronize()
    delta_non_inplace = torch.cuda.memory_allocated() - baseline
    del out, hf_sd

    # Inplace
    _reset_cuda_mem_stats()
    hf_sd2, _ = make_hf_state_dict()
    torch.cuda.synchronize()
    baseline2 = torch.cuda.memory_allocated()
    out2 = adapter.from_hf(hf_sd2, inplace=True)
    torch.cuda.synchronize()
    delta_inplace = torch.cuda.memory_allocated() - baseline2
    del out2, hf_sd2

    assert (delta_non_inplace - delta_inplace) >= int(expected_saved_bytes * 0.70), (
        f"Expected inplace conversion to free most of split expert tensor VRAM.\n"
        f"expected_saved_bytes={expected_saved_bytes}\n"
        f"delta_non_inplace={delta_non_inplace}\n"
        f"delta_inplace={delta_inplace}\n"
    )

