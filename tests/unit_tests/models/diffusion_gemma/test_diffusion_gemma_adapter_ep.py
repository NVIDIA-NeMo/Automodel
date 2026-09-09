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

"""EP state-dict adapter tests for ``diffusion_gemma``.

These tests use fake mesh objects and monkeypatched state-dict utilities so they
can run on CPU.  They validate the diffusion-specific prefix mapping together
with the EP expert slicing/gathering contract used during checkpoint load/save.
"""

import torch

from nemo_automodel.components.models.diffusion_gemma import state_dict_adapter as adapter_mod
from nemo_automodel.components.models.diffusion_gemma.state_dict_adapter import (
    DiffusionGemmaStateDictAdapter,
)

N_EXPERTS = 4
HIDDEN = 8
EXPERT_INTER = 6


class _FakeSubmesh:
    def __init__(self, *, size: int = 1, local_rank: int = 0, rank: int = 0):
        self._size = size
        self._local_rank = local_rank
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_rank(self) -> int:
        return self._rank


class _FakeDeviceMesh:
    mesh_dim_names = ("ep_shard", "ep")


def _make_adapter() -> DiffusionGemmaStateDictAdapter:
    moe_config = type("MoE", (), {"n_routed_experts": N_EXPERTS})()
    return DiffusionGemmaStateDictAdapter(config=None, moe_config=moe_config, backend=None, dtype=torch.float32)


def _patch_ep_mesh_utils(monkeypatch, *, expert_range=(2, 4), ep_shard_rank=1):
    monkeypatch.setattr(
        adapter_mod.state_dict_utils,
        "get_expert_range_for_rank_from_mesh",
        lambda device_mesh, n_experts: expert_range,
    )
    monkeypatch.setattr(
        adapter_mod.state_dict_utils,
        "get_submesh",
        lambda device_mesh, dims: _FakeSubmesh(size=2, local_rank=ep_shard_rank)
        if dims == ("ep_shard",)
        else _FakeSubmesh(size=2, rank=ep_shard_rank),
    )
    monkeypatch.setattr(
        adapter_mod.state_dict_utils,
        "create_dtensor_from_local",
        lambda local_tensor, device_mesh, rank: local_tensor,
    )


def test_from_hf_with_ep_mesh_slices_experts_and_ep_shard_dimension(monkeypatch):
    adapter = _make_adapter()
    _patch_ep_mesh_utils(monkeypatch, expert_range=(2, 4), ep_shard_rank=1)

    gate_up = torch.arange(N_EXPERTS * 2 * EXPERT_INTER * HIDDEN, dtype=torch.float32).reshape(
        N_EXPERTS, 2 * EXPERT_INTER, HIDDEN
    )
    down = torch.arange(N_EXPERTS * HIDDEN * EXPERT_INTER, dtype=torch.float32).reshape(N_EXPERTS, HIDDEN, EXPERT_INTER)
    scale = torch.arange(1, N_EXPERTS + 1, dtype=torch.float32)
    router = torch.randn(N_EXPERTS, HIDDEN)

    native = adapter.from_hf(
        {
            # Tied embedding (-> model.embed_tokens.weight; the adapter rebuilds the
            # absent lm_head from it). Always present in a real checkpoint.
            "model.decoder.embed_tokens.weight": torch.randn(16, HIDDEN),
            "model.decoder.layers.0.router.proj.weight": router,
            "model.decoder.layers.0.router.per_expert_scale": scale,
            "model.decoder.layers.0.experts.gate_up_proj": gate_up,
            "model.decoder.layers.0.experts.down_proj": down,
        },
        device_mesh=_FakeDeviceMesh(),
    )

    gate_and_up = native["model.layers.0.moe.experts.gate_and_up_projs"]
    down_projs = native["model.layers.0.moe.experts.down_projs"]

    expected_gate_and_up = gate_up.transpose(-2, -1)[2:4, HIDDEN // 2 :, :]
    expected_down = (down.transpose(-2, -1) * scale[:, None, None])[2:4, EXPERT_INTER // 2 :, :]

    torch.testing.assert_close(gate_and_up, expected_gate_and_up)
    torch.testing.assert_close(down_projs, expected_down)
    torch.testing.assert_close(native["model.layers.0.moe.gate.proj.weight"], router)
    assert "model.decoder.layers.0.experts.gate_up_proj" not in native
    assert "model.decoder.layers.0.router.per_expert_scale" not in native


def test_gather_expert_tensor_with_ep_mesh_restores_global_expert_shape(monkeypatch):
    adapter = _make_adapter()
    _patch_ep_mesh_utils(monkeypatch, expert_range=(2, 4), ep_shard_rank=0)
    monkeypatch.setattr(adapter_mod.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(adapter_mod.state_dict_utils, "is_dtensor", lambda tensor: False)

    local_tensor = torch.randn(2, HIDDEN, 2 * EXPERT_INTER)

    global_tensor = adapter._gather_expert_tensor(
        local_tensor,
        device_mesh=_FakeDeviceMesh(),
        n_experts=N_EXPERTS,
    )

    assert global_tensor.shape == (N_EXPERTS, HIDDEN, 2 * EXPERT_INTER)
    torch.testing.assert_close(global_tensor[2:4], local_tensor)
    torch.testing.assert_close(global_tensor[:2], torch.zeros_like(global_tensor[:2]))
