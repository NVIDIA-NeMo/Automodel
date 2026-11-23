# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import pytest
import torch
from torch.distributed._tensor.placement_types import Replicate, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_linear.state_dict_adapter import KimiLinear48BStateDictAdapter
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig


@dataclass
class MockKimiLinear48BConfig:
    hidden_size: int = 6
    intermediate_size: int = 12
    moe_intermediate_size: int = 4
    num_experts: int = 3
    torch_dtype: str = "bfloat16"


@pytest.fixture
def config():
    return MockKimiLinear48BConfig()


@pytest.fixture
def moe_config(config):
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=2.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
        force_e_score_correction_bias=True,
    )


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="eager",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


@pytest.fixture
def adapter(config, moe_config, backend):
    return KimiLinear48BStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)


class _FakeMesh:
    def __init__(self, *, rank: int, size: int, names: tuple[str, ...]):
        self._rank = rank
        self._size = size
        self.mesh_dim_names = names

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _FakeDTensor:
    def __init__(self, *, local_tensor: torch.Tensor, placement: object, mesh: _FakeMesh):
        """Create a DTensor-like wrapper for adapter unit tests.

        Args:
            local_tensor: Rank-local expert tensor of shape [local_experts, ...].
            placement: DTensor placement metadata attached to the fake value.
            mesh: Device mesh metadata attached to the fake value.
        """
        self._local_tensor = local_tensor
        self.placements = (placement,)
        self.device_mesh = mesh

    def to_local(self) -> torch.Tensor:
        return self._local_tensor


def _patch_fake_dtensor(monkeypatch):
    monkeypatch.setattr(state_dict_utils, "is_dtensor", lambda tensor: isinstance(tensor, _FakeDTensor))


def _make_hf_expert_state(n_experts: int, inter_dim: int, dim: int) -> dict[str, torch.Tensor]:
    state = {}
    for expert in range(n_experts):
        base = expert * 1000
        state[f"model.layers.1.block_sparse_moe.experts.{expert}.w1.weight"] = torch.arange(
            base, base + inter_dim * dim, dtype=torch.float32
        ).reshape(inter_dim, dim)
        state[f"model.layers.1.block_sparse_moe.experts.{expert}.w3.weight"] = torch.arange(
            base + 100, base + 100 + inter_dim * dim, dtype=torch.float32
        ).reshape(inter_dim, dim)
        state[f"model.layers.1.block_sparse_moe.experts.{expert}.w2.weight"] = torch.arange(
            base + 200, base + 200 + dim * inter_dim, dtype=torch.float32
        ).reshape(dim, inter_dim)
    return state


def test_from_hf_groups_kimi_w1_w2_w3_experts(adapter, moe_config):
    hf_state = _make_hf_expert_state(
        moe_config.n_routed_experts,
        moe_config.moe_inter_dim,
        moe_config.dim,
    )

    native = adapter.from_hf(hf_state)

    gate_up_key = "model.layers.1.mlp.experts.gate_and_up_projs"
    down_key = "model.layers.1.mlp.experts.down_projs"
    assert gate_up_key in native
    assert down_key in native

    gate_up = native[gate_up_key]
    down = native[down_key]
    assert gate_up.shape == (moe_config.n_routed_experts, moe_config.dim, 2 * moe_config.moe_inter_dim)
    assert down.shape == (moe_config.n_routed_experts, moe_config.moe_inter_dim, moe_config.dim)

    w1 = hf_state["model.layers.1.block_sparse_moe.experts.0.w1.weight"]
    w3 = hf_state["model.layers.1.block_sparse_moe.experts.0.w3.weight"]
    w2 = hf_state["model.layers.1.block_sparse_moe.experts.0.w2.weight"]
    assert torch.equal(gate_up[0, :, : moe_config.moe_inter_dim], w1.T.to(gate_up.dtype))
    assert torch.equal(gate_up[0, :, moe_config.moe_inter_dim :], w3.T.to(gate_up.dtype))
    assert torch.equal(down[0], w2.T.to(down.dtype))


def test_from_hf_upcasts_kda_and_router_fp32_keys(adapter):
    hf_state = {
        "model.layers.0.self_attn.A_log": torch.ones(1, 1, 2, 1, dtype=torch.bfloat16),
        "model.layers.0.self_attn.dt_bias": torch.ones(8, dtype=torch.bfloat16),
        "model.layers.1.block_sparse_moe.gate.e_score_correction_bias": torch.ones(3, dtype=torch.bfloat16),
    }

    native = adapter.from_hf(hf_state)

    assert native["model.layers.0.self_attn._fp32_params.A_log"].dtype is torch.float32
    assert native["model.layers.0.self_attn._fp32_params.dt_bias"].dtype is torch.float32
    assert native["model.layers.1.mlp.gate.e_score_correction_bias"].dtype is torch.float32


def test_to_hf_strips_kda_fp32_holder(adapter):
    native = {
        "model.layers.0.self_attn._fp32_params.A_log": torch.ones(1, 1, 2, 1, dtype=torch.float32),
        "model.layers.0.self_attn._fp32_params.dt_bias": torch.ones(8, dtype=torch.float32),
    }

    hf_state = adapter.to_hf(native)

    assert "model.layers.0.self_attn.A_log" in hf_state
    assert "model.layers.0.self_attn.dt_bias" in hf_state
    assert "model.layers.0.self_attn._fp32_params.A_log" not in hf_state
    assert "model.layers.0.self_attn._fp32_params.dt_bias" not in hf_state


def test_to_hf_splits_grouped_experts_back_to_kimi_names(adapter, moe_config):
    native = {
        "model.layers.1.mlp.experts.gate_and_up_projs": torch.randn(
            moe_config.n_routed_experts,
            moe_config.dim,
            2 * moe_config.moe_inter_dim,
        ),
        "model.layers.1.mlp.experts.down_projs": torch.randn(
            moe_config.n_routed_experts,
            moe_config.moe_inter_dim,
            moe_config.dim,
        ),
    }

    hf_state = adapter.to_hf(native)

    assert "model.layers.1.block_sparse_moe.experts.0.w1.weight" in hf_state
    assert "model.layers.1.block_sparse_moe.experts.0.w2.weight" in hf_state
    assert "model.layers.1.block_sparse_moe.experts.0.w3.weight" in hf_state
    assert "model.layers.1.block_sparse_moe.experts.0.gate_proj.weight" not in hf_state

    assert hf_state["model.layers.1.block_sparse_moe.experts.0.w1.weight"].shape == (
        moe_config.moe_inter_dim,
        moe_config.dim,
    )
    assert hf_state["model.layers.1.block_sparse_moe.experts.0.w2.weight"].shape == (
        moe_config.dim,
        moe_config.moe_inter_dim,
    )
    assert hf_state["model.layers.1.block_sparse_moe.experts.0.w3.weight"].shape == (
        moe_config.moe_inter_dim,
        moe_config.dim,
    )


def test_convert_single_tensor_to_hf_respects_exclude_regex(adapter):
    tensor = torch.randn(1)

    result = adapter.convert_single_tensor_to_hf(
        "model.layers.1.block_sparse_moe.gate.weight",
        tensor,
        exclude_key_regex=r".*gate\.weight$",
    )

    assert result == []


def test_split_experts_weights_dtensor_replicate_returns_all_experts(adapter, monkeypatch):
    _patch_fake_dtensor(monkeypatch)
    local_tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    weight = _FakeDTensor(
        local_tensor=local_tensor,
        placement=Replicate(),
        mesh=_FakeMesh(rank=1, size=2, names=("tp",)),
    )

    split = adapter._split_experts_weights(weight, n_experts=4)

    assert torch.equal(torch.stack(split), local_tensor)
    assert adapter._last_expert_ids == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("rank", "expected_ids"),
    [
        (0, [0, 1, 2]),
        (1, [3, 4]),
    ],
)
def test_split_experts_weights_dtensor_shard_uses_uneven_expert_distribution(
    adapter,
    monkeypatch,
    rank,
    expected_ids,
):
    _patch_fake_dtensor(monkeypatch)
    local_tensor = torch.arange(len(expected_ids) * 2, dtype=torch.float32).reshape(len(expected_ids), 2)
    weight = _FakeDTensor(
        local_tensor=local_tensor,
        placement=Shard(0),
        mesh=_FakeMesh(rank=rank, size=2, names=("tp",)),
    )

    split = adapter._split_experts_weights(weight, n_experts=5)

    assert torch.equal(torch.stack(split), local_tensor)
    assert adapter._last_expert_ids == expected_ids


def test_split_experts_weights_dtensor_shard_uses_active_ep_submesh(adapter, monkeypatch):
    _patch_fake_dtensor(monkeypatch)
    active_mesh = _FakeMesh(rank=0, size=99, names=("dp", "ep"))
    ep_mesh = _FakeMesh(rank=2, size=3, names=("ep",))
    calls = []

    def fake_get_submesh(mesh, dims):
        calls.append((mesh, dims))
        return ep_mesh

    monkeypatch.setattr(state_dict_utils, "get_submesh", fake_get_submesh)
    adapter._active_device_mesh = active_mesh
    local_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    weight = _FakeDTensor(
        local_tensor=local_tensor,
        placement=Shard(0),
        mesh=_FakeMesh(rank=0, size=99, names=("tp",)),
    )

    split = adapter._split_experts_weights(weight, n_experts=8)

    assert torch.equal(torch.stack(split), local_tensor)
    assert adapter._last_expert_ids == [6, 7]
    assert calls == [(active_mesh, ("ep",))]


def test_split_experts_weights_dtensor_shard_validates_local_expert_count(adapter, monkeypatch):
    _patch_fake_dtensor(monkeypatch)
    weight = _FakeDTensor(
        local_tensor=torch.zeros(1, 2),
        placement=Shard(0),
        mesh=_FakeMesh(rank=0, size=2, names=("tp",)),
    )

    with pytest.raises(ValueError, match="Expected local Kimi expert tensor first dimension to be 3"):
        adapter._split_experts_weights(weight, n_experts=5)


def test_moe_children_map_between_block_sparse_moe_and_mlp(adapter):
    """The block names its feed-forward ``mlp``; the checkpoint says ``block_sparse_moe``."""
    hf_state = {
        "model.layers.1.block_sparse_moe.gate.weight": torch.ones(3, 4),
        "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight": torch.ones(4, 3),
        # A dense layer keeps ``mlp`` on both sides and must survive untouched.
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 3),
    }

    native = adapter.from_hf(hf_state)

    assert "model.layers.1.mlp.gate.weight" in native
    assert "model.layers.1.mlp.shared_experts.down_proj.weight" in native
    assert "model.layers.0.mlp.down_proj.weight" in native

    round_tripped = adapter.to_hf(native)

    assert set(round_tripped) == set(hf_state)


_PEFT_PREFIX = "base_model.model."


def _peft_lora_state(n_experts: int, dim: int, inter_dim: int, rank: int = 2) -> dict[str, torch.Tensor]:
    """LoRA tensors as ``ModelState`` emits them on a PEFT save.

    Grouped-expert shapes follow ``GroupedExpertsLoRA``: ``*_A`` is [experts, in_features, rank] and
    ``*_B`` is [experts, rank, out_features]. Shared-expert and attention LoRA tensors are plain
    ``nn.Linear`` weights: ``lora_A`` is [rank, in_features] and ``lora_B`` is [out_features, rank].
    """
    experts = f"{_PEFT_PREFIX}model.layers.1.mlp.experts"
    return {
        f"{experts}.lora_gate_and_up_A": torch.randn(n_experts, dim, rank),
        f"{experts}.lora_gate_and_up_B": torch.randn(n_experts, rank, 2 * inter_dim),
        f"{experts}.lora_down_A": torch.randn(n_experts, inter_dim, rank),
        f"{experts}.lora_down_B": torch.randn(n_experts, rank, dim),
        f"{_PEFT_PREFIX}model.layers.1.mlp.shared_experts.gate_proj.lora_A.weight": torch.randn(rank, dim),
        f"{_PEFT_PREFIX}model.layers.1.self_attn.q_proj.lora_A.weight": torch.randn(rank, dim),
    }


def test_peft_lora_keys_match_base_export_layout(adapter, moe_config):
    """Adapter keys must name the same modules as the base weights exported by the same adapter."""
    n_experts, dim, inter_dim = moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim
    base = adapter.to_hf(
        {
            "model.layers.1.mlp.experts.gate_and_up_projs": torch.randn(n_experts, dim, 2 * inter_dim),
            "model.layers.1.mlp.experts.down_projs": torch.randn(n_experts, inter_dim, dim),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(inter_dim, dim),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(dim, dim),
        }
    )
    base_modules = {key.removesuffix(".weight") for key in base}

    lora = adapter.to_hf(_peft_lora_state(n_experts, dim, inter_dim))

    for key in lora:
        assert key.startswith(_PEFT_PREFIX), key
        assert key.removeprefix(_PEFT_PREFIX).rsplit(".lora_", 1)[0] in base_modules, key
    for expert in range(n_experts):
        for proj in ("w1", "w3", "w2"):
            for half in ("A", "B"):
                key = f"{_PEFT_PREFIX}model.layers.1.block_sparse_moe.experts.{expert}.{proj}.lora_{half}.weight"
                assert key in lora, key


def test_peft_lora_keys_round_trip_through_from_hf(adapter, moe_config):
    """Loading a saved adapter back rebuilds the grouped native LoRA keys and values."""
    native = _peft_lora_state(moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim)

    hf = adapter.to_hf({key: value.clone() for key, value in native.items()})
    back = adapter.from_hf(dict(hf))

    assert set(back) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(back[key].to(value.dtype), value, rtol=0, atol=0)


def test_from_hf_groups_hf_layout_peft_expert_lora_keys(adapter, moe_config):
    """A saved adapter in the HF layout loads back into the grouped native LoRA tensors."""
    n_experts, dim, inter_dim, rank = moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim, 2
    hf_experts = f"{_PEFT_PREFIX}model.layers.1.block_sparse_moe.experts"
    hf_state = {}
    for expert in range(n_experts):
        hf_state[f"{hf_experts}.{expert}.w1.lora_A.weight"] = torch.randn(rank, dim)
        hf_state[f"{hf_experts}.{expert}.w1.lora_B.weight"] = torch.randn(inter_dim, rank)
        hf_state[f"{hf_experts}.{expert}.w3.lora_A.weight"] = hf_state[f"{hf_experts}.{expert}.w1.lora_A.weight"]
        hf_state[f"{hf_experts}.{expert}.w3.lora_B.weight"] = torch.randn(inter_dim, rank)
        hf_state[f"{hf_experts}.{expert}.w2.lora_A.weight"] = torch.randn(rank, inter_dim)
        hf_state[f"{hf_experts}.{expert}.w2.lora_B.weight"] = torch.randn(dim, rank)

    native = adapter.from_hf(dict(hf_state))

    experts = f"{_PEFT_PREFIX}model.layers.1.mlp.experts"
    assert set(native) == {
        f"{experts}.lora_gate_and_up_A",
        f"{experts}.lora_gate_and_up_B",
        f"{experts}.lora_down_A",
        f"{experts}.lora_down_B",
    }
    # The native layout is the transpose of the HF per-expert weight: [experts, in_features, out_features].
    for expert in range(n_experts):
        torch.testing.assert_close(
            native[f"{experts}.lora_down_B"][expert], hf_state[f"{hf_experts}.{expert}.w2.lora_B.weight"].T
        )
        torch.testing.assert_close(
            native[f"{experts}.lora_gate_and_up_B"][expert],
            torch.cat(
                [
                    hf_state[f"{hf_experts}.{expert}.w1.lora_B.weight"].T,
                    hf_state[f"{hf_experts}.{expert}.w3.lora_B.weight"].T,
                ],
                dim=-1,
            ),
        )


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ("model.layers.1.mlp.experts.0.gate_proj", "model.layers.1.block_sparse_moe.experts.0.w1"),
        ("model.layers.1.mlp.experts.2.up_proj", "model.layers.1.block_sparse_moe.experts.2.w3"),
        ("model.layers.12.mlp.experts.255.down_proj", "model.layers.12.block_sparse_moe.experts.255.w2"),
        ("model.layers.1.mlp.shared_experts.gate_proj", "model.layers.1.block_sparse_moe.shared_experts.gate_proj"),
        ("model.layers.1.mlp.gate", "model.layers.1.block_sparse_moe.gate"),
        # Dense feed-forward and attention modules keep their names.
        ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj"),
        ("model.layers.0.mlp.down_proj", "model.layers.0.mlp.down_proj"),
        ("model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.q_proj"),
    ],
)
def test_peft_target_modules_use_hf_layout(adapter, native, expected):
    assert adapter.map_peft_target_module_to_hf(native) == expected
