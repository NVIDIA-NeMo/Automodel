# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checkpoint targets must retain CPU storage even on a CUDA device mesh."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import (
    DeepseekV41StateDictAdapter,
    dequantize_checkpoint_weight,
)


class TinyEPExperts(torch.nn.Module):
    def __init__(self, mesh):
        super().__init__()
        from torch.distributed.tensor import distribute_tensor

        self.weight = torch.nn.Parameter(distribute_tensor(torch.ones(4, 4, 4, device="cuda"), mesh, [Shard(0)]))

    def forward(self, x):
        weight = self.weight.to_local()
        assert weight.device.type == "cuda"
        return x @ weight[0]


def probe_singleton_expert_offload(rank):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("expert_fsdp", "ep"))
    model = TinyEPExperts(mesh["ep"])
    fully_shard(
        model,
        mesh=mesh["expert_fsdp"],
        shard_placement_fn=lambda p: Shard(1),
        offload_policy=CPUOffloadPolicy(pin_memory=False),
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        reshard_after_forward=True,
    )
    model.cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(4, 4, device="cuda", dtype=torch.bfloat16)).float().sum().backward()
        assert model.weight.grad.to_local().device.type == "cpu"
        optimizer.step()
        assert model.weight.to_local().device.type == "cpu"
        for state in optimizer.state.values():
            assert state["exp_avg"].to_local().device.type == "cpu"
        assert model.weight.to_local()[0].mean() < 1
    print(f"CPU_SINGLETON_EP_FORWARD_BACKWARD_ADAM_PASSED rank={rank}", flush=True)


def probe_expert_checkpoint_roundtrip(rank):
    from torch.distributed.tensor import distribute_tensor

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
    from nemo_automodel.components.moe.config import MoEConfig

    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("ep_shard", "ep"))
    text = DeepseekV41TextConfig(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        dtype="float32",
    )
    moe = MoEConfig(
        dim=32,
        inter_dim=32,
        moe_inter_dim=32,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        norm_topk_prob=True,
        dtype=torch.float32,
    )
    adapter = DeepseekV41StateDictAdapter(
        DeepseekV41Config(text_config=text),
        moe,
        BackendConfig(linear="torch", attn="sdpa", dispatcher="torch", experts="torch_mm"),
        dtype=torch.float32,
    )
    native = {
        f"model.layers.0.ffn.experts.{name}": distribute_tensor(
            torch.ones(shape, device="cuda"), mesh, [Shard(1), Shard(0)]
        ).cpu()
        for name, shape in [("gate_and_up_projs", (2, 32, 64)), ("down_projs", (2, 32, 32))]
    }
    targets = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    for name, tensor in targets.items():
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        assert local.device.type == "cpu"
        if name.endswith(".scale"):
            local.view(torch.uint8).fill_(127)
        else:
            local.fill_(0x11)
    loaded = adapter.from_hf(targets, device_mesh=mesh)
    for name, target in native.items():
        assert loaded[name].device_mesh == target.device_mesh
        assert loaded[name].placements == target.placements
        target.copy_(loaded[name])
        torch.testing.assert_close(target.to_local(), torch.full_like(target.to_local(), 0.5), rtol=0, atol=0)
    print(f"CPU_SINGLETON_EP_CHECKPOINT_ROUNDTRIP_PASSED rank={rank}", flush=True)


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        probe_expert_checkpoint_roundtrip(rank)
        probe_singleton_expert_offload(rank)
        mesh = init_device_mesh("cuda", (2,))
        # Setup must move a DTensor buffer without corrupting wrapper metadata.
        broken = DTensor.from_local(torch.ones(8, device="cuda"), mesh, [Shard(0)])
        broken.data = broken.data.cpu()
        try:
            broken.detach()
        except RuntimeError as error:
            assert "storage" in str(error), str(error)
            print(f"CPU_BUFFER_DATA_REPRODUCED rank={rank}", flush=True)
        else:
            raise AssertionError("Expected unsafe .data move to reproduce storage error")
        module = torch.nn.Module()
        module.register_buffer("bias", DTensor.from_local(torch.ones(8, device="cuda"), mesh, [Shard(0)]))
        for buffer in module.buffers():
            torch.utils.swap_tensors(buffer, buffer.cpu())
        module.to("cpu")
        assert module.state_dict()["bias"].to_local().device.type == "cpu"
        print(f"CPU_BUFFER_STATE_DICT_PASSED rank={rank}", flush=True)
        value = DTensor.from_local(torch.ones(8, 64, device="cuda"), mesh, [Shard(0)]).cpu()
        assert value.to_local().device.type == "cpu"
        for name, expected in [("layers.0.attn.wq_a.weight", 1.5), ("layers.0.ffn.experts.0.w1.weight", 0.5)]:
            targets = DeepseekV41StateDictAdapter._quantized_load_targets(name, value)
            weight, scale = [tensor for _, tensor in targets]
            raw = weight.to_local()
            local_scale = scale.to_local() if isinstance(scale, DTensor) else scale
            assert raw.device.type == local_scale.device.type == "cpu"
            if raw.dtype == torch.int8:
                raw.fill_(0x11)  # Two packed E2M1 values, both 0.5.
            else:
                raw.copy_(torch.full(raw.shape, 1.5).to(raw.dtype))
            local_scale.view(torch.uint8).fill_(127)
            decoded = dequantize_checkpoint_weight(weight, scale, dtype=torch.float32)
            assert decoded.device_mesh == mesh and decoded.placements == value.placements
            assert decoded.to_local().device.type == "cpu"
            torch.testing.assert_close(decoded.to_local(), torch.full((8, 64), expected), rtol=0, atol=0)
        print(f"CPU_CHECKPOINT_CUDA_MESH_PARITY_PASSED rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cpu_checkpoint_storage_on_cuda_mesh(tmp_path) -> None:
    mp.spawn(_worker, args=(str(tmp_path / "checkpoint-mesh"),), nprocs=2, join=True)
