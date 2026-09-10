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

"""Two-GPU regression coverage for CSA2 state, precision, and checkpointing."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.parallelizer import parallelize_model
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.components.utils.model_utils import freeze_deepseek_v4_indexer_params
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, tiny_config

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="FSDP2/EP regression requires two CUDA devices")


def _local_snapshot(value):
    """Copy an optimizer-state leaf without retaining distributed storage.

    Args:
        value: Scalar metadata or a tensor of arbitrary shape. DTensors use their
            per-rank local shard; ordinary tensors keep their existing layout.

    Returns:
        Unchanged scalar metadata or an independent tensor with the same local
        shape, dtype, and device.
    """
    if isinstance(value, DTensor):
        value = value.to_local()
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def _loss(model: torch.nn.Module, owner: int, micro: int, device: torch.device) -> torch.Tensor:
    """Return scalar FP32 next-token loss on a deterministic 11-token sample."""
    tokens = ((torch.arange(11, device=device) + owner * 23 + micro * 37) % 200 + 5).unsqueeze(0)
    logits = model(tokens).logits.float()
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())


def _compare(model: torch.nn.Module, reference: torch.nn.Module, *, gradients: bool, bf16: bool) -> None:
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        name = name.replace("_checkpoint_wrapped_module.", "")
        actual = parameter.grad if gradients else parameter
        target = expected[name].grad if gradients else expected[name]
        if target is None:
            assert actual is None, name
            continue
        assert actual is not None, name
        if isinstance(actual, DTensor):
            actual = actual.full_tensor()
        if gradients:
            atol, rtol = (0.003, 0.08) if bf16 else (2e-6, 2e-4)
        else:
            atol, rtol = (0.001, 0.02) if bf16 else (2e-6, 2e-4)
        torch.testing.assert_close(actual, target, atol=atol, rtol=rtol, msg=lambda message: f"{name}: {message}")


def _distributed_case(rank: int, directory: str, mode: str) -> None:
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    # Compare checkpoint restoration exactly, without atomic reduction order noise.
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda", rank)
    bf16 = mode == "bf16_ep_ac_tilelang"
    use_ep = mode != "fp32_dp"
    use_ac = mode != "fp32_dp"
    dtype = torch.bfloat16 if bf16 else torch.float32
    dist.init_process_group(
        "nccl", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        ep = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",)) if use_ep else None
        config = tiny_config(engram_trainable=True, kv_cache_fake_quant=False, index_n_heads=16, num_experts_per_tok=6)
        reference = build_tiny_model(config, attn="tilelang" if bf16 else "sdpa").to(device).train()
        model = (
            build_tiny_model(config, enable_fsdp_optimizations=True, attn="tilelang" if bf16 else "sdpa")
            .to(device)
            .train()
        )
        if bf16:
            cast_model_to_dtype(reference, dtype)
            cast_model_to_dtype(model, dtype)
        freeze_deepseek_v4_indexer_params(reference)
        freeze_deepseek_v4_indexer_params(model)
        parallelize_model(
            model,
            mesh,
            ep,
            dp_axis_names=("dp",),
            ep_axis_name="ep" if use_ep else None,
            activation_checkpointing=use_ac,
            ignore_router_for_ac=True,
            wrap_outer_model=True,
            reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype),
        )
        assert isinstance(model.model.layers["2"], FSDPModule)
        expert = model.model.layers["2"].mlp.experts.gate_and_up_projs
        assert isinstance(expert, DTensor)
        if use_ep:
            assert expert.to_local().shape[0] == config.n_routed_experts // 2
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, foreach=False)
        reference_optimizer = torch.optim.AdamW(
            (p for p in reference.parameters() if p.requires_grad), lr=1e-4, foreach=False
        )
        # Accumulation two, followed by a one-microbatch final window.
        for step, accumulation in enumerate((2, 1)):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_loss = torch.zeros((), device=device)
            for owner in range(2):
                for micro in range(accumulation):
                    loss = _loss(reference, owner, micro + step, device) / (2 * accumulation)
                    expected_loss += loss.detach()
                    loss.backward()
            model.prepare_for_grad_accumulation()
            actual_loss = torch.zeros((), device=device)
            for micro in range(accumulation):
                if micro == accumulation - 1:
                    model.prepare_for_final_backward()
                loss = _loss(model, rank, micro + step, device) / accumulation
                actual_loss += loss.detach()
                loss.backward()
            dist.all_reduce(actual_loss)
            actual_loss /= 2
            scale_grads_and_clip_grad_norm(
                None, [model], device_mesh=mesh, moe_mesh=ep, ep_axis_name="ep" if use_ep else None, dp_group_size=2
            )
            torch.testing.assert_close(actual_loss, expected_loss, atol=0.02 if bf16 else 1e-5, rtol=0)
            _compare(model, reference, gradients=True, bf16=bf16)
            optimizer.step()
            reference_optimizer.step()
            _compare(model, reference, gradients=False, bf16=bf16)

        state = {
            "model": ModelState(model, has_expert_parallelism=use_ep),
            "optimizer": OptimizerState(model, optimizer, has_expert_parallelism=use_ep),
        }
        dcp.save(state, checkpoint_id=Path(directory) / "checkpoint")
        saved = {name: _local_snapshot(p) for name, p in model.named_parameters()}
        saved_optimizer = tree_map(_local_snapshot, optimizer.state_dict())
        before = _loss(model, rank, 4, device).detach()
        with torch.no_grad():
            for parameter in model.parameters():
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                local.add_(0.25)
            for values in optimizer.state.values():
                for value in values.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        dcp.load(state, checkpoint_id=Path(directory) / "checkpoint")
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(_local_snapshot(parameter), saved[name], atol=0, rtol=0)
        torch.testing.assert_close(tree_map(_local_snapshot, optimizer.state_dict()), saved_optimizer, atol=0, rtol=0)
        torch.testing.assert_close(_loss(model, rank, 4, device).detach(), before, atol=0, rtol=0)
        optimizer.zero_grad(set_to_none=True)
        _loss(model, rank, 5, device).backward()
        scale_grads_and_clip_grad_norm(
            None, [model], device_mesh=mesh, moe_mesh=ep, ep_axis_name="ep" if use_ep else None, dp_group_size=2
        )
        optimizer.step()
        assert all(torch.isfinite(_local_snapshot(p)).all() for p in model.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["fp32_dp", "fp32_ep_ac", "bf16_ep_ac_tilelang"])
def test_fsdp_shared_state_gradients_and_checkpoint(tmp_path, monkeypatch, mode: str) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.multiprocessing.spawn(_distributed_case, args=(str(tmp_path), mode), nprocs=2, join=True)
