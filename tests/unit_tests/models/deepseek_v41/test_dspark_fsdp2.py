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

"""Real CPU/Gloo regression for the mixed-dtype DeepSeek V4.1 DSpark FSDP path."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed.activation_checkpointing import apply_submodule_checkpointing
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.dspark import DeepseekV41DSparkModel
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

# Keep a hard watchdog for spawned workers; each distributed test declares its runtime budget below.
pytestmark = pytest.mark.timeout(60)

_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(dtype: str = "bfloat16") -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=[0, 0, 0, 0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        num_nextn_predict_layers=3,
        dspark_noise_token_id=31,
        dspark_target_layer_ids=[0, 1],
        dspark_markov_rank=4,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dtype=dtype,
    )


def _fully_shard_model(model: DeepseekV41DSparkModel, mesh: DeviceMesh, dtype: torch.dtype) -> None:
    """Apply the same layer and root FSDP2 wrapping used by the DSpark recipe."""
    policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard_by_dtype(
            layer,
            mesh=mesh,
            mp_policy=policy,
            offload_policy=None,
            fp32_compute_module_names=tuple(model._keep_in_fp32_modules_strict),
        )
    fully_shard(model, mesh=mesh, mp_policy=policy)


def _worker(rank: int, port: int, activation_checkpointing: bool, checkpoint_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        torch.manual_seed(17)
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        model = DeepseekV41DSparkModel(
            _config(),
            BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"),
            num_anchors=1,
            enable_confidence_head=True,
        )
        cast_model_to_dtype(model, torch.bfloat16)
        model.set_embedding_head_trainable(False)
        if activation_checkpointing:
            apply_submodule_checkpointing(list(model.layers), has_kv_sharing=False)
        _fully_shard_model(model, mesh, torch.bfloat16)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        target_hidden_states = torch.randn(1, 8, 32, dtype=torch.bfloat16)
        output = model(input_ids, target_hidden_states, torch.ones_like(input_ids))
        loss = output.draft_logits.square().mean() + output.confidence_pred.square().mean()
        loss.backward()

        assert torch.isfinite(loss)
        assert model.mtp[0].main_proj.weight.grad is not None
        assert model.mtp[-1].markov_head.head.weight.grad is not None
        assert model.mtp[-1].confidence_head.proj.weight.grad is not None
        torch.optim.SGD(model.parameters(), lr=0.01).step()

        # Exercise the real HF/DCP save and resume path, including rank-local
        # expert splitting on the named, EP-free FSDP mesh.
        expert_weight = model.mtp[0].ffn.experts.gate_and_up_projs
        assert isinstance(expert_weight, DTensor)
        assert expert_weight.device_mesh.mesh_dim_names == ("dp",)
        expected = {
            key: (value.to_local() if isinstance(value, DTensor) else value).clone()
            for key, value in model.state_dict().items()
        }
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=checkpoint_dir,
                model_save_format="safetensors",
                save_consolidated=False,
                model_cache_dir=str(Path(checkpoint_dir) / "cache"),
                model_repo_id="test/deepseek-v41-dspark",
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=dist.group.WORLD,
        )
        try:
            trained = Path(checkpoint_dir) / "trained"
            checkpointer.save_model(model, str(trained))
            for value in model.state_dict().values():
                (value.to_local() if isinstance(value, DTensor) else value).zero_()
            checkpointer.load_model(model, str(trained / "model"))
            for key, value in model.state_dict().items():
                local_value = value.to_local() if isinstance(value, DTensor) else value
                torch.testing.assert_close(local_value, expected[key], rtol=0, atol=0)
        finally:
            checkpointer.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    45,
    hard_timeout=60,
    reason="Spawns two FSDP workers, compiles draft forward/backward, and saves and restores a sharded checkpoint.",
)
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_bf16_dspark_fsdp_forward_backward_checkpoint_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, activation_checkpointing: bool
) -> None:
    # Other unit-test modules disable compilation during collection. Spawned workers
    # import the fullgraph MoE activation afresh, so explicitly enable its compiler.
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    mp.spawn(_worker, args=(_free_port(), activation_checkpointing, str(tmp_path)), nprocs=_WORLD_SIZE, join=True)


def _parity_worker(rank: int, port: int) -> None:
    """Compare two-rank FSDP2 gradients and updates with an FP32 reference."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        torch.manual_seed(17)
        backend = BackendConfig(
            attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"
        )
        reference = DeepseekV41DSparkModel(
            _config(dtype="float32"), backend, num_anchors=1, enable_confidence_head=True
        )
        reference.set_embedding_head_trainable(False)
        model = DeepseekV41DSparkModel(_config(dtype="float32"), backend, num_anchors=1, enable_confidence_head=True)
        model.load_state_dict(reference.state_dict())
        model.set_embedding_head_trainable(False)

        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        _fully_shard_model(model, mesh, torch.float32)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        target_hidden_states = torch.randn(1, 8, 32)
        loss_mask = torch.ones_like(input_ids)
        torch.manual_seed(29)
        expected_output = reference(input_ids, target_hidden_states, loss_mask)
        assert expected_output.confidence_pred is not None
        expected_loss = expected_output.draft_logits.square().mean() + expected_output.confidence_pred.square().mean()
        expected_loss.backward()

        torch.manual_seed(29)
        actual_output = model(input_ids, target_hidden_states, loss_mask)
        assert actual_output.confidence_pred is not None
        actual_loss = actual_output.draft_logits.square().mean() + actual_output.confidence_pred.square().mean()
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)

        expected_parameters = dict(reference.named_parameters())
        actual_parameters = dict(model.named_parameters())
        assert actual_parameters.keys() == expected_parameters.keys()
        for name, parameter in actual_parameters.items():
            expected_gradient = expected_parameters[name].grad
            if expected_gradient is None:
                assert parameter.grad is None
            else:
                assert parameter.grad is not None
                torch.testing.assert_close(parameter.grad.full_tensor(), expected_gradient, rtol=1e-5, atol=1e-6)

        expected_norm = scale_grads_and_clip_grad_norm(0.25, [reference])
        actual_norm = scale_grads_and_clip_grad_norm(0.25, [model])
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)
        for name, parameter in actual_parameters.items():
            expected_gradient = expected_parameters[name].grad
            if expected_gradient is not None:
                torch.testing.assert_close(parameter.grad.full_tensor(), expected_gradient, rtol=1e-5, atol=1e-6)

        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        expected_optimizer.step()
        actual_optimizer.step()
        for name, parameter in actual_parameters.items():
            torch.testing.assert_close(parameter.full_tensor(), expected_parameters[name], rtol=1e-5, atol=1e-6)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    45,
    hard_timeout=60,
    reason="Spawns two FSDP workers and compares compiled draft gradients, clipping, and updates with an FP32 reference.",
)
def test_fp32_dspark_fsdp_gradient_and_step_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    # Restore the caller's environment after spawning, without changing other tests.
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    mp.spawn(_parity_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
