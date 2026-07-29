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

"""Two-rank regression for ERNIE's strict-fp32 router under EP and FSDP2."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie4_5.model import Ernie4_5_MoeForCausalLM
from nemo_automodel.components.moe.parallelizer import parallelize_model

_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tiny_config() -> Ernie4_5_MoeConfig:
    return Ernie4_5_MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        moe_intermediate_size=16,
        moe_k=2,
        moe_num_experts=4,
        moe_num_shared_experts=0,
        moe_layer_start_index=1,
        moe_layer_end_index=1,
        moe_layer_interval=1,
        router_aux_loss_coef=0.0,
        torch_dtype=torch.bfloat16,
    )


def _worker(rank: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)

        torch.manual_seed(1234)
        model = Ernie4_5_MoeForCausalLM(
            _tiny_config(),
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                experts="torch",
                dispatcher="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
            ),
        ).cuda(rank)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.02)

        moe = model.model.layers["1"].mlp
        router_input = torch.randn(8, model.config.hidden_size, device=rank, dtype=torch.bfloat16)
        token_mask = torch.ones(router_input.shape[0], device=rank, dtype=torch.bool)
        with torch.no_grad():
            reference_scores = F.linear(router_input.float(), moe.gate.weight)
            reference_probabilities = reference_scores.softmax(dim=-1)
            reference_choice_scores = reference_probabilities + moe.gate.e_score_correction_bias
            reference_indices = reference_choice_scores.topk(moe.gate.topk, dim=-1).indices
            reference_weights = reference_probabilities.gather(1, reference_indices)
            reference_weights /= reference_weights.sum(dim=-1, keepdim=True) + 1e-20
            reference_weights = reference_weights.to(torch.bfloat16)
            actual_weights, actual_indices, _ = moe.gate(router_input, token_mask, cp_mesh=None)
        torch.testing.assert_close(actual_indices, reference_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual_weights, reference_weights, rtol=0, atol=0)

        reference_gate_weight = moe.gate.weight.detach().clone()
        reference_correction_bias = moe.gate.e_score_correction_bias.detach().clone()
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(dp_size=_WORLD_SIZE, ep_size=_WORLD_SIZE),
            world_size=_WORLD_SIZE,
        )
        parallelize_model(
            model,
            mesh_context.device_mesh,
            mesh_context.moe_mesh,
            activation_checkpointing=True,
            **mesh_context.parallelize_axis_kwargs(),
        )

        assert moe.gate.weight.dtype == torch.float32
        assert moe.gate.e_score_correction_bias.dtype == torch.float32
        assert moe.experts.gate_and_up_projs.dtype == torch.bfloat16

        gate_observation: dict[str, torch.Tensor] = {}

        def capture_gate_io(_module, inputs, output) -> None:
            gate_input = inputs[0]
            gate_weights, gate_indices, _ = output
            gate_observation["input"] = gate_input.detach().clone()
            gate_observation["weights"] = gate_weights.detach().clone()
            gate_observation["indices"] = gate_indices.detach().clone()

        input_ids = torch.tensor([[1, 2, 3, 4]], device=rank)
        gate_hook = moe.gate.register_forward_hook(capture_gate_io)
        try:
            logits = model(input_ids).logits
        finally:
            gate_hook.remove()

        captured_input = gate_observation["input"]
        captured_weights = gate_observation["weights"]
        captured_indices = gate_observation["indices"]
        if isinstance(captured_input, DTensor):
            captured_input = captured_input.to_local()
        if isinstance(captured_weights, DTensor):
            captured_weights = captured_weights.to_local()
        if isinstance(captured_indices, DTensor):
            captured_indices = captured_indices.to_local()

        with torch.no_grad():
            sharded_reference_scores = F.linear(captured_input.float(), reference_gate_weight)
            sharded_reference_probabilities = sharded_reference_scores.softmax(dim=-1)
            sharded_reference_choice_scores = sharded_reference_probabilities + reference_correction_bias
            sharded_reference_indices = sharded_reference_choice_scores.topk(moe.gate.topk, dim=-1).indices
            sharded_reference_weights = sharded_reference_probabilities.gather(1, sharded_reference_indices)
            sharded_reference_weights /= sharded_reference_weights.sum(dim=-1, keepdim=True) + 1e-20
            sharded_reference_weights = sharded_reference_weights.to(torch.bfloat16)
        torch.testing.assert_close(captured_indices, sharded_reference_indices, rtol=0, atol=0)
        torch.testing.assert_close(captured_weights, sharded_reference_weights, rtol=0, atol=0)

        loss = logits.float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()

        assert moe.gate.weight.grad is not None
        assert moe.gate.weight.grad.dtype == torch.float32
        assert torch.isfinite(moe.gate.weight.grad.to_local()).all()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires two CUDA devices",
)
def test_ernie_router_stays_fp32_through_ep_fsdp_forward_backward():
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
