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

"""CPU/Gloo regression for the mixed-dtype DeepSeek V4.1 DSpark FSDP path."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.dspark import DeepseekV41DSparkModel

_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config() -> DeepseekV41TextConfig:
    config = DeepseekV41TextConfig(
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
        dtype="bfloat16",
    )
    config.dspark_num_anchors = 1
    config.dspark_enable_confidence_head = True
    return config


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        torch.manual_seed(17)
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        model = DeepseekV41DSparkModel(
            _config(),
            BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch"),
        )
        model.set_embedding_head_trainable(False)
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.layers:
            fully_shard_by_dtype(
                layer,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=None,
                fp32_compute_module_names=tuple(model._keep_in_fp32_modules_strict),
            )
        fully_shard(model, mesh=mesh, mp_policy=policy)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        target_hidden_states = torch.randn(1, 8, 32, dtype=torch.bfloat16)
        output = model(input_ids, target_hidden_states, torch.ones_like(input_ids))
        loss = output.draft_logits.square().mean() + output.confidence_pred.square().mean()
        loss.backward()

        assert torch.isfinite(loss)
        assert model.mtp[0].main_proj.weight.grad is not None
        assert model.mtp[-1].markov_head.head.weight.grad is not None
        assert model.mtp[-1].confidence_head.proj.weight.grad is not None
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_bf16_dspark_fsdp_forward_backward() -> None:
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
