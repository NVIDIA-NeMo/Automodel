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

"""Dist-MoE checkpoints must load through the production loader under EP x expert FSDP.

Four Gloo ranks form ``ep_shard=2 x ep=2``, so expert FSDP shards the native
feature dimension. The HF checkpoint is written by Transformers' own Qwen3-MoE
model and is the independent oracle for the loaded canonical expert tensors.
"""

import os
import socket
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM
from nemo_automodel.components.moe.dist_moe_experts import GroupedExpertsDistMoE
from nemo_automodel.components.moe.parallelizer import parallelize_model

_WORLD_SIZE = 4
_EXPERTS = 4


def _config() -> transformers.Qwen3MoeConfig:
    return transformers.Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_experts=_EXPERTS,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        torch_dtype="float32",
    )


def _build_model(seed: int) -> tuple[Qwen3MoeForCausalLM, MeshContext]:
    """Construct a Dist-MoE Qwen3-MoE model and apply the production EP/FSDP parallelizer."""
    torch.manual_seed(seed)
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch_mm",
        dispatcher="dist_moe",
        dist_moe={"max_tokens_per_rank": 4},
        enable_hf_state_dict_adapter=True,
    )
    model = Qwen3MoeForCausalLM(_config(), backend=backend).to(torch.float32)
    mesh = MeshContext.build(FSDP2Config(), ParallelismSizes(dp_size=_WORLD_SIZE, ep_size=2), world_size=_WORLD_SIZE)
    # The shared Dist-MoE kernel context needs the CUDA package; parameter placement does not.
    with patch.object(GroupedExpertsDistMoE, "initialize_shared_context"):
        parallelize_model(model, mesh.device_mesh, mesh.moe_mesh, **mesh.parallelize_axis_kwargs())
    return model, mesh


def _checkpointer(root: Path, rank: int, mesh: MeshContext) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(root / "checkpoints"),
            model_cache_dir=str(root / "cache"),
            model_repo_id="test/qwen3-moe",
            model_save_format="safetensors",
            save_consolidated=False,
        ),
        dp_rank=rank,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=mesh.moe_mesh,
    )


def _experts(model: Qwen3MoeForCausalLM) -> list[GroupedExpertsDistMoE]:
    experts = [module for module in model.modules() if isinstance(module, GroupedExpertsDistMoE)]
    assert len(experts) == 2
    return experts


def _assert_native_layout(experts: GroupedExpertsDistMoE) -> None:
    for parameter in (experts.routed_gate_up_weights, experts.routed_down_weights):
        assert isinstance(parameter, DTensor)
        assert parameter.device_mesh.mesh_dim_names == ("ep_shard", "ep")
        assert parameter.placements == (Shard(1), Shard(0))


def _worker(rank: int, port: int, root: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        root_path = Path(root)
        hf_path = root_path / "hf"
        if rank == 0:
            torch.manual_seed(1234)
            transformers.Qwen3MoeForCausalLM(_config()).to(torch.float32).save_pretrained(hf_path)
        dist.barrier()
        reference = transformers.Qwen3MoeForCausalLM.from_pretrained(hf_path, torch_dtype=torch.float32)

        model, mesh = _build_model(seed=5678)
        checkpointer = _checkpointer(root_path, rank, mesh)
        checkpointer.load_model(model, str(hf_path), is_init_step=True)

        for layer, experts in enumerate(_experts(model)):
            _assert_native_layout(experts)
            hf_experts = reference.model.layers[layer].mlp.experts
            # Transformers stores [experts, 2 * intermediate, hidden] (gate rows,
            # then up rows) and [experts, hidden, intermediate].
            expected_gate_up = hf_experts.gate_up_proj.detach().transpose(-1, -2)
            expected_down = hf_experts.down_proj.detach().transpose(-1, -2)
            torch.testing.assert_close(experts.gate_and_up_projs.full_tensor(), expected_gate_up, rtol=0, atol=0)
            torch.testing.assert_close(experts.down_projs.full_tensor(), expected_down, rtol=0, atol=0)

        # Native save followed by a load into differently initialized parameters.
        checkpointer.save_model(model, str(root_path / "saved"))
        dist.barrier()
        restored, _ = _build_model(seed=91011)
        _checkpointer(root_path, rank, mesh).load_model(restored, str(root_path / "saved" / "model"))
        for experts, restored_experts in zip(_experts(model), _experts(restored)):
            _assert_native_layout(restored_experts)
            for name in ("routed_gate_up_weights", "routed_down_weights"):
                torch.testing.assert_close(
                    getattr(restored_experts, name).to_local(), getattr(experts, name).to_local(), rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


def test_dist_moe_hf_checkpoint_loads_under_ep_and_expert_fsdp(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=_WORLD_SIZE, join=True)
