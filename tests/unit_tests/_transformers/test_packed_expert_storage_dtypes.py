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

"""Packed expert storage keeps its registered dtypes through ``apply_model_infrastructure``.

The infrastructure runs the generic frozen-parameter casts: FSDP2 storage preparation and the post-wrap frozen
cast. The codes and scales of mxfp4 / nvfp4 experts are the weight format; casting the nvfp4 fp32 row scales to
bf16 would change the frozen base weights. The experts start materialized, so the packed conversion quantizes them
in place; one rank keeps plain parameters, so the post-wrap cast reaches them. Expert parallelism is left to the
model tests: EP shards are loaded from a packed checkpoint through the model's state-dict adapter, and materialized
e8m0 scales cannot be scattered over NCCL.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen3MoeConfig

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure, instantiate_infrastructure
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

# Storage format -> packed parameter suffix -> registered dtype.
_PACKED_DTYPES = {
    "mxfp4": {"_packed": torch.int8, "_scales": torch.float8_e8m0fnu},
    "nvfp4": {"_nvfp4": torch.uint8, "_scales": torch.float8_e4m3fn, "_global": torch.float32},
}
_NUM_LAYERS = 2
_PROJECTIONS = ("gate_and_up_projs", "down_projs")


def _build_model(setup: DistributedSetup, device: torch.device, storage_format: str) -> torch.nn.Module:
    config = Qwen3MoeConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        use_cache=False,
        torch_dtype="float32",
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch_mm",
        dispatcher="torch",
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
    )
    with device:
        model = Qwen3MoeForCausalLM(config, backend=backend)
    model.initialize_weights(buffer_device=device, dtype=torch.float32)
    model_wrapper, _, parallelize_fn, _ = instantiate_infrastructure(
        distributed_config=setup.strategy_config,
        moe_parallel_config=setup.moe_parallel_config,
        activation_checkpointing=setup.activation_checkpointing,
        device=device,
        mesh=setup.mesh_context,
    )
    return apply_model_infrastructure(
        model,
        is_meta_device=False,
        device=device,
        model_wrapper=model_wrapper,
        mesh=setup.mesh_context,
        peft_config=PeftConfig(target_modules=["*"], dim=8, alpha=16, dropout=0.0, expert_weight_format=storage_format),
        parallelize_fn=parallelize_fn,
        load_base_model=False,
    )


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=1)
    try:
        setup = DistributedSetup.build(strategy="fsdp2", parallelism_sizes=ParallelismSizes(ep_size=1))
        for storage_format, expected in _PACKED_DTYPES.items():
            model = _build_model(setup, torch.device("cuda", rank), storage_format)
            packed_names = tuple(f"{projection}{suffix}" for projection in _PROJECTIONS for suffix in expected)
            dtypes = {name: p.dtype for name, p in model.named_parameters() if name.endswith(packed_names)}
            assert len(dtypes) == _NUM_LAYERS * len(packed_names), (storage_format, sorted(dtypes))
            for name, dtype in dtypes.items():
                assert dtype == expected["_" + name.rsplit("_", 1)[1]], (storage_format, name, dtype)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device for the FSDP2 infrastructure path")
def test_infrastructure_keeps_packed_expert_dtypes(tmp_path: Path) -> None:
    """With unsharded parameters, the post-wrap frozen cast must leave the packed codes and scales alone."""
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=1, join=True)
