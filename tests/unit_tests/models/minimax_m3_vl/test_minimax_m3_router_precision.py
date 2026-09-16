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

"""MiniMax-M3 router precision contract.

Released MiniMax-M3 checkpoints store the router gate weight in fp32 and the
correction bias as the same 1e-3-quantized fp32 lattice as MiniMax-M2.7, so
the checkpoint-faithful router keeps both tensors fp32 from allocation
through load (AMINT-286 pattern).
"""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from tests.unit_tests.models.minimax_m3_vl.conftest import TINY_CFG


def _first_moe_block(model):
    """First decoder block with a routed MoE mlp (M3 leads with dense layers)."""
    for block in model.model.layers.values():
        if hasattr(block.mlp, "gate"):
            return block
    raise AssertionError("tiny config produced no MoE layer")


def _cpu_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_router_fp32_contract_is_model_owned():
    config = MiniMaxM3VLTextConfig(torch_dtype="bfloat16", **TINY_CFG)
    model = MiniMaxM3SparseForCausalLM(config, backend=_cpu_backend()).eval()

    assert model.model.backend.gate_precision == torch.float32
    for cls in (MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration):
        assert "mlp.gate.weight" in cls._keep_in_fp32_modules_strict
        assert "mlp.gate.e_score_correction_bias" in cls._keep_in_fp32_modules_strict

    # After the model-wide bf16 cast, the fp32 contract keeps the router gate
    # parameter and bias fp32 while the rest of the model is bf16.
    model.initialize_weights(dtype=torch.bfloat16)
    moe_block = _first_moe_block(model)
    gate = moe_block.mlp.gate
    assert gate.weight.dtype == torch.float32
    assert gate.e_score_correction_bias.dtype == torch.float32
    assert moe_block.self_attn.q_proj.weight.dtype == torch.bfloat16


def test_gate_is_fp32_at_construction_for_fsdp_dtype_grouping():
    """The gate must be fp32 from allocation, before any init or checkpoint cast.

    FSDP shards the freshly constructed module: a bf16-allocated gate weight
    whose compute dtype is pinned fp32 shares its module with the fp32
    correction-bias buffer, which FSDP cannot isolate (the MiniMax-M2.7
    failure from pipeline 64344786).
    """
    import torch.distributed.fsdp as fsdp

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

    config = MiniMaxM3VLTextConfig(torch_dtype="bfloat16", **TINY_CFG)
    model = MiniMaxM3SparseForCausalLM(config, backend=_cpu_backend())
    block = _first_moe_block(model)

    # No initialize_weights on purpose: this is the state FSDP shards.
    assert block.mlp.gate.weight.dtype == torch.float32
    assert block.mlp.gate.e_score_correction_bias.dtype == torch.float32

    fully_shard_by_dtype(
        block,
        mesh=None,
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        offload_policy=None,
        fp32_compute_module_names=tuple(MiniMaxM3SparseForCausalLM._keep_in_fp32_modules_strict),
        fully_shard_fn=lambda *args, **kwargs: None,
    )
