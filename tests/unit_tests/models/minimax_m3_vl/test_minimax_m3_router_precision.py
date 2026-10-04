import dataclasses

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

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

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


@pytest.mark.skipif(
    "param_dtype_override_fn" not in {f.name for f in dataclasses.fields(MixedPrecisionPolicy)},
    reason="needs torch >= 2.15 MixedPrecisionPolicy.param_dtype_override_fn",
)
def test_gate_is_fp32_at_construction_for_fsdp_dtype_grouping(monkeypatch):
    """The gate is fp32 from allocation and stays fp32 compute inside the block's FSDP unit.

    FSDP2 keeps strict fp32 parameters in fp32 through a per-parameter override, which
    requires fp32 storage; under fp32 master weights the whole block is one unit.
    """
    import torch.distributed.fsdp as fsdp

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

    policies = []
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard",
        lambda _module, **kwargs: policies.append(kwargs["mp_policy"]),
    )

    config = MiniMaxM3VLTextConfig(torch_dtype="bfloat16", **TINY_CFG)
    model = MiniMaxM3SparseForCausalLM(config, backend=_cpu_backend())
    block = _first_moe_block(model)

    # No initialize_weights on purpose: this is the state FSDP shards.
    assert block.mlp.gate.weight.dtype == torch.float32
    assert block.mlp.gate.e_score_correction_bias.dtype == torch.float32

    fp32_config = MiniMaxM3VLTextConfig(torch_dtype="float32", **TINY_CFG)
    fp32_block = _first_moe_block(MiniMaxM3SparseForCausalLM(fp32_config, backend=_cpu_backend()))
    fp32_block.to(torch.float32)  # fp32 master weights: one storage dtype per unit
    fully_shard_by_dtype(
        fp32_block,
        mesh=None,
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        offload_policy=None,
        fp32_compute_module_names=tuple(MiniMaxM3SparseForCausalLM._keep_in_fp32_modules_strict),
    )
    (policy,) = policies
    assert policy.param_dtype_override_fn(fp32_block.mlp.gate.weight) is torch.float32
    assert policy.param_dtype_override_fn(next(iter(fp32_block.mlp.experts.parameters()))) is None
