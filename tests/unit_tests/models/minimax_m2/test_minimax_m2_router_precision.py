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

"""MiniMax-M2 router precision contract.

Released MiniMax-M2 checkpoints store the router gate weight in fp32 and the
HF reference projects with ``hidden_states.to(weight.dtype)``, so the
checkpoint-faithful router is fp32 end to end: fp32 parameter, fp32
projection, fp32 scoring, fp32 selected weights (AMINT-286).
"""

import torch
from transformers import AutoConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM

TINY = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    rotary_dim=16,
    num_local_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=128,
)


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
    config = AutoConfig.for_model("minimax_m2", torch_dtype="bfloat16", **TINY)
    model = MiniMaxM2ForCausalLM(config, backend=_cpu_backend()).eval()

    assert model.model.backend.gate_precision == torch.float32
    assert model.model.moe_config.router_weights_fp32 is True
    assert "mlp.gate.weight" in MiniMaxM2ForCausalLM._keep_in_fp32_modules_strict
    assert "mlp.gate.e_score_correction_bias" in MiniMaxM2ForCausalLM._keep_in_fp32_modules_strict

    # After the model-wide bf16 cast, the fp32 contract keeps the router gate
    # parameter and bias in fp32 while the rest of the model is bf16.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    gate = model.model.layers["0"].mlp.gate
    assert gate.weight.dtype == torch.float32
    assert gate.e_score_correction_bias.dtype == torch.float32
    assert model.model.layers["0"].self_attn.q_proj.weight.dtype == torch.bfloat16

    # Selected routing weights stay fp32 through the gate output, matching the
    # HF reference's top_k_weights.to(router_logits.dtype) with an fp32 weight.
    # x: Tensor of shape [tokens, hidden] in bf16, like real routed inputs.
    x = torch.randn(8, TINY["hidden_size"], dtype=torch.bfloat16)
    weights, indices, _aux = gate(x, torch.ones(8, dtype=torch.bool), None)
    assert weights.dtype == torch.float32
    assert indices.shape == (8, TINY["num_experts_per_tok"])


def test_gate_is_fp32_at_construction_for_fsdp_dtype_grouping():
    """The gate must be fp32 from allocation, before any init or checkpoint cast.

    FSDP shards the freshly constructed (meta/from_pretrained) module: a
    bf16-allocated gate weight with an fp32-pinned compute dtype shares its
    module with the fp32 correction-bias buffer, which FSDP cannot isolate
    (pipeline 64344786: "FSDP could not isolate parameters with a distinct
    dtype from siblings in the same module: mlp.gate.weight").
    """
    import torch.distributed.fsdp as fsdp

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

    config = AutoConfig.for_model("minimax_m2", torch_dtype="bfloat16", **TINY)
    model = MiniMaxM2ForCausalLM(config, backend=_cpu_backend())
    block = model.model.layers["0"]

    # No initialize_weights on purpose: this is the state FSDP shards.
    assert block.mlp.gate.weight.dtype == torch.float32
    assert block.mlp.gate.e_score_correction_bias.dtype == torch.float32

    fully_shard_by_dtype(
        block,
        mesh=None,
        mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        offload_policy=None,
        fp32_compute_module_names=tuple(MiniMaxM2ForCausalLM._keep_in_fp32_modules_strict),
        fully_shard_fn=lambda *args, **kwargs: None,
    )


def test_explicit_gate_precision_override_is_preserved():
    config = AutoConfig.for_model("minimax_m2", torch_dtype="bfloat16", **TINY)
    backend = _cpu_backend()
    backend.gate_precision = torch.bfloat16
    model = MiniMaxM2ForCausalLM(config, backend=backend).eval()
    assert model.model.backend.gate_precision == torch.bfloat16
