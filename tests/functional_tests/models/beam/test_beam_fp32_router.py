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

"""Beam's FP32 router and rotary inputs through the production DP/EP sharding paths."""

import copy
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.beam.layers import BeamAttention, BeamGate
from nemo_automodel.components.models.beam.model import BeamForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.parallelizer import parallelize_model


def _worker(rank: int, rendezvous: str, ep_size: int, activation_checkpointing: bool) -> None:
    from transformers.models.beam import BeamConfig

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("highest")
        config = BeamConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            mlp_layer_types=["dense", "sparse", "sparse"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": None,
            },
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=2.5,
            norm_topk_prob=True,
            final_logit_softcapping=30.0,
            tie_word_embeddings=False,
            use_cache=False,
            dtype="float32",
        )
        reference = BeamForCausalLM(
            config,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                experts="torch",
                dispatcher="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
            ),
        ).to(device)
        reference.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
        reference_gates = {name: module for name, module in reference.named_modules() if isinstance(module, BeamGate)}
        with torch.no_grad():
            for gate in reference_gates.values():
                gate.e_score_correction_bias.normal_(std=0.1)
                for tensor in (gate.weight, gate.e_score_correction_bias):
                    assert not torch.equal(tensor, tensor.bfloat16().float())
        model = copy.deepcopy(reference)

        # Each rank gets different sequences; the unsharded reference sees their complete global batch.
        input_ids = torch.randint(0, config.vocab_size, (4, 16), generator=torch.Generator().manual_seed(7)).to(device)
        reference_tables = []

        def capture_reference_rope(_module, _args, kwargs) -> None:
            """Capture the unsharded rotary inputs.

            Args:
                _module: Beam attention module.
                _args: First tensor is hidden states [batch, sequence, hidden].
                kwargs: rope_tables contains FP32 (cos, sin), each [batch, sequence, 1, head_dim].
            """
            reference_tables.append(tuple(table.detach().clone() for table in kwargs["rope_tables"]))

        reference_hook = reference.model.layers["0"].self_attn.register_forward_pre_hook(
            capture_reference_rope, with_kwargs=True
        )
        try:
            reference_logits = reference(input_ids).logits.float()
            F.cross_entropy(reference_logits[:, :-1].flatten(0, 1), input_ids[:, 1:].flatten()).backward()
        finally:
            reference_hook.remove()
        assert len(reference_tables) == 1

        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32
        )
        strategy = FSDP2Config(mp_policy=policy, activation_checkpointing=activation_checkpointing)
        mesh = MeshContext.build(strategy, ParallelismSizes(dp_size=2, ep_size=ep_size), world_size=2)
        if ep_size == 2:
            parallelize_model(
                model,
                mesh.device_mesh,
                mesh.moe_mesh,
                activation_checkpointing=activation_checkpointing,
                mp_policy=policy,
                **mesh.parallelize_axis_kwargs(),
            )
        else:
            FSDP2Manager(strategy, device_mesh=mesh.device_mesh).parallelize(model)

        gates = {
            name.replace("_checkpoint_wrapped_module.", ""): module
            for name, module in model.named_modules()
            if isinstance(module, BeamGate)
        }
        attentions = [module for module in model.modules() if isinstance(module, BeamAttention) and module.use_rope]
        assert set(gates) == set(reference_gates) and len(gates) == 2
        assert len(attentions) == 2
        seen_gates, seen_attention = set(), set()

        def check_gate(name, module, _args) -> None:
            """Check the original FP32 values after FSDP unshards the router.

            Args:
                name: Unwrapped router module name.
                module: Beam router with weight [experts, hidden] and bias [experts].
                _args: Hidden states [tokens, hidden] and an optional token mask [tokens].
            """
            for suffix in ("weight", "e_score_correction_bias"):
                actual = getattr(module, suffix)
                actual = actual.to_local() if isinstance(actual, DTensor) else actual
                expected = getattr(reference_gates[name], suffix)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.dtype == torch.float32
            seen_gates.add(name)

        def check_rope(module, args, kwargs) -> None:
            """Check hidden states [batch, sequence, hidden] and FP32 rotary tables.

            Args:
                module: Beam attention module, after block-level FSDP input handling.
                args: First tensor is BF16 hidden states [batch, sequence, hidden].
                kwargs: rope_tables contains (cos, sin), each [batch, sequence, 1, head_dim].
            """
            assert args[0].dtype == torch.bfloat16
            for actual, expected in zip(kwargs["rope_tables"], reference_tables[0]):
                assert actual.dtype == torch.float32
                torch.testing.assert_close(actual, expected.chunk(2)[rank], rtol=0, atol=0)
            seen_attention.add(module)

        hooks = [gate.register_forward_pre_hook(partial(check_gate, name)) for name, gate in gates.items()]
        hooks.extend(attention.register_forward_pre_hook(check_rope, with_kwargs=True) for attention in attentions)
        try:
            local_ids = input_ids.chunk(2)[rank]
            logits = model(local_ids).logits.float()
            torch.testing.assert_close(logits, reference_logits.chunk(2)[rank], rtol=1e-2, atol=1e-2)
            F.cross_entropy(logits[:, :-1].flatten(0, 1), local_ids[:, 1:].flatten()).backward()
        finally:
            for hook in hooks:
                hook.remove()
        assert seen_gates == set(gates)
        assert seen_attention == set(attentions)
        for name, gate in gates.items():
            assert gate.weight.grad is not None
            actual = gate.weight.grad.full_tensor()
            expected = reference_gates[name].weight.grad
            assert actual.dtype == torch.float32
            assert torch.isfinite(actual).all()
            assert expected.norm() > 0
            # BF16 dispatch can change rounding; a relative norm also detects incorrect DP/EP gradient scaling.
            assert (actual - expected).norm() <= 0.05 * expected.norm(), name
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize(("ep_size", "activation_checkpointing"), [(1, False), (2, True)], ids=["dp2", "ep2_ac"])
def test_beam_fp32_router(tmp_path: Path, ep_size: int, activation_checkpointing: bool) -> None:
    """Match unsharded Beam while preserving router values and RoPE through distributed execution."""
    mp.spawn(
        _worker,
        args=((tmp_path / "rendezvous").as_uri(), ep_size, activation_checkpointing),
        nprocs=2,
        join=True,
    )
