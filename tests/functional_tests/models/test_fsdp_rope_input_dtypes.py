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

"""Default FSDP input policy with native dense models and original TE RoPE.

The production BackendConfig guard still disables fused RoPE. This diagnostic
overrides only that test instance after construction, enabling both RoPE paths.
The reference uses identical weights/inputs without FSDP and retains original
FP32 buffers during BF16 parameter conversion. No precision adapter or replacement
kernel is used. Explicit cast_forward_inputs=True reproduces the old policy.
"""

import copy
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.config import FSDP2Config


def _make_model(fusion: bool, device: torch.device):
    from transformers import Qwen3MoeConfig

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeModel

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=65536,
        torch_dtype=torch.float32,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")
    backend.rope_fusion = fusion  # Diagnostic only: leave the production guard unchanged.
    model = Qwen3MoeModel(config, backend).to(device)
    model.init_weights(device)
    assert model.layers["0"].self_attn.backend.rope_fusion is fusion
    for name, parameter in model.layers["0"].self_attn.named_parameters():
        if "_proj.weight" in name:
            torch.nn.init.normal_(parameter, std=0.12)
    return model


def _record_embedding(_module, _args, output: torch.Tensor, observations: list[torch.Tensor]) -> None:
    """Keep the embedding activation for gradient comparison.

    Args:
        _module: Embedding module.
        _args: Its positional token-ID input [batch, sequence].
        output: Activation [batch, sequence, hidden].
        observations: Recorded activations with the same layout; retain aliases.
    """
    output.retain_grad()
    observations.append(output)


def _check_dense_models(mesh: DeviceMesh, device: torch.device) -> None:
    """Preserve dense-model eager outputs and gradients after removing input casts."""
    from transformers import LlamaConfig, Qwen2Config, Qwen3Config

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.llama.model import LlamaForCausalLM
    from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM
    from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

    policy = FSDP2Config().mp_policy
    for config_cls, model_cls in (
        (LlamaConfig, LlamaForCausalLM),
        (Qwen2Config, Qwen2ForCausalLM),
        (Qwen3Config, Qwen3ForCausalLM),
    ):
        torch.manual_seed(218)
        config = config_cls(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=2048,
            torch_dtype=torch.float32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "sdpa"
        model = model_cls(
            config, backend=BackendConfig(linear="torch", attn="sdpa", rms_norm="torch_fp32", rope_fusion=False)
        ).to(device)
        legacy = copy.deepcopy(model)
        for candidate, casting in ((model, False), (legacy, True)):
            active_policy = replace(policy, cast_forward_inputs=casting)
            for block in candidate.model.layers:
                fully_shard(block, mesh=mesh, mp_policy=active_policy)
            fully_shard(candidate, mesh=mesh, mp_policy=active_policy)
        ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        positions = torch.tensor([[0, 1, 255, 511, 1028]], device=device)
        actual = model(ids, position_ids=positions, use_cache=False).logits
        expected = legacy(ids, position_ids=positions, use_cache=False).logits
        assert actual.dtype == torch.bfloat16
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        for (name, parameter), (reference_name, reference_parameter) in zip(
            model.named_parameters(), legacy.named_parameters()
        ):
            assert name == reference_name
            assert parameter.grad is not None, name
            gradient = parameter.grad.full_tensor()
            assert torch.isfinite(gradient).all(), name
            torch.testing.assert_close(gradient, reference_parameter.grad.full_tensor(), rtol=0, atol=0, msg=name)
        if dist.get_rank() == 0:
            print(f"{model_cls.__name__}: eager outputs and gradients match legacy cast=True", flush=True)
        dist.barrier()


def _worker(rank: int, rendezvous: str) -> None:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        mesh = init_device_mesh("cuda", (2,))
        policy = FSDP2Config().mp_policy
        assert policy.cast_forward_inputs is False
        assert policy.param_dtype == torch.bfloat16
        for fusion in (False, True):
            for checkpoint in (False, True):
                torch.manual_seed(204)
                model = _make_model(fusion, device)
                reference, old_cast = copy.deepcopy(model), copy.deepcopy(model)
                buffers = dict(reference.named_buffers(remove_duplicate=False))
                reference.to(dtype=torch.bfloat16)
                for name, buffer in buffers.items():
                    owner, _, local_name = name.rpartition(".")
                    setattr(reference.get_submodule(owner), local_name, buffer)
                angles, embeddings = [[], [], []], [[], [], []]
                for index, candidate in enumerate((model, reference, old_cast)):
                    candidate.embed_tokens.register_forward_hook(
                        lambda module, args, output, index=index: _record_embedding(
                            module, args, output, embeddings[index]
                        )
                    )
                    candidate.layers["0"].self_attn.register_forward_pre_hook(
                        lambda _module, _args, kwargs, index=index: angles[index].append(
                            kwargs["freqs_cis"].detach().clone()
                        ),
                        with_kwargs=True,
                    )
                    if index != 1:
                        active_policy = policy if index == 0 else replace(policy, cast_forward_inputs=True)
                        if checkpoint:
                            candidate.layers["0"] = checkpoint_wrapper(candidate.layers["0"])
                        fully_shard(candidate.layers["0"], mesh=mesh, mp_policy=active_policy)
                        fully_shard(candidate, mesh=mesh, mp_policy=active_policy)
                ids = torch.tensor([[1, 7, 3, 11]], device=device)
                positions = torch.tensor([[0, 1028, 4099, 32769]], device=device)
                with patch(
                    "transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb", wraps=apply_rotary_pos_emb
                ) as te_rope:
                    actual, expected, old_output = (
                        candidate(ids, position_ids=positions) for candidate in (model, reference, old_cast)
                    )
                    assert te_rope.call_count == (6 if fusion else 0)
                assert all(states[0].dtype == torch.bfloat16 for states in embeddings)
                assert angles[0][0].dtype == angles[1][0].dtype == torch.float32
                assert angles[2][0].dtype == torch.bfloat16
                torch.testing.assert_close(angles[0][0], angles[1][0], rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                upstream = torch.randn_like(actual)
                for output in (actual, expected, old_output):
                    output.backward(upstream)
                torch.testing.assert_close(embeddings[0][0].grad, embeddings[1][0].grad, rtol=0.02, atol=0.02)
                max_parameter_error = 0.0
                for (name, parameter), (reference_name, reference_parameter) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    assert name.replace("_checkpoint_wrapped_module.", "") == reference_name
                    if reference_parameter.grad is None:
                        assert parameter.grad is None
                        continue
                    gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
                    error = (gradient.float() - reference_parameter.grad.float()).abs().max().item()
                    max_parameter_error = max(max_parameter_error, error)
                    torch.testing.assert_close(
                        gradient.float(), reference_parameter.grad.float(), rtol=0.02, atol=0.02, msg=name
                    )
                old_error = (old_output.float() - expected.float()).abs().max().item()
                if fusion:
                    assert old_error > 0.05
                    assert not torch.equal(angles[0][0], angles[2][0].float())
                else:
                    torch.testing.assert_close(actual, old_output, rtol=0, atol=0)
                if rank == 0:
                    print(
                        dict(
                            fusion=fusion,
                            checkpoint=checkpoint,
                            output_error=0.0,
                            old_cast_error=old_error,
                            embedding_gradient_error=(embeddings[0][0].grad - embeddings[1][0].grad).abs().max().item(),
                            parameter_gradient_error=max_parameter_error,
                        ),
                        flush=True,
                    )
                dist.barrier()
        _check_dense_models(mesh, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices and original TE RoPE")
def test_default_fsdp_rope_input_dtypes(tmp_path):
    pytest.importorskip("transformer_engine")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "qwen3_moe"),), nprocs=2, join=True)
