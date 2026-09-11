# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Backbone construction, model API, routing precision and distributed ownership contracts."""

import copy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel._transformers.capabilities import _is_deepseek_v4
from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    get_parallelization_strategy,
)
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41AttentionState
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import _is_deepseek_v4_model, apply_ac

# Over the default 5s budget on purpose: this module runs full-model forwards and distributed FSDP checks.
# Shrink the model fixtures and process startup before lowering this further.
pytestmark = pytest.mark.timeout(60)


def _tiny_config() -> DeepseekV41Config:
    return DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=6,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0, 0, 2, 2, 1, 1],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            candidate_source_layer_id=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[],
            engram_num_embeddings=[],
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
            dtype="float32",
        ),
    )


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")


def test_full_tiny_ced_model_trains_after_meta_initialization() -> None:
    torch.manual_seed(8)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    inputs = torch.randint(0, 64, (2, 12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = model(inputs, labels=inputs, output_hidden_states=True)
    assert initial.logits.shape == (2, 12, 64)
    assert len(initial.hidden_states) == 6
    initial_loss = initial.loss.item()
    initial.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    optimizer.zero_grad()
    assert model(inputs, labels=inputs).loss.item() < initial_loss


def test_tied_embeddings_are_rejected() -> None:
    config = _tiny_config()
    config.tie_word_embeddings = True
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        DeepseekV41ForCausalLM(config, backend=_backend())


def _model(dtype: str = "float32") -> DeepseekV41ForCausalLM:
    config = DeepseekV41Config(
        text_config=DeepseekV41TextConfig(
            vocab_size=17,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[],
            dtype=dtype,
        ),
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
    )
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")
    model = DeepseekV41ForCausalLM(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=getattr(torch, dtype))
    return model


def test_default_policy_is_explicit_and_routing_correction_is_fixed() -> None:
    config = _model().config
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
    assert model.config_class is DeepseekV41Config and model.base_model_prefix == "model"
    assert model.backend.attn == "tilelang"
    assert model.backend.linear == "torch" and model.backend.rms_norm == "torch_fp32"
    assert model.backend.dispatcher == "hybridep" and model.backend.experts == "torch_mm"
    assert model.moe_config.gate_bias_update_factor == 0


def test_bf16_backbone_returns_unrounded_fp32_logits():
    torch.manual_seed(31)
    model = _model("bfloat16")
    with torch.no_grad():
        model.lm_head.weight.copy_(
            torch.linspace(-0.04321, 0.05137, model.lm_head.weight.numel()).view_as(model.lm_head.weight)
        )
    result = model(torch.tensor([[3, 5, 7, 9]]), return_hidden_states=True)
    assert result.hidden_states.dtype == torch.bfloat16
    assert result.logits.dtype == torch.float32
    expected = F.linear(result.hidden_states.float(), model.lm_head.weight)
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result.logits != result.logits.bfloat16().float()) > 0
    result.logits.square().mean().backward()
    assert model.lm_head.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_labels_match_independent_shifted_logprob_and_head_gradient():
    torch.manual_seed(17)
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    labels = ids.clone()
    labels[0, 2] = -100
    original = labels.clone()
    result = model(ids, labels=labels, return_hidden_states=True)
    # Compare API variants under the same grad mode and before backward.
    plain = model(ids, return_hidden_states=True)
    assert plain.loss is None
    torch.testing.assert_close(plain.logits, result.logits, rtol=0, atol=0)
    reference = result.logits.detach().clone().requires_grad_()
    targets = labels[:, 1:]
    valid = targets != -100
    probabilities = reference[:, :-1].log_softmax(-1)
    expected = -probabilities.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].mean()
    torch.testing.assert_close(result.loss, expected, rtol=1e-6, atol=1e-7)
    assert result.loss.dtype == torch.float32 and result.loss.ndim == 0
    assert result.hidden_states.shape == (2, 4, model.config.text_config.hidden_size)
    assert "loss" in result and torch.equal(labels, original)
    result.logits.retain_grad()
    result.loss.backward()
    expected.backward()
    torch.testing.assert_close(result.logits.grad, reference.grad, rtol=1e-6, atol=1e-8)
    assert torch.count_nonzero(result.logits.grad[:, -1]) == 0
    assert torch.count_nonzero(result.logits.grad[0, 1]) == 0
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize(
    "metadata",
    [
        {"qkv_format": "thd"},
        {"packed_seq_ids": torch.ones(1, 4, dtype=torch.long)},
        {"seq_lens": torch.tensor([2, 2])},
        {"cu_seqlens": torch.tensor([0, 2, 4])},
        {"cu_seqlens_q": torch.tensor([0, 2, 4])},
    ],
)
def test_labels_reject_packing_before_numerical_forward(metadata):
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(ids, labels=ids, **metadata)


@pytest.mark.parametrize("logits_to_keep", [1, torch.tensor([0, 2])])
def test_labels_require_full_ordered_logits(logits_to_keep):
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(ValueError, match="logits for every input position"):
        model(ids, labels=ids, logits_to_keep=logits_to_keep)


def test_labels_shape_and_inputs_embeds_contract() -> None:
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(ValueError, match="logits for every input position"):
        model(ids, labels=ids[:, :-1])
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids), labels=ids)


def test_captured_streams_and_final_hidden_states_have_distinct_contracts() -> None:
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    final = model(ids, return_hidden_states=True)
    captured = model(ids, output_hidden_states=True)
    assert final.hidden_states.shape == (1, 4, 16)
    assert len(captured.hidden_states) == 1
    assert captured.hidden_states[0].shape == (1, 4, 4, 16)
    torch.testing.assert_close(final.logits, captured.logits, rtol=0, atol=0)


def _build_model(config=None):
    model = DeepseekV41ForCausalLM(config or _tiny_config(), backend=_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def test_default_bf16_router_matches_fp32_reference():
    config = _tiny_config()
    config.text_config.n_routed_experts = 384
    config.text_config.num_experts_per_tok = 6
    model = _build_model(config)
    cast_model_to_dtype(model, torch.bfloat16)
    gate = model.model.layers["2"].ffn.gate
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2048, config.text_config.hidden_size, generator=generator).bfloat16()
    with torch.no_grad():
        gate.e_score_correction_bias.copy_(torch.linspace(-0.04, 0.04, 384))
    scores = F.softplus(F.linear(x.float(), gate.weight.float())).sqrt()
    indices = (scores + gate.e_score_correction_bias).topk(6, dim=-1).indices
    expected = scores.gather(1, indices)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * config.text_config.routed_scaling_factor
    weights, actual, _ = gate(x, torch.ones(2048, dtype=torch.bool), None)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(actual, indices, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    assert model.backend.gate_precision is None


@pytest.mark.parametrize("checkpoint", [False, True])
def test_copied_layer_inputs_preserve_shared_state_and_gradients(checkpoint):
    torch.manual_seed(91)
    reference = _build_model()
    model = copy.deepcopy(reference)
    parameters = dict(model.named_parameters())
    snapshots = []

    def copy_state(module, args, kwargs):
        state = args[2]
        snapshots.append((state, vars(state).copy()))
        return (*args[:2], replace(state)), kwargs

    for layer in model.model.layers.values():
        layer.register_forward_pre_hook(copy_state, with_kwargs=True)
    if checkpoint:
        apply_ac(model)
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    expected = reference(tokens).logits
    actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for state, fields in snapshots:
        assert all(getattr(state, name) is value for name, value in fields.items())
    for name, parameter in reference.named_parameters():
        grad = parameters[name].grad
        if parameter.grad is None:
            assert grad is None
        else:
            torch.testing.assert_close(grad, parameter.grad, atol=0, rtol=0)
    assert model.model.layers["2"].attn.compressor.wkv.weight.grad is not None


def test_block_rejects_rounded_carried_coefficients():
    model = _build_model()
    block = model.model.layers["0"]
    streams = torch.randn(1, 4, model.config.text_config.hc_mult, model.config.text_config.hidden_size)
    with pytest.raises(TypeError, match="FP32 carried coefficients"):
        block(
            streams,
            torch.zeros(1, 4, model.config.text_config.hc_mult, dtype=torch.bfloat16),
            DeepseekV41AttentionState(),
            position_ids=torch.arange(4)[None],
        )


def test_indexers_are_frozen_by_the_model_constructor() -> None:
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    expected = {name for name, _ in model.named_parameters() if ".attn.indexer." in name}
    assert expected and frozen == expected


def test_v41_uses_generic_moe_parallelization() -> None:
    model = DeepseekV41ForCausalLM(_tiny_config(), backend=_backend())
    assert not _is_deepseek_v4_model(model)
    assert type(get_parallelization_strategy(model)) is DefaultParallelizationStrategy
    assert not dsv4_fsdp._is_deepseek_v4_module(model)
    assert not _is_deepseek_v4(model)
    assert _is_deepseek_v4(SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4")))
    for layer in model.model.layers.values():
        assert layer.mlp is layer.ffn
        assert all(".mlp." not in name for name in model.state_dict())


def _fsdp_initialization_worker(rank: int, rendezvous: str, storage_dtype: torch.dtype) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    try:
        config = _tiny_config()
        config.text_config.dtype = storage_dtype
        with torch.device("meta"):
            model = DeepseekV41ForCausalLM(config, backend=_backend())
        model.to_empty(device="cpu")
        expected_dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
        strict_names = model._keep_in_fp32_modules_strict
        for name, parameter in model.named_parameters():
            protected = any(keyword in name for keyword in strict_names)
            assert parameter.dtype == (torch.float32 if protected else storage_dtype), name

        mesh = init_device_mesh("cpu", (2,))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
        )
        for module in model.model.layers.values():
            fully_shard_by_dtype(
                module,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=None,
                fp32_compute_module_names=tuple(strict_names),
                reshard_after_forward=True,
            )
        fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(
            model.lm_head,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        assert isinstance(model, FSDPModule)
        assert all(isinstance(layer, FSDPModule) for layer in model.model.layers.values())
        parameters = dict(model.named_parameters())
        assert all(isinstance(parameter, DTensor) for parameter in parameters.values())

        torch.manual_seed(419 + rank)
        # This covers real CPU FSDP wrapping and initialization, not a CUDA
        # all-gather or forward. The requested compute dtype must not replace
        # either BF16 storage or explicitly requested FP32 master storage.
        model.initialize_weights(torch.device("cpu"), dtype=torch.bfloat16)
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name], name
            assert parameter.dtype == parameter.to_local().dtype == expected_dtypes[name], name
            assert torch.isfinite(parameter.to_local()).all(), name
            if name.endswith((".attn_hc.fn", ".ffn_hc.fn")):
                local = parameter.to_local()
                assert torch.any(local != local.bfloat16().float()), name

        name = "model.layers.0.attn_hc.fn"
        # A checkpoint value deliberately between BF16 representable numbers
        # must reach the actual FSDP local shard without rounding or replacement.
        source = torch.full(parameters[name].shape, 1.00123, dtype=torch.float32)
        state = model.state_dict()
        state[name] = distribute_tensor(source, mesh, [Shard(0)])
        model.load_state_dict(state, strict=True)
        for key, parameter in model.named_parameters():
            assert parameter is parameters[key], key
            assert parameter.dtype == parameter.to_local().dtype == expected_dtypes[key], key
        torch.testing.assert_close(parameters[name].to_local(), source.chunk(2, dim=0)[rank], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32_master"])
def test_fsdp_initialization_preserves_local_storage_dtype_and_checkpoint_values(
    tmp_path: Path, storage_dtype: torch.dtype
) -> None:
    mp.spawn(_fsdp_initialization_worker, args=(str(tmp_path / "rendezvous"), storage_dtype), nprocs=2, join=True)
