# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.fp8_qdq import FP8QDQConfig
from nemo_automodel.components.moe.layers import MLP, MoE


def _reference(x: torch.Tensor, *, weight: bool) -> torch.Tensor:
    """Scalar-block reference, independent of production reshape logic.

    Args:
        x: Tensor [out, in] for weights or [tokens, hidden] for activations.
        weight: Whether blocks cover 128 rows instead of one token.
    Returns:
        Dequantized tensor with x's shape and dtype, with no gradient.
    """
    result = torch.empty_like(x)
    with torch.no_grad():
        for row in range(0, x.shape[0], 128 if weight else 1):
            for col in range(0, x.shape[1], 128):
                sl = (slice(row, row + (128 if weight else 1)), slice(col, col + 128))
                block = x[sl].float()
                maximum = block.abs().max()
                if weight:
                    multiplier = 448.0 / maximum if maximum != 0 else torch.tensor(1.0)
                    quantized = (block * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                    result[sl] = (quantized * multiplier.reciprocal()).to(x.dtype)
                else:
                    scale = maximum.clamp_min(1e-10) / 448.0
                    result[sl] = ((block / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight", [False, True])
def test_block_scales_layout_and_ste(dtype, weight):
    torch.manual_seed(53)
    shape = (2, 256, 256) if weight else (2, 3, 256)
    x = torch.randn(shape).to(dtype)
    x[..., :128] *= 0.01
    x[0].zero_()
    x.requires_grad_()
    before = x.detach().clone()
    config = FP8QDQConfig(weights=True, activations=True)
    actual = config.weight(x) if weight else config.activation(x)
    expected = (
        torch.stack([_reference(t, weight=True) for t in x])
        if weight
        else _reference(x.flatten(0, 1), weight=False).reshape(shape)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


def test_empty_activations_and_disabled_identity():
    x = torch.empty(0, 256, requires_grad=True)
    FP8QDQConfig(activations=True).activation(x).sum().backward()
    assert x.grad.shape == x.shape
    disabled = FP8QDQConfig()
    odd_shape = torch.randn(3, 17)
    assert disabled.activation(odd_shape) is odd_shape
    assert disabled.weight(odd_shape) is odd_shape
    with pytest.raises(ValueError, match="divisible by 128"):
        FP8QDQConfig(weights=True).weight(odd_shape)
    with pytest.raises(ValueError, match="divisible by 128"):
        FP8QDQConfig(activations=True).activation(odd_shape)


@pytest.mark.parametrize("weights,activations", list(product([False, True], repeat=2)))
def test_mlp_forward_backward_reload_and_weight_update(weights, activations):
    torch.manual_seed(5)
    config = FP8QDQConfig(weights=weights, activations=activations)
    baseline = MLP(128, 256, "torch", dtype=torch.bfloat16, swiglu_limit=10)
    model = MLP(128, 256, "torch", dtype=torch.bfloat16, swiglu_limit=10, fp8_qdq=config)
    model.load_state_dict(baseline.state_dict(), strict=True)
    assert model.state_dict().keys() == baseline.state_dict().keys()
    x = torch.randn(4, 128).bfloat16().requires_grad_()
    actual = model(x)
    with torch.no_grad():
        qx = _reference(x, weight=False) if activations else x
        gate_w = _reference(model.gate_proj.weight, weight=True) if weights else model.gate_proj.weight
        up_w = _reference(model.up_proj.weight, weight=True) if weights else model.up_proj.weight
        down_w = _reference(model.down_proj.weight, weight=True) if weights else model.down_proj.weight
        gate = F.linear(qx, gate_w).float().clamp(max=10)
        up = F.linear(qx, up_w).float().clamp(-10, 10)
        intermediate = (F.silu(gate) * up).bfloat16()
        expected = F.linear(_reference(intermediate, weight=False) if activations else intermediate, down_w)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0
    saved = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.down_proj.weight.zero_()
    assert model(x).count_nonzero() == 0  # No stale quantized weight cache.
    model.load_state_dict(saved, strict=True)
    torch.testing.assert_close(model(x), actual, rtol=0, atol=0)
    if not config.enabled:
        torch.testing.assert_close(actual, baseline(x), rtol=0, atol=0)


def _text_config(**overrides):
    return Glm5NextTextConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        n_routed_experts=3,
        num_experts_per_tok=2,
        **overrides,
    )


@pytest.mark.parametrize("weights,activations", list(product([False, True], repeat=2)))
def test_routed_loop_matches_independent_expert_reference(weights, activations):
    torch.manual_seed(9)
    text = _text_config(routed_fp8_weight_qdq=weights, routed_fp8_activation_qdq=activations)
    config = build_glm5_next_moe_config(text, torch.bfloat16)
    model = GroupedExperts(config, BackendConfig(experts="torch", dispatcher="torch"))
    with torch.no_grad():
        model.init_weights(torch.device("cpu"), 0.02)
    x = torch.randn(3, 128).bfloat16().requires_grad_()
    probs = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]], requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]])  # Expert 2 receives no tokens.
    mask = torch.tensor([True, True, False])
    actual = model(x, mask, probs, indices)
    with torch.no_grad():
        outputs = torch.zeros_like(x, dtype=torch.float32)
        for token in range(2):
            for slot in range(2):
                expert = indices[token, slot]
                first = model.gate_and_up_projs[expert].T
                second = model.down_projs[expert].T
                qx = _reference(x[token : token + 1], weight=False) if activations else x[token : token + 1]
                if weights:
                    first, second = _reference(first, weight=True), _reference(second, weight=True)
                gate, up = F.linear(qx, first).float().chunk(2, -1)
                intermediate = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
                if activations:
                    intermediate = _reference(intermediate, weight=False)
                outputs[token] += F.linear(intermediate, second)[0].float() * probs[token, slot]
    torch.testing.assert_close(actual, outputs.bfloat16(), rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[2].count_nonzero() == 0
    assert probs.grad is not None and torch.isfinite(probs.grad).all()
    assert model.gate_and_up_projs.grad[2].count_nonzero() == 0
    assert model.down_projs.grad[2].count_nonzero() == 0


@pytest.mark.parametrize("group", ["dense", "routed", "shared"])
def test_config_serialization_and_independent_switches(group):
    options = {f"{group}_fp8_weight_qdq": True, f"{group}_fp8_activation_qdq": True}
    text = Glm5NextTextConfig.from_dict(_text_config(**options).to_dict())
    for name in ("dense", "routed", "shared"):
        assert getattr(text, f"{name}_fp8_weight_qdq") is (name == group)
        assert getattr(text, f"{name}_fp8_activation_qdq") is (name == group)
    config = build_glm5_next_moe_config(text, torch.bfloat16)
    model = MoE(config, BackendConfig(linear="torch", experts="torch", dispatcher="torch"))
    assert model.experts.config.routed_fp8_qdq.enabled is (group == "routed")
    # Changing shared/dense switches must never quantize the router weight.
    assert set(model.gate.state_dict()) == set(
        MoE(
            build_glm5_next_moe_config(_text_config(), torch.bfloat16),
            BackendConfig(linear="torch", experts="torch", dispatcher="torch"),
        ).gate.state_dict()
    )


def test_unsupported_routed_backends_and_layout_fail_early(monkeypatch):
    config = build_glm5_next_moe_config(_text_config(routed_fp8_weight_qdq=True), torch.bfloat16)
    backend = BackendConfig(experts="torch", dispatcher="torch")
    monkeypatch.setattr(backend, "experts", "te")  # Avoid optional-TE CPU fallback during construction.
    with pytest.raises(ValueError, match="supports torch"):
        MoE(config, backend)
    with pytest.raises(ValueError, match="divisible by 128"):
        MLP(127, 256, "torch", dtype=torch.bfloat16, fp8_qdq=FP8QDQConfig(weights=True))


@pytest.mark.parametrize("weights,activations", list(product([False, True], repeat=2)))
@pytest.mark.parametrize("empty", [False, True])
def test_grouped_mm_matches_loop_forward_and_backward(weights, activations, empty):
    torch.manual_seed(14)
    config = build_glm5_next_moe_config(
        _text_config(routed_fp8_weight_qdq=weights, routed_fp8_activation_qdq=activations), torch.bfloat16
    )
    loop = GroupedExperts(config, BackendConfig(experts="torch", dispatcher="torch"))
    grouped = GroupedExperts(config, BackendConfig(experts="torch_mm", dispatcher="torch"))
    with torch.no_grad():
        loop.init_weights(torch.device("cpu"), 0.02)
    grouped.load_state_dict(loop.state_dict())
    inputs = [torch.randn(4, 128).bfloat16().requires_grad_()]
    inputs.append(inputs[0].detach().clone().requires_grad_())
    probabilities = [torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2]], requires_grad=True)]
    probabilities.append(probabilities[0].detach().clone().requires_grad_())
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    mask = torch.zeros(4, dtype=torch.bool) if empty else torch.tensor([True, True, True, False])
    outputs = [module(x, mask, p, indices) for module, x, p in zip((loop, grouped), inputs, probabilities)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for output in outputs:
        output.float().square().sum().backward()
    if empty:
        # The pre-existing loop leaves the unused input gradient absent,
        # while grouped_mm explicitly anchors a zero gradient.
        assert inputs[0].grad is None
        assert inputs[1].grad.count_nonzero() == 0
    else:
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=0.02, atol=2e-5)
    for (_, left), (_, right) in zip(loop.named_parameters(), grouped.named_parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=0.02, atol=2e-5)


def test_shared_expert_and_dense_construction_and_autocast():
    from nemo_automodel.components.models.glm5_next.layers import Glm5NextDecoderLayer

    torch.manual_seed(21)
    text = _text_config(
        dense_fp8_weight_qdq=True,
        dense_fp8_activation_qdq=True,
        shared_fp8_weight_qdq=True,
        shared_fp8_activation_qdq=True,
    )
    config = build_glm5_next_moe_config(text, torch.bfloat16)
    backend = BackendConfig(linear="torch", experts="torch", dispatcher="torch")
    dense = Glm5NextDecoderLayer(text, 0, config, backend).mlp
    shared = MoE(config, backend).shared_experts
    for module in (dense, shared):
        assert module.gate_proj.qdq.enabled and module.down_proj.qdq.enabled
        x = torch.randn(3, 128).bfloat16().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module(x)
        torch.testing.assert_close(actual, module(x), rtol=0, atol=0)
        actual.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_tiny_hybrid_model_with_all_qdq_and_checkpoint_roundtrip():
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config

    config = tiny_glm5_next_config()
    text = config.text_config.to_dict()
    text.update(
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        index_head_dim=128,
        indexer_fp8_fake_quant=True,
    )
    for group in ("dense", "routed", "shared"):
        for operand in ("weight", "activation"):
            text[f"{group}_fp8_{operand}_qdq"] = True
    config.text_config = Glm5NextTextConfig.from_dict(text)
    model = Glm5NextForConditionalGeneration(config, backend=tiny_backend(adapter=True))
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3]])
    logits = model(input_ids=ids).logits
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()
    assert model.model.language_model.layers["0"].mlp.gate_proj.weight.grad.count_nonzero() > 0
    assert model.model.language_model.layers["3"].mlp.experts.gate_and_up_projs.grad.count_nonzero() > 0
    restored = Glm5NextForConditionalGeneration(config, backend=tiny_backend(adapter=True))
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(input_ids=ids).logits, logits, rtol=0, atol=0)


def _ep_qdq_worker(rank: int, init_file: str):
    """Check real two-rank expert sharding, token exchange and STE gradients."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{init_file}")
    try:
        torch.manual_seed(29)
        text = Glm5NextTextConfig(
            hidden_size=128,
            moe_intermediate_size=128,
            n_routed_experts=4,
            num_experts_per_tok=2,
            routed_fp8_weight_qdq=True,
            routed_fp8_activation_qdq=True,
        )
        config = build_glm5_next_moe_config(text, torch.bfloat16)
        backend = BackendConfig(experts="torch_mm", dispatcher="torch")
        reference = GroupedExperts(config, backend)
        with torch.no_grad():
            reference.init_weights(torch.device("cpu"), 0.02)
        distributed = GroupedExperts(config, backend)
        mesh = init_device_mesh("cpu", (2,))
        for name, parameter in reference.named_parameters():
            setattr(
                distributed, name, torch.nn.Parameter(distribute_tensor(parameter.detach().clone(), mesh, [Shard(0)]))
            )
        all_x = torch.randn(5, 128).bfloat16().requires_grad_()
        all_probs = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]], requires_grad=True)
        indices = torch.tensor([[0, 2], [1, 3], [0, 3], [1, 2], [2, 3]])
        mask = torch.tensor([True, True, True, True, False])
        sl = slice(0, 2) if rank == 0 else slice(2, 5)
        x = all_x[sl].detach().clone().requires_grad_()
        probs = all_probs[sl].detach().clone().requires_grad_()
        expected = reference(all_x, mask, all_probs, indices)
        actual = distributed(x, mask[sl], probs, indices[sl])
        torch.testing.assert_close(actual, expected[sl], rtol=0, atol=0)
        actual.float().square().sum().backward()
        expected.float().square().sum().backward()
        torch.testing.assert_close(x.grad, all_x.grad[sl], rtol=0.025, atol=2e-5)
        torch.testing.assert_close(probs.grad, all_probs.grad[sl], rtol=0.025, atol=2e-5)
        for (name, param), (_, ref) in zip(distributed.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(
                param.grad.to_local(),
                ref.grad[rank * 2 : (rank + 1) * 2],
                rtol=0.025,
                atol=2e-5,
                msg=lambda message: f"rank={rank} {name}: {message}",
            )
    finally:
        dist.destroy_process_group()


def test_two_rank_expert_qdq_forward_backward(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_ep_qdq_worker, args=(str(tmp_path / "gloo-init"),), nprocs=2, join=True)
