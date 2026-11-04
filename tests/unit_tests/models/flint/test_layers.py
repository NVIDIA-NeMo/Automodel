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

"""CPU component parity for Flint attention, decoder layers and the shared SDPA sink backend."""

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.layers import FlintAttention, FlintDecoderLayer
from nemo_automodel.components.moe.layers import MLP, MoE
from tests.functional_tests.models.flint.reference import rms, rope
from tests.unit_tests.models.flint.conftest import tiny_backend, tiny_config

_CASES = {"dense_full": 0, "moe_sliding": 2, "moe_full": 5}
_CASE_IDS = tuple(_CASES)
_SEQUENCE_LENGTHS = (5, 3)


def _backend(**overrides: Any) -> BackendConfig:
    """Return explicit CPU backend settings, optionally changing one failure case."""
    settings = dict(
        attn="eager",
        linear="torch",
        rms_norm="torch",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
        fake_balanced_gate=False,
    )
    settings.update(overrides)
    return BackendConfig(**settings)


def _case(case_id: str) -> tuple[FlintConfig, int, dict[str, torch.Tensor]]:
    """Build a tiny six-layer schedule and seeded inputs for one dense/full, MoE/sliding or MoE/full block.

    Args:
        case_id: Identifier of the dense/full, MoE/sliding, or MoE/full case.

    Returns:
        Config, layer index, and inputs. ``input`` and ``upstream`` have shape
        [batch, sequence, hidden]; ``positions`` and ``valid_tokens`` are
        [batch, sequence]. The upstream adjoint is zero on padded positions so
        that padded rows, whose outputs are unspecified, carry no gradient.
    """
    config = FlintConfig(
        hidden_size=12,
        intermediate_size=20,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=31,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=7,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0, 1],
        layer_types=[
            "full_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        no_rope_layer_indices=[0, 1, 5],
        sliding_window=3,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.0,
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(101 * (list(_CASES).index(case_id) + 1))
    batch, sequence = len(_SEQUENCE_LENGTHS), max(_SEQUENCE_LENGTHS)
    valid = torch.arange(sequence)[None, :] < torch.tensor(_SEQUENCE_LENGTHS)[:, None]
    upstream = torch.randn(batch, sequence, config.hidden_size, generator=generator) * valid[..., None]
    inputs = {
        "input": torch.randn(batch, sequence, config.hidden_size, generator=generator),
        "positions": torch.arange(sequence).expand(batch, sequence).clone(),
        "valid_tokens": valid,
        "upstream": upstream,
    }
    return config, _CASES[case_id], inputs


def _seed_parameters(layer: FlintDecoderLayer, seed: int) -> None:
    """Fill every block parameter with seeded random values, including norms and sinks."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            values = torch.randn(parameter.shape, generator=generator) * 0.3
            if name.endswith("norm.weight"):
                values = values + 1
            parameter.copy_(values)


def _source_parameters(layer: FlintDecoderLayer) -> dict[str, torch.Tensor]:
    """Return independent FP32 leaves in the source [out,in] / [experts,out,in] layout.

    Grouped gate/up [experts,hidden,2*intermediate] splits into gate/up
    [experts,intermediate,hidden]; grouped down [experts,intermediate,hidden]
    becomes [experts,hidden,intermediate]. Other tensors keep their layout.
    """
    source: dict[str, torch.Tensor] = {}
    for name, parameter in layer.named_parameters():
        tensor = parameter.detach().clone()
        if name == "mlp.experts.gate_and_up_projs":
            gate, up = tensor.chunk(2, dim=-1)
            source["mlp.experts.gate_proj.weight"] = gate.transpose(-1, -2).contiguous()
            source["mlp.experts.up_proj.weight"] = up.transpose(-1, -2).contiguous()
        elif name == "mlp.experts.down_projs":
            source["mlp.experts.down_proj.weight"] = tensor.transpose(-1, -2).contiguous()
        else:
            source[
                name.replace("self_attn._fp32_params.", "self_attn.").replace(
                    "mlp.shared_experts.", "mlp.shared_expert."
                )
            ] = tensor
    return {name: tensor.requires_grad_() for name, tensor in source.items()}


def _mapped_gradient(name: str, source: dict[str, torch.Tensor]) -> torch.Tensor:
    """Map a native parameter name to the reference gradient in the native layout."""
    if name == "mlp.experts.gate_and_up_projs":
        gate = source["mlp.experts.gate_proj.weight"].grad.transpose(-1, -2)
        up = source["mlp.experts.up_proj.weight"].grad.transpose(-1, -2)
        return torch.cat((gate, up), dim=-1)
    if name == "mlp.experts.down_projs":
        return source["mlp.experts.down_proj.weight"].grad.transpose(-1, -2)
    return source[
        name.replace("self_attn._fp32_params.", "self_attn.").replace("mlp.shared_experts.", "mlp.shared_expert.")
    ].grad


def _reference_block(
    config: FlintConfig, layer_idx: int, weights: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Evaluate one decoder block with test-owned math; no production attention, routing or expert helpers.

    Returns the block output plus the normalized attention input, the attention
    output after ``o_proj``, the normalized MLP input and the routed expert
    selection sorted by expert ID. Padded queries attend only to the sink.
    """
    hidden, valid = inputs["input"], inputs["valid_tokens"]
    b, s, h = hidden.shape
    qh, kh, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    eps = config.rms_norm_eps
    normalized = rms(hidden, weights["input_layernorm.weight"], eps)
    q = F.linear(normalized, weights["self_attn.q_proj.weight"]).reshape(b, s, qh, d)
    k = F.linear(normalized, weights["self_attn.k_proj.weight"]).reshape(b, s, kh, d)
    v = F.linear(normalized, weights["self_attn.v_proj.weight"]).reshape(b, s, kh, d)
    q, k = rms(q, weights["self_attn.q_norm.weight"], eps), rms(k, weights["self_attn.k_norm.weight"], eps)
    if layer_idx not in config.no_rope_layer_indices:
        theta = float(config.rope_parameters["rope_theta"])
        q, k = rope(q, theta), rope(k, theta)
    distances = inputs["positions"][:, :, None] - inputs["positions"][:, None, :]
    allowed = distances >= 0
    if config.layer_types[layer_idx] == "sliding_attention":
        allowed &= distances < config.sliding_window
    allowed = allowed & valid[:, None, :] & valid[:, :, None]
    heads = []
    for head in range(qh):
        kv = head // (qh // kh)
        scores = (q[:, :, head] @ k[:, :, kv].transpose(-1, -2)) / math.sqrt(d)
        sink = weights["self_attn.softmax_offset"][head].expand(b, s, 1)
        probability = torch.cat((scores.masked_fill(~allowed, -torch.inf), sink), -1).softmax(-1)[..., :-1]
        heads.append(probability @ v[:, :, kv])
    attention = F.linear(torch.stack(heads, 2).reshape(b, s, qh * d), weights["self_attn.o_proj.weight"])
    residual = hidden + attention
    normalized_mlp = rms(residual, weights["post_attention_layernorm.weight"], eps)

    def mlp(values: torch.Tensor, prefix: str) -> torch.Tensor:
        gate = F.linear(values, weights[prefix + ".gate_proj.weight"])
        up = F.linear(values, weights[prefix + ".up_proj.weight"])
        return F.linear(F.silu(gate) * up, weights[prefix + ".down_proj.weight"])

    result = {"normalized_input": normalized, "attention_projected": attention, "normalized_mlp_input": normalized_mlp}
    if layer_idx in config.mlp_only_layers:
        result["output"] = residual + mlp(normalized_mlp, "mlp")
        return result
    flat = normalized_mlp.reshape(-1, h)
    scores = torch.sigmoid(F.linear(flat, weights["mlp.gate.weight"]))
    selected, ids = scores.topk(config.num_experts_per_tok, dim=-1, sorted=False)
    probabilities = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    order = ids.argsort(dim=-1)
    result["canonical_route_indices"] = ids.gather(1, order).reshape(b, s, -1)
    result["canonical_route_weights"] = probabilities.gather(1, order).reshape(b, s, -1)
    routed = torch.zeros_like(flat)
    for expert in range(config.num_experts):
        coefficient = ((ids == expert) * probabilities).sum(-1, keepdim=True)
        gate = flat @ weights["mlp.experts.gate_proj.weight"][expert].T
        up = flat @ weights["mlp.experts.up_proj.weight"][expert].T
        routed = routed + (F.silu(gate) * up) @ weights["mlp.experts.down_proj.weight"][expert].T * coefficient
    output = routed + mlp(flat, "mlp.shared_expert")
    result["output"] = residual + output.reshape(b, s, h)
    return result


def _block_and_reference(
    case_id: str,
) -> tuple[FlintConfig, FlintDecoderLayer, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build a seeded native block and evaluate the independent reference on the same weights and inputs."""
    config, layer_idx, inputs = _case(case_id)
    layer = FlintDecoderLayer(config, layer_idx, _backend())
    _seed_parameters(layer, seed=7 + layer_idx)
    source = _source_parameters(layer)
    reference = _reference_block(config, layer_idx, source, inputs)
    return config, layer, inputs, source, reference


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_block_forward_and_every_parameter_gradient_match_independent_reference(case_id: str) -> None:
    config, layer, inputs, source, reference = _block_and_reference(case_id)
    hidden = inputs["input"].clone().requires_grad_()
    reference_input = inputs["input"].clone().requires_grad_()
    valid = inputs["valid_tokens"]
    output = layer(hidden, position_ids=inputs["positions"], attention_mask=valid)
    # Shared NeMo MoE drops padded routed rows; their outputs are unspecified.
    torch.testing.assert_close(output[valid], reference["output"][valid], atol=1e-5, rtol=0)
    objective = (output * inputs["upstream"]).sum()
    objective.backward()
    expected = _reference_block(config, _CASES[case_id], source, {**inputs, "input": reference_input})["output"]
    (expected * inputs["upstream"]).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_input.grad, atol=1e-5, rtol=0)
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, _mapped_gradient(name, source), atol=1e-5, rtol=0, msg=name)
    assert torch.count_nonzero(hidden.grad[~valid]) == 0
    assert layer.self_attn.q_proj.weight.shape[0] != config.hidden_size
    assert layer.self_attn._fp32_params.softmax_offset.requires_grad
    assert layer.self_attn._fp32_params.softmax_offset.dtype == torch.float32


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_attention_alone_matches_independent_projected_reference(case_id: str) -> None:
    _, layer, inputs, _, reference = _block_and_reference(case_id)
    actual = layer.self_attn(
        reference["normalized_input"].detach(), position_ids=inputs["positions"], attention_mask=inputs["valid_tokens"]
    )
    torch.testing.assert_close(actual, reference["attention_projected"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("case_id", ("moe_sliding", "moe_full"))
def test_common_moe_preserves_unbiased_sigmoid_routes_and_ungated_shared_width(case_id: str) -> None:
    config, layer, inputs, _, reference = _block_and_reference(case_id)
    assert isinstance(layer.mlp, MoE)
    assert layer.mlp.gate.e_score_correction_bias is None
    assert layer.mlp.gate.bias is None
    assert layer.mlp.shared_expert_gate is None
    assert layer.mlp.shared_experts.gate_proj.weight.shape == (
        config.shared_expert_intermediate_size,
        config.hidden_size,
    )
    assert layer.mlp.gate.aux_loss_coeff == layer.mlp.gate.bias_update_factor == 0
    normalized = reference["normalized_mlp_input"].detach().flatten(0, 1)
    valid = inputs["valid_tokens"].flatten()
    weights, indices, aux = layer.mlp.gate(normalized, valid, None)
    order = indices.argsort(dim=-1)
    torch.testing.assert_close(
        indices.gather(1, order), reference["canonical_route_indices"].flatten(0, 1), atol=0, rtol=0
    )
    torch.testing.assert_close(
        weights.gather(1, order), reference["canonical_route_weights"].flatten(0, 1), atol=1e-6, rtol=0
    )
    assert weights.dtype == torch.float32
    assert aux is None


def _uniform_attention(*, sliding: bool) -> FlintAttention:
    """Create Q/K-zero attention with an observable copy of each query head's value."""
    config, _, _ = _case("moe_sliding" if sliding else "dense_full")
    config.sliding_window = 513
    attention = FlintAttention(config, 2 if sliding else 0, _backend())
    attention.init_weights(torch.device("cpu"))
    with torch.no_grad():
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
            projection.weight.zero_()
        for head in range(config.num_key_value_heads):
            attention.v_proj.weight[head * config.head_dim, 0] = 1
        for head in range(config.num_attention_heads):
            attention.o_proj.weight[head, head * config.head_dim] = 1
    return attention


def test_unscaled_sink_zero_value_and_analytic_gradient() -> None:
    attention = _uniform_attention(sliding=False)
    hidden = torch.zeros(1, 1, attention.hidden_size)
    hidden[..., 0] = 2
    output = attention(hidden)
    torch.testing.assert_close(
        output[..., : attention.num_heads], torch.ones(1, 1, attention.num_heads), atol=0, rtol=0
    )
    (gradient,) = torch.autograd.grad(output.sum(), attention._fp32_params.softmax_offset)
    torch.testing.assert_close(gradient, torch.full_like(gradient, -0.5), atol=0, rtol=0)
    with torch.no_grad():
        attention._fp32_params.softmax_offset.fill_(math.log(3))
    torch.testing.assert_close(
        attention(hidden)[..., : attention.num_heads], torch.full((1, 1, attention.num_heads), 0.5), atol=1e-7, rtol=0
    )
    masked = attention(hidden, attention_mask=torch.zeros(1, 1, dtype=torch.bool))
    assert torch.isfinite(masked).all() and torch.count_nonzero(masked) == 0


def test_native_window_boundary_distinguishes_full_from_sliding() -> None:
    sliding = _uniform_attention(sliding=True)
    full = _uniform_attention(sliding=False)
    hidden = torch.zeros(1, 514, sliding.hidden_size)
    hidden[:, 0, 0] = 1
    local, global_output = sliding(hidden), full(hidden)
    torch.testing.assert_close(
        local[0, 512, : sliding.num_heads], torch.full((sliding.num_heads,), 1 / 514), atol=1e-8, rtol=0
    )
    assert torch.count_nonzero(local[0, 513]) == 0
    torch.testing.assert_close(
        global_output[0, 513, : full.num_heads], torch.full((full.num_heads,), 1 / 515), atol=1e-8, rtol=0
    )


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_meta_materialization_and_initializer_leave_finite_trainable_components(case_id: str) -> None:
    config, layer_idx, _ = _case(case_id)
    with torch.device("meta"):
        layer = FlintDecoderLayer(config, layer_idx, _backend())
    layer.to_empty(device="cpu")
    layer.init_weights(torch.device("cpu"))
    assert all(torch.isfinite(parameter).all() for parameter in layer.parameters())
    hidden = torch.randn(1, 4, config.hidden_size, requires_grad=True)
    layer(hidden).square().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters())
    assert isinstance(layer.mlp, MLP if case_id == "dense_full" else MoE)


def test_supported_dtype_cast_keeps_trainable_sink_holder_in_fp32() -> None:
    attention = _uniform_attention(sliding=False)
    cast_model_to_dtype(attention, torch.bfloat16, skip_modules=("_fp32_params",))
    assert attention.q_proj.weight.dtype == torch.bfloat16
    assert attention._fp32_params.softmax_offset.dtype == torch.float32
    assert attention._fp32_params.softmax_offset.requires_grad


def test_outer_cpu_autocast_preserves_fp32_sink_math_and_gradients() -> None:
    config, layer, inputs, _, reference = _block_and_reference("moe_sliding")
    attention = layer.self_attn
    cast_model_to_dtype(attention, torch.bfloat16, skip_modules=("_fp32_params",))
    hidden = reference["normalized_input"].detach().bfloat16()
    valid = inputs["valid_tokens"]
    baseline = attention(hidden, attention_mask=valid)
    upstream = torch.linspace(-0.8, 1.2, baseline.numel()).reshape_as(baseline).bfloat16()
    (baseline_grad,) = torch.autograd.grad((baseline * upstream).sum(), attention._fp32_params.softmax_offset)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = attention(hidden, attention_mask=valid)
    (actual_grad,) = torch.autograd.grad((actual * upstream).sum(), attention._fp32_params.softmax_offset)
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
    torch.testing.assert_close(actual_grad, baseline_grad, atol=1e-7, rtol=0)
    assert actual_grad.dtype == torch.float32
    assert torch.count_nonzero(actual_grad) == config.num_attention_heads


@pytest.mark.parametrize(
    "override",
    [
        dict(attn="flex"),
        dict(linear="quack"),
        dict(linear="te"),
        dict(rms_norm="te"),
        dict(rope="triton"),
        dict(experts="torch_mm_mxfp8"),
        dict(experts="te", dispatcher="deepep"),
        dict(gate_precision=torch.float32),
        dict(fake_balanced_gate=True),
    ],
)
def test_unsupported_backends_fail_before_component_execution(override: dict[str, Any]) -> None:
    config, _, _ = _case("dense_full")
    with pytest.raises(ValueError, match="Flint"):
        FlintAttention(config, 0, _backend(**override))


def test_invalid_padding_mask_is_rejected_instead_of_broadcasting() -> None:
    attention = _uniform_attention(sliding=False)
    hidden = torch.zeros(1, 3, attention.hidden_size)
    with pytest.raises(ValueError, match="attention_mask"):
        attention(hidden, attention_mask=torch.ones(1, 1, 3, 3))
    with pytest.raises(ValueError, match="zero/one"):
        attention(hidden, attention_mask=torch.full((1, 3), 0.5))


@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("mask_kind", ["valid", "holes", "empty_row", "documents"])
def test_actual_sdpa_matches_eager_outputs_inputs_and_every_parameter_gradient(
    layer_index: int,
    mask_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check FP32 [2,6,12] outputs and adjoints, including trainable [4] sinks."""
    torch.manual_seed(194 + layer_index)
    config = tiny_config()
    eager, sdpa = (
        FlintAttention(config, layer_index, tiny_backend()),
        FlintAttention(config, layer_index, tiny_backend("sdpa")),
    )
    eager.init_weights(torch.device("cpu"))
    with torch.no_grad():
        eager._fp32_params.softmax_offset.copy_(torch.tensor([-0.7, 0.3, 0.8, -0.1]))
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    assert sdpa.state_dict().keys() == eager.state_dict().keys()
    valid = torch.ones(2, 6, dtype=torch.bool)
    if mask_kind == "holes":
        valid[0] = torch.tensor([False, True, False, True, True, False])
    elif mask_kind == "empty_row":
        valid[0] = False
    kwargs = {"attention_mask": valid}
    if mask_kind == "documents":
        kwargs["document_ids"] = torch.tensor([[1, 1, 1, 2, 2, 0], [1, 1, 2, 2, 3, 3]])
    upstream, original = torch.randn(2, 6, 12), torch.randn(2, 6, 12)
    calls = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def observe(*args: object, **options: object) -> torch.Tensor:
        calls.append(True)
        return original_sdpa(*args, **options)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    eager_input, sdpa_input = original.clone().requires_grad_(), original.clone().requires_grad_()
    expected, actual = eager(eager_input, **kwargs), sdpa(sdpa_input, **kwargs)
    assert calls == [True]
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    eager_grad = torch.autograd.grad((expected * upstream).sum(), (eager_input, *eager.parameters()))
    sdpa_grad = torch.autograd.grad((actual * upstream).sum(), (sdpa_input, *sdpa.parameters()))
    for expected_grad, actual_grad in zip(eager_grad, sdpa_grad, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=3e-5)
    sink_index = list(dict(sdpa.named_parameters())).index("_fp32_params.softmax_offset") + 1
    assert torch.count_nonzero(sdpa_grad[sink_index]) == config.num_attention_heads


def test_explicit_source_schedule_preserves_prefix_layer_phase() -> None:
    """A four-layer prefix of 49 does not acquire a synthetic final full layer."""
    config = tiny_config(
        num_hidden_layers=4,
        schedule_num_hidden_layers=49,
        layer_types=None,
        no_rope_layer_indices=None,
        mlp_only_layers=None,
    )
    assert config.layer_types[:4] == ["full_attention", "full_attention", "sliding_attention", "sliding_attention"]
    assert len(config.layer_types) == 49
    assert config.no_rope_layer_indices[-1] == 48
    restored = FlintConfig.from_dict(config.to_dict())
    assert restored.layer_types == config.layer_types
    assert restored.num_hidden_layers == 4 and restored.schedule_num_hidden_layers == 49
    with pytest.raises(ValueError, match="active depth"):
        tiny_config(num_hidden_layers=5, schedule_num_hidden_layers=4)


def test_explicit_fp32_residual_keeps_bf16_branches_and_trainable_sink() -> None:
    """A BF16 dense block emits FP32 residual [2,6,12] under the explicit policy."""
    config = tiny_config(dtype=torch.bfloat16, residual_dtype="fp32")
    layer = FlintDecoderLayer(config, 0, tiny_backend("sdpa"))
    layer.init_weights(torch.device("cpu"))
    x = torch.randn(2, 6, 12, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    assert output.dtype == torch.float32
    output.square().sum().backward()
    assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn._fp32_params.softmax_offset.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("projection_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("residual_policy", ["fp32", "bf16"])
def test_explicit_residual_dtype_matches_each_block_and_pipeline_boundary(
    projection_dtype: torch.dtype, residual_policy: str
) -> None:
    """Intermediate PP metadata agrees with dense/MoE outputs and norm inputs."""
    torch.manual_seed(437501)
    config = tiny_config(
        dtype=projection_dtype,
        residual_dtype=residual_policy,
        mlp_only_layers=[0],
        router_aux_loss_coef=0.0,
    )
    model = config.build(backend=tiny_backend("sdpa"))
    model.initialize_weights()
    # Use the same pruned boundary as a non-final, non-first pipeline stage.
    model.model.embed_tokens = model.model.norm = model.lm_head = None
    expected_dtype = torch.float32 if residual_policy == "fp32" else torch.bfloat16
    inputs, outputs = model.get_pipeline_stage_metas(
        is_first=False, microbatch_size=2, seq_len=6, dtype=projection_dtype
    )
    assert inputs[0].dtype == outputs[0].dtype == expected_dtype
    hidden = torch.randn(2, 6, 12, dtype=expected_dtype, requires_grad=True)
    norm_dtypes = []

    def observe_norm(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Observe residual inputs without changing values or their gradients.

        Args:
            module: Actual decoder RMSNorm module.
            args: One residual tensor of shape [batch, sequence, hidden].
        """
        norm_dtypes.append(args[0].dtype)

    handles = [
        norm.register_forward_pre_hook(observe_norm)
        for layer in model.model.layers
        for norm in (layer.input_layernorm, layer.post_attention_layernorm)
    ]
    actual = model(inputs_embeds=hidden).logits
    for handle in handles:
        handle.remove()
    assert actual.shape == outputs[0].shape
    assert actual.dtype == outputs[0].dtype
    assert norm_dtypes == [expected_dtype] * (2 * config.num_hidden_layers)
    actual.float().square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and torch.count_nonzero(hidden.grad)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_attention_precision_policy_is_explicit_typed_and_te_only() -> None:
    """The optional FP16 boundary serializes without changing default precision."""
    config = tiny_config(attention_compute_dtype="fp16")
    restored = FlintConfig.from_dict(config.to_dict())
    assert restored.attention_compute_dtype == "fp16"
    assert tiny_config().attention_compute_dtype is None
    with pytest.raises(ValueError, match="attention_compute_dtype"):
        tiny_config(attention_compute_dtype="float32")
    with pytest.raises(ValueError, match="TE backend"):
        FlintAttention(config, 0, tiny_backend("sdpa"))
