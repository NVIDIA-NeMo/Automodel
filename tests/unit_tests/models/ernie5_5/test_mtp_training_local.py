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

"""Observable CPU contracts for one-depth shared-body ERNIE MTP training."""

import copy
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, mtp_decoder_config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.mtp_training import Ernie5_5MTPForCausalLM
from tests.unit_tests.models.ernie5_5._mtp_reference_loss import ernie5_5_mtp_loss

pytestmark = pytest.mark.cpu_only

OWNED_PARAMETERS = {
    "mtp.enorm.weight",
    "mtp.hnorm.weight",
    "mtp.e_proj.weight",
    "mtp.h_proj.weight",
    "mtp.hyper_head.hc_fn",
    "mtp.hyper_head.hc_base",
    "mtp.hyper_head.hc_scale",
    "mtp.final_norm.weight",
}


@pytest.mark.parametrize("invalid_ratio", [True, 7])
def test_mtp_decoder_config_validates_earlier_decoder_schedule_entries(invalid_ratio) -> None:
    """Reject invalid earlier layers even when the shared final DSA/MoE pair is valid."""
    source = config()
    source.compress_ratios[0] = invalid_ratio
    with pytest.raises(ValueError, match="Only HCA ratio 128 and DSA ratio -2 are currently supported"):
        mtp_decoder_config(source)
    assert source.num_nextn_predict_layers == 1 and source.mtp_shared_last_layer
    assert source.compress_ratios[0] == invalid_ratio


@pytest.mark.parametrize("change", ["last_decoder_block_hca", "mtp_entry_hca", "last_decoder_block_dense"])
def test_mtp_decoder_config_requires_the_last_block_to_be_dsa_and_moe(change: str) -> None:
    """Reject a schedule the layer plan accepts but whose last decoder block cannot be the shared MTP body."""
    source = config()
    if change == "last_decoder_block_hca":
        source.compress_ratios = [-2, 128, -2]
    elif change == "mtp_entry_hca":
        source.compress_ratios = [-2, -2, 128]
    else:
        source.first_k_dense_replace = 2
    with pytest.raises(ValueError, match="The shared MTP body must be the last DSA/MoE decoder block"):
        mtp_decoder_config(source)
    assert source.num_nextn_predict_layers == 1 and source.mtp_shared_last_layer


def config(dtype: str = "float32") -> Ernie5_5Config:
    """Return two tiny DSA blocks, dense then MoE, and one shared MTP depth."""
    return Ernie5_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        mtp_shared_last_layer=True,
        compress_ratios=[-2, -2, -2],
        max_position_embeddings=1024,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        first_k_dense_replace=1,
        pad_token_id=0,
        torch_dtype=dtype,
    )


def backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")


def model(dtype: str = "float32", seed: int = 19) -> Ernie5_5MTPForCausalLM:
    torch.manual_seed(seed)
    result = Ernie5_5MTPForCausalLM(config(dtype), backend())
    result.initialize_weights()
    return result


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return input IDs and already-shifted labels, both [1, 8]."""
    return torch.arange(1, 9)[None], torch.tensor([[2, 3, 4, 5, 6, 7, 8, -100]])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_main_logits_match_existing_decoder_and_body_is_called_twice(dtype: str) -> None:
    candidate = model(dtype)
    baseline = Ernie5_5_MoeForCausalLM(mtp_decoder_config(config(dtype)), backend())
    baseline.load_state_dict(
        {key: value for key, value in candidate.state_dict().items() if not key.startswith("mtp.")}
    )
    tokens, _ = batch()
    hook = Mock(return_value=None)
    handle = candidate.model.layers[-1].register_forward_pre_hook(hook)
    result = candidate(tokens)
    handle.remove()
    assert hook.call_count == 2
    assert hook.call_args_list[0].args[1][0].dtype == torch.float32
    assert hook.call_args_list[1].args[1][0].dtype == getattr(torch, dtype)
    torch.testing.assert_close(result.logits, baseline(tokens).logits, rtol=0, atol=0)
    assert result.mtp_per_depth_logits[0].shape == (1, 8, 32)
    assert torch.isfinite(result.mtp_per_depth_logits[0]).all()
    owned = list(candidate.mtp.named_parameters())
    assert len(owned) == 8
    parameters = list(candidate.named_parameters(remove_duplicate=False))
    assert len({id(value) for _, value in parameters}) == len(parameters)


def test_shift_operates_on_token_ids_not_zero_filled_embeddings() -> None:
    candidate = model()
    hook = Mock(return_value=None)
    handle = candidate.model.embed_tokens.register_forward_pre_hook(hook)
    candidate(batch()[0])
    handle.remove()
    assert hook.call_count == 2
    assert hook.call_args_list[1].args[1][0].tolist() == [[2, 3, 4, 5, 6, 7, 8, 0]]


def test_joint_gradient_equals_sum_and_all_owned_parameters_receive_gradients() -> None:
    reference = model()
    state = copy.deepcopy(reference.state_dict())
    runs = []
    tokens, labels = batch()
    for phase in ("joint", "main", "auxiliary"):
        candidate = model()
        candidate.load_state_dict(state)
        output = candidate(tokens)
        main_loss = F.cross_entropy(output.logits.flatten(0, 1), labels.flatten())
        loss = main_loss
        if phase != "main":
            auxiliary = ernie5_5_mtp_loss(output.mtp_per_depth_logits[0], labels).loss
            loss = auxiliary if phase == "auxiliary" else main_loss + auxiliary
        loss.backward()
        runs.append(
            {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in candidate.named_parameters()
            }
        )
    for name, joint in runs[0].items():
        if name in OWNED_PARAMETERS:
            assert joint is not None, name
            assert torch.isfinite(joint).all() and bool(joint.abs().sum() > 0), name
        if joint is None:
            assert runs[1][name] is None and runs[2][name] is None
            continue
        main = torch.zeros_like(joint) if runs[1][name] is None else runs[1][name]
        auxiliary = torch.zeros_like(joint) if runs[2][name] is None else runs[2][name]
        torch.testing.assert_close(joint, main + auxiliary, rtol=2e-4, atol=3e-7)
        if name.startswith("mtp."):
            assert torch.isfinite(joint).all() and bool(joint.abs().sum() > 0), name
    assert {name for name in runs[0] if name.startswith("mtp.")} == OWNED_PARAMETERS
    assert all(runs[2][name] is not None and bool(runs[2][name].abs().sum() > 0) for name in OWNED_PARAMETERS)
    for name in ("model.layers.0.mlp.down_proj.weight", "model.layers.1.moe.fc2_latent_proj.weight"):
        assert runs[2][name] is not None and bool(runs[2][name].abs().sum() > 0), name


def test_post_roll_row_denominators_match_explicit_token_oracle_and_accumulation() -> None:
    torch.manual_seed(2)
    logits = torch.randn(3, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, -100], [1, -100, 2, -100, -100], [-100] * 5])
    result = ernie5_5_mtp_loss(logits, labels)
    expected = []
    for row in range(3):
        terms = [
            -F.log_softmax(logits[row, pos].float(), dim=-1)[labels[row, pos + 1]]
            for pos in range(4)
            if labels[row, pos + 1] != -100
        ]
        expected.append(torch.stack(terms).mean() if terms else logits[row].sum() * 0)
    oracle = 0.1 * torch.stack(expected).mean()
    torch.testing.assert_close(result.loss, oracle)
    assert result.label_counts.tolist() == [3, 1, 0]
    assert result.targets[:, -1].tolist() == [-100, -100, -100]
    accumulated = sum(ernie5_5_mtp_loss(logits[i : i + 1], labels[i : i + 1]).loss for i in range(3)) / 3
    torch.testing.assert_close(accumulated, oracle)
    expected_gradient = torch.autograd.grad(oracle, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(result.loss, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("sequence", [1, 5])
def test_zero_supervision_is_differentiable_finite_zero(sequence: int) -> None:
    logits = torch.randn(2, sequence, 7, requires_grad=True)
    labels = torch.full((2, sequence), -100)
    loss = ernie5_5_mtp_loss(logits, labels).loss
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(loss)
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_document_boundaries_do_not_shift_a_target_into_another_document() -> None:
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[2, -100, 4, 5, -100]])
    documents = torch.tensor([[1, 1, 2, 2, 2]])
    result = ernie5_5_mtp_loss(logits, labels, document_ids=documents)
    assert result.targets.tolist() == [[-100, -100, 5, -100, -100]]
    assert result.label_counts.item() == 1


def test_meta_materialization_preserves_unique_owners_and_initializes_all_state() -> None:
    with torch.device("meta"):
        candidate = Ernie5_5MTPForCausalLM(config(), backend())
    candidate.to_empty(device="cpu")
    candidate.initialize_weights()
    parameters = list(candidate.named_parameters(remove_duplicate=False))
    assert len(parameters) == len({id(value) for _, value in parameters})
    assert all(torch.isfinite(value).all() for _, value in parameters)
    assert not candidate.mtp.router_state.e_score_correction_bias.any()
    assert torch.isfinite(candidate(batch()[0]).mtp_per_depth_logits[0]).all()


def test_native_and_hf_roundtrip_and_conflicting_alias_rejection() -> None:
    candidate = model()
    candidate.mtp.router_state.e_score_correction_bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    native = copy.deepcopy(candidate.state_dict())
    restored = model(seed=88)
    restored.load_state_dict(native, strict=True)
    exported = candidate.state_dict_adapter.to_hf(native)
    target = "model.layers.2."
    assert exported[target + "block_sparse_moe.e_score_correction_bias"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    converted = candidate.state_dict_adapter.from_hf(exported)
    restored.load_state_dict(converted, strict=True)
    for key, value in native.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert candidate.state_dict_adapter.supports_low_memory_dcp_load
    shared = next(key for key in exported if key.startswith(target) and ".self_attn." in key)
    broken = dict(exported)
    broken[shared] = broken[shared].clone() + 1
    with pytest.raises(ValueError, match="differs from its decoder"):
        candidate.state_dict_adapter.from_hf(broken)
    broken = dict(exported)
    del broken[target + "enorm.weight"]
    with pytest.raises(ValueError, match="exactly cover"):
        candidate.state_dict_adapter.from_hf(broken)


def test_eval_main_only_and_unsupported_configuration_rejected() -> None:
    candidate = model().eval()
    assert candidate(batch()[0]).mtp_per_depth_logits is None
    with pytest.raises(ValueError, match="no KV cache"):
        candidate(batch()[0], use_cache=True)
    with pytest.raises(ValueError, match="one shared-last-layer"):
        Ernie5_5MTPForCausalLM(mtp_decoder_config(config()), backend())


def test_hf_mtp_hyperconnection_names_match_released_spelling() -> None:
    """Check published names independently of this adapter's symmetric roundtrip.

    Release68d19ee5 uses transformer_layer.* only for the two MTP mHC sites.
    These ten names are distinct from the decoder's unwrapped mHC names.
    """
    candidate = model()
    exported = candidate.state_dict_adapter.to_hf(candidate.state_dict())
    canonical = dict(exported)
    for site in ("mlp_hyper_connection", "self_attention_hyper_connection"):
        for tensor in ("mapping_proj.weight", "alpha_pre", "alpha_post", "alpha_res", "bias"):
            suffix = f"{site}.{tensor}"
            released_name = "model.layers.2.transformer_layer." + suffix
            wrong_name = "model.layers.2." + suffix
            assert released_name in exported and wrong_name not in exported
            # Build the incoming MTP value from its decoder owner using the
            # literal release spelling, independently of the MTP export route.
            canonical[released_name] = exported["model.layers.1." + suffix]
    converted = candidate.state_dict_adapter.from_hf(canonical)
    candidate.load_state_dict(converted, strict=True)
    wrong_name = "model.layers.2.self_attention_hyper_connection.mapping_proj.weight"
    canonical[wrong_name] = canonical.pop(
        "model.layers.2.transformer_layer.self_attention_hyper_connection.mapping_proj.weight"
    )
    with pytest.raises(ValueError, match="exactly cover"):
        candidate.state_dict_adapter.from_hf(canonical)


def _released_owned_tensors(candidate: Ernie5_5MTPForCausalLM) -> dict[str, torch.Tensor]:
    """Return the nine MTP-owned tensors under their literal released names and layouts.

    The fixture has two decoder layers, so the MTP layer is ``model.layers.2``.
    The released head projection is [streams * hidden, streams], the transpose
    of the native parameter.
    """
    mtp = candidate.mtp
    return {
        "model.layers.2.enorm.weight": mtp.enorm.weight,
        "model.layers.2.hnorm.weight": mtp.hnorm.weight,
        "model.layers.2.e_proj.weight": mtp.e_proj.weight,
        "model.layers.2.h_proj.weight": mtp.h_proj.weight,
        "model.layers.2.hc_head_fn": mtp.hyper_head.hc_fn.T,
        "model.layers.2.hc_head_base": mtp.hyper_head.hc_base,
        "model.layers.2.hc_head_scale": mtp.hyper_head.hc_scale,
        "model.layers.2.shared_head.norm.weight": mtp.final_norm.weight,
        "model.layers.2.block_sparse_moe.e_score_correction_bias": mtp.router_state.e_score_correction_bias,
    }


def test_owned_mtp_tensors_use_the_literal_released_names_in_both_directions() -> None:
    """Pin the nine released names independently of the adapter's own name table.

    Every other expected checkpoint in these tests is produced by that table, so
    exchanged or misspelled entries would agree with themselves in a round trip.
    """
    torch.manual_seed(3)
    source = model(seed=7)
    with torch.no_grad():
        for parameter in source.mtp.parameters():
            parameter.copy_(torch.randn_like(parameter))
        source.mtp.router_state.e_score_correction_bias.copy_(torch.randn(4))
    released = _released_owned_tensors(source)
    assert len(released) == 9
    exported = source.state_dict_adapter.to_hf(source.state_dict())
    for name, value in released.items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0, msg=name)

    # Incoming direction: a mapping whose nine MTP entries are written by hand
    # under the released names, loaded into a model built from another seed.
    target = model(seed=88)
    incoming = dict(target.state_dict_adapter.to_hf(target.state_dict()))
    written = {name: torch.randn_like(value) for name, value in released.items()}
    incoming.update(written)
    target.load_state_dict(target.state_dict_adapter.from_hf(incoming), strict=True)
    for name, value in _released_owned_tensors(target).items():
        torch.testing.assert_close(value, written[name], rtol=0, atol=0, msg=name)


def _nonidentity_model(dtype: str = "float32") -> Ernie5_5MTPForCausalLM:
    """Make the independent MTP stages and all four MultiMax pieces observable."""
    candidate = model(dtype)
    with torch.no_grad():
        candidate.mtp.enorm.weight.copy_(torch.linspace(0.5, 1.5, 16))
        candidate.mtp.hnorm.weight.copy_(torch.linspace(1.6, 0.6, 16))
        candidate.mtp.final_norm.weight.copy_(torch.linspace(0.7, 1.3, 16))
        candidate.mtp.hyper_head.hc_base.copy_(torch.tensor([-0.4, 0.3]))
        candidate.mtp.hyper_head.hc_scale.fill_(0.7)
        candidate.mtp.hyper_head.hc_fn.mul_(2.5)
        candidate.model.hyper_head.hc_base.copy_(torch.tensor([0.8, -0.5]))
        candidate.model.layers[-1].moe.gate.e_score_correction_bias.copy_(torch.tensor([10.0, 10.0, -10.0, -10.0]))
        candidate.mtp.router_state.e_score_correction_bias.copy_(torch.tensor([-10.0, -10.0, 10.0, 10.0]))
        raw = candidate(batch()[0]).mtp_per_depth_logits[0].float().flatten()
        candidate.model.multimax.multimax_ranges.copy_(
            torch.quantile(raw, torch.tensor([0.4, 0.6, 0.35, 0.65])) + 0.00037
        )
        candidate.model.multimax.multimax_ts.copy_(torch.tensor([0.3, -0.2, 0.25, 0.15]))
    return candidate


def _norm_definition(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rounded = value.to(weight.dtype).float()
    return (rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + eps) * weight.float()).to(weight.dtype)


def _multimax_definition(value: torch.Tensor, ranges: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    z, r, t = value.float(), ranges.float(), scales.float()

    # Choose the zero subgradient at a knot, including BF16-quantized ties.
    def positive(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.zeros_like(x))

    return (
        z
        + t[0] * positive(r[0] - z)
        + t[1] * positive(z - r[1])
        + t[2] * positive(r[2] - z).square()
        + t[3] * positive(z - r[3]).square()
    ).to(value.dtype)


def _mtp_definition(candidate: Ernie5_5MTPForCausalLM, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the layer definition without calling the MTP implementation.

    The already-validated production decoder/body are the common primitives.
    Shift, three norms, both projections, stream fusion, learned contraction,
    vocabulary projection and MultiMax are expressed independently. Source:
    Bridge60f7921a ERNIE MTP definition, Apache2.0; no oracle code is imported.
    """
    captured = []
    handle = candidate.model.layers[-1].register_forward_hook(lambda _m, _a, value: captured.append(value))
    candidate.model(tokens)
    handle.remove()
    streams = captured[0]
    shifted = torch.cat((tokens[:, 1:], torch.zeros_like(tokens[:, :1])), dim=1)
    embedding = F.embedding(shifted, candidate.model.embed_tokens.weight)
    mtp, eps = candidate.mtp, candidate.config.rms_norm_eps
    e = F.linear(_norm_definition(embedding, mtp.enorm.weight, eps), mtp.e_proj.weight)
    h = F.linear(_norm_definition(streams, mtp.hnorm.weight, eps), mtp.h_proj.weight)
    body = candidate.model.layers[-1](
        h + e.unsqueeze(-2),
        candidate.model.attention_inputs(tokens),
        routing_state=mtp.router_state,
    )
    x = body.float()
    flat = x.flatten(start_dim=2)
    inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    head = mtp.hyper_head
    gates = torch.sigmoid(F.linear(flat, head.hc_fn) * inverse_rms * head.hc_scale + head.hc_base) + eps
    contracted = (gates.unsqueeze(-1) * x).sum(-2).to(body.dtype)
    hidden = _norm_definition(contracted, mtp.final_norm.weight, eps)
    raw = F.linear(hidden, candidate.model.lm_head.weight)
    mm = candidate.model.multimax
    return _multimax_definition(raw, mm.multimax_ranges, mm.multimax_ts), raw


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mtp_forward_matches_independent_layer_definition(dtype: str) -> None:
    actual = _nonidentity_model(dtype)
    reference = model(dtype)
    reference.load_state_dict(actual.state_dict())
    tokens, labels = batch()
    output = actual(tokens)
    expected, raw = _mtp_definition(reference, tokens)
    torch.testing.assert_close(output.mtp_per_depth_logits[0], expected, rtol=0, atol=0)
    assert not torch.equal(expected, raw)
    r = reference.model.multimax.multimax_ranges
    for piece in (r[0] - raw, raw - r[1], r[2] - raw, raw - r[3]):
        assert bool((piece > 0).any()) and bool((piece < 0).any())
    assert output.mtp_loss_scaling_factor == 0.1
    observed_loss = ernie5_5_mtp_loss(
        output.mtp_per_depth_logits[0], labels, coefficient=output.mtp_loss_scaling_factor
    ).loss
    # Targets advance once more; count the supervised targets after that shift.
    expected_loss = 0.1 * F.cross_entropy(expected[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten())
    torch.testing.assert_close(observed_loss, expected_loss)
    # Full named-gradient bounds and fixed independent directional checks live
    # in test_mtp_training_gradients.py; no weaker duplicate standard here.


@pytest.mark.parametrize("decoder_factor", [0.0, 0.01])
def test_distinct_correction_biases_select_distinct_experts_and_keep_main_logits(decoder_factor: float) -> None:
    candidate = model()
    gate = candidate.model.layers[-1].moe.gate
    gate.bias_update_factor = decoder_factor
    gate.e_score_correction_bias.copy_(torch.tensor([10.0, 10.0, -10.0, -10.0]))
    candidate.mtp.router_state.e_score_correction_bias.copy_(-gate.e_score_correction_bias)
    selections = []
    handle = gate.register_forward_hook(lambda _module, _args, output: selections.append(output[1].detach().clone()))
    try:
        first = candidate(batch()[0])
        candidate.mtp.router_state.e_score_correction_bias.copy_(gate.e_score_correction_bias)
        second = candidate(batch()[0])
    finally:
        handle.remove()
    assert len(selections) == 4
    for actual, experts in zip(selections, [(0, 1), (2, 3), (0, 1), (0, 1)], strict=True):
        torch.testing.assert_close(actual.sort(dim=-1).values, torch.tensor(experts).expand_as(actual))
    if decoder_factor:
        assert gate._cumulative_expert_load.tolist() == [16, 16, 0, 0]
    else:
        assert gate._cumulative_expert_load is None
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
    assert not torch.equal(first.mtp_per_depth_logits[0], second.mtp_per_depth_logits[0])


def test_shared_hf_exports_equal_literal_last_decoder_sources_and_exact_count() -> None:
    candidate = model()
    exported = candidate.state_dict_adapter.to_hf(candidate.state_dict())
    expected = {}
    for key, value in exported.items():
        if not key.startswith("model.layers.1.") or key.endswith("block_sparse_moe.e_score_correction_bias"):
            continue
        suffix = key.removeprefix("model.layers.1.")
        if suffix.startswith(("mlp_hyper_connection.", "self_attention_hyper_connection.")):
            suffix = "transformer_layer." + suffix
        expected["model.layers.2." + suffix] = value
    assert len(expected) == 49
    for key, value in expected.items():
        torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
    assert len([key for key in exported if key.startswith("model.layers.2.")]) == 49 + 9
