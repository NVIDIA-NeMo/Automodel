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

"""Ling / Bailing MoE V2 owns one router precision policy.

The policy has three parts: the dtype of the router projection
(``gate_precision``), the dtype of the selected routing weights handed to
expert compute (``router_weights_fp32``), and the fp32 score-correction
bias. Unlike DeepSeek, one class serves all three published checkpoints, so
the axis here is the config rather than the construction path: Ling-mini-2.0,
Ling-flash-2.0 and Ling-1T differ in their dense/MoE split and rope
convention, but share one router geometry and one ``route_scale``.

Param and Score already matched the reference on main; this PR moves Proj and
Out. All four are pinned so a later shared-router change cannot silently move
them back.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.models.ling_v2.model import BailingMoeV2ForCausalLM
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate

# One construction path, so no class parametrization: BailingMoeV2ForCausalLM is
# the only entry point and BailingMoeV2Model is the only inner model.
_INNER_MODEL = "nemo_automodel.components.models.ling_v2.model.BailingMoeV2Model"


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        tie_word_embeddings=False,
        hidden_size=16,
        vocab_size=32,
        torch_dtype="bfloat16",
    )


def _backend(*, gate_precision: torch.dtype | None = None) -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        gate_precision=gate_precision,
    )


def test_ling_gate_precision_defaults_to_fp32_without_mutating_backend():
    backend = _backend()

    with patch(_INNER_MODEL) as inner_model_cls:
        model = BailingMoeV2ForCausalLM(_config(), backend=backend)

    assert backend.gate_precision is None
    assert model.backend.gate_precision is torch.float32
    assert inner_model_cls.call_args.kwargs["backend"] is model.backend


def test_ling_gate_precision_respects_explicit_override():
    backend = _backend(gate_precision=torch.bfloat16)

    with patch(_INNER_MODEL) as inner_model_cls:
        model = BailingMoeV2ForCausalLM(_config(), backend=backend)

    assert model.backend is backend
    assert model.backend.gate_precision is torch.bfloat16
    assert inner_model_cls.call_args.kwargs["backend"] is backend


def test_ling_declares_the_score_correction_bias_as_strict_fp32():
    assert "e_score_correction_bias" in BailingMoeV2ForCausalLM._keep_in_fp32_modules_strict


# These cases construct for real, so a Gate exists and every stage can be
# asserted on the router the model actually built. Expert dims are tiny;
# n_group=2 keeps grouped routing on. Each variant needs num_hidden_layers >
# first_k_dense_replace so at least one MoE layer exists -- Ling-1T's first four
# layers are dense, so a two-layer 1T stand-in would have no router at all.
#   mini / flash : first_k_dense_replace=1, partial_rotary_factor
#   1T           : first_k_dense_replace=4, rotary_dim
# Routing is identical across all three published checkpoints -- 256 experts,
# top-8, 8 groups limited to 4, route_scale 2.5 -- so the variant axis here is
# the dense/MoE split and the rope convention, nothing the router itself sees.
_COMMON = dict(
    vocab_size=100,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    routed_scaling_factor=2.5,
    torch_dtype="bfloat16",
)

_MOE = dict(
    moe_intermediate_size=64,
    num_experts=8,
    num_shared_experts=1,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
)


def _mini_config() -> BailingMoeV2Config:
    return BailingMoeV2Config(
        **_COMMON,
        **_MOE,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )


def _flash_config() -> BailingMoeV2Config:
    # Same router geometry and dense split as mini; differs in depth and width,
    # neither of which the router sees.
    return BailingMoeV2Config(
        **_COMMON,
        **_MOE,
        num_hidden_layers=4,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )


def _1t_config() -> BailingMoeV2Config:
    # Ling-1T expresses half-RoPE as rotary_dim rather than partial_rotary_factor
    # and puts four dense layers ahead of the first MoE layer. Five layers: four
    # dense, one MoE.
    return BailingMoeV2Config(
        **_COMMON,
        **_MOE,
        num_hidden_layers=5,
        first_k_dense_replace=4,
        rotary_dim=8,
    )


def _router_config(**overrides) -> MoEConfig:
    """Tiny grouped router matching the Ling policy (n_expert_groups > 1)."""
    base = dict(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=2,
        n_limited_groups=1,
        train_gate=True,
        # Aux-loss-free: Ling loads the bias from the checkpoint and does not
        # update it, so force the buffer rather than creating it via an update
        # factor the way DeepSeek does.
        gate_bias_update_factor=0.0,
        force_e_score_correction_bias=True,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=2.5,
        norm_topk_prob=True,
        router_weights_fp32=True,
        dtype=torch.bfloat16,
    )
    base.update(overrides)
    return MoEConfig(**base)


def _run_gate(moe_config, gate_precision):
    torch.manual_seed(0)
    gate = Gate(moe_config, gate_precision=gate_precision)
    gate.weight.data.normal_(std=0.02)
    x = torch.randn(16, moe_config.dim, dtype=torch.bfloat16)
    token_mask = torch.ones(16, dtype=torch.bool)
    return gate, gate(x, token_mask, None)


def test_gate_hands_fp32_weights_to_expert_compute():
    """Out stage: with the Ling policy, bf16 in still yields fp32 weights."""
    gate, (weights, indices, _) = _run_gate(_router_config(), torch.float32)

    assert gate.score_dtype is torch.float32
    assert weights.dtype is torch.float32
    assert indices.shape == (16, 2)


def test_gate_without_policy_downcasts_weights():
    """The bug this PR fixes: type_as(x) hands bf16 weights to expert compute."""
    _, (weights, _, _) = _run_gate(_router_config(router_weights_fp32=False), None)

    assert weights.dtype is torch.bfloat16


def test_explicit_gate_precision_also_moves_scoring():
    """Score has no separate knob: overriding Proj to bf16 takes Score with it."""
    gate, _ = _run_gate(_router_config(), torch.bfloat16)

    assert gate.score_dtype is torch.bfloat16


# (config factory, index of the first MoE layer). The index is
# first_k_dense_replace: 1 for mini and flash, 4 for 1T.
_LING_MOE_CONFIG_CASES = (
    pytest.param(_mini_config, 1, id="mini"),
    pytest.param(_flash_config, 1, id="flash"),
    pytest.param(_1t_config, 4, id="1t"),
)


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_selected_router_weights_default_to_fp32(config_fn, first_moe_layer):
    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    assert model.model.moe_config.router_weights_fp32 is True


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_selected_router_weights_are_overridable(config_fn, first_moe_layer):
    model = BailingMoeV2ForCausalLM(
        config_fn(),
        backend=_backend(),
        moe_overrides={"router_weights_fp32": False},
    )
    assert model.model.moe_config.router_weights_fp32 is False


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_router_param_stays_in_model_dtype(config_fn, first_moe_layer):
    """Param stage: the reference stores the router weight in model dtype and casts at use."""
    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    assert model.model.moe_config.gate_dtype is None


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_router_stages_on_constructed_model(config_fn, first_moe_layer):
    """Param / Proj / Score / Out asserted on the Gate the model actually built."""
    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    gate = model.model.layers[str(first_moe_layer)].mlp.gate
    assert isinstance(gate, Gate)  # fails if first_k_dense_replace left this layer dense

    assert gate.weight.dtype is torch.bfloat16  # Param: model dtype, cast at use
    assert gate.e_score_correction_bias.dtype is torch.float32
    assert gate.gate_precision is torch.float32  # Proj
    assert gate.score_dtype is torch.float32  # Score
    assert gate.router_weights_fp32 is True

    torch.manual_seed(0)
    gate.weight.data.normal_(std=0.02)
    x = torch.randn(16, model.model.moe_config.dim, dtype=torch.bfloat16)
    weights, indices, _ = gate(x, torch.ones(16, dtype=torch.bool), None)
    assert weights.dtype is torch.float32  # Out
    assert indices.shape == (16, model.model.moe_config.n_activated_experts)


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_expert_compute_receives_fp32_weights(config_fn, first_moe_layer):
    """Out is defined at the expert boundary, not the Gate boundary.

    MoE.forward hands the gate's weights to self.experts uncast today, so the two
    coincide. Nothing else in this file would notice a type_as(x) appearing in
    between and undoing this PR, so pin the dtype where the audit defines the
    stage. This is also the only path here that runs a full MoE layer rather than
    calling the Gate in isolation.
    """
    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    moe = model.model.layers[str(first_moe_layer)].mlp

    seen = {}

    def capture(_module, args):
        """Record the dtype of the routing weights entering expert compute.

        Args:
            _module: The GroupedExperts module being called; unused.
            args: Positional forward args, all passed positionally by MoE.forward as
                (x: Tensor [tokens, dim], token_mask: Tensor [tokens],
                weights: Tensor [tokens, n_activated_experts],
                indices: Tensor [tokens, n_activated_experts]). Only ``weights``
                (index 2) is read; nothing is mutated.
        """
        seen["weights_dtype"] = args[2].dtype

    handle = moe.experts.register_forward_pre_hook(capture)
    try:
        torch.manual_seed(0)
        x = torch.randn(2, 8, model.model.moe_config.dim, dtype=torch.bfloat16)
        with torch.no_grad():
            moe(x)
    finally:
        handle.remove()

    assert seen["weights_dtype"] is torch.float32


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_score_correction_bias_survives_a_model_cast(config_fn, first_moe_layer):
    """test_ling_declares_the_score_correction_bias_as_strict_fp32 checks a list
    entry; this is the behavior that entry stands for.

    ling_v2/model.py routes through cast_model_to_dtype, which reads
    _keep_in_fp32_modules_strict. Tiny quantization errors in the bias change
    routing, so a bf16 cast must leave it alone.
    """
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype

    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    gate = model.model.layers[str(first_moe_layer)].mlp.gate
    assert gate.e_score_correction_bias.dtype is torch.float32

    cast_model_to_dtype(model, torch.bfloat16)

    assert gate.e_score_correction_bias.dtype is torch.float32
    assert gate.weight.dtype is torch.bfloat16  # Param does follow the cast


@pytest.mark.parametrize(("config_fn", "first_moe_layer"), _LING_MOE_CONFIG_CASES)
def test_ling_router_storage_policy_wins_over_incoming_dtypes(config_fn, first_moe_layer):
    """Param across a load: the module's storage policy decides, not the source tensor.

    A real checkpoint is out of scope for a unit test; Ling-mini's full load is
    covered by _real_forward_smoke.py and the maintainer's GPU harness. What this
    pins is the coercion contract a load depends on, fed the opposite of the
    published layout on purpose: an fp32 gate weight must not promote Param, and a
    bf16 bias must not demote the fp32 correction buffer.
    """
    model = BailingMoeV2ForCausalLM(config_fn(), backend=_backend())
    gate = model.model.layers[str(first_moe_layer)].mlp.gate
    dim = model.model.moe_config.dim

    torch.manual_seed(0)
    incoming = {
        "weight": torch.randn(gate.n_experts, dim, dtype=torch.float32),
        "e_score_correction_bias": torch.randn(gate.n_experts, dtype=torch.bfloat16),
    }
    # strict=True on purpose: the gate's state_dict is exactly these two keys, and
    # under strict=False a rename would make the load a no-op that still passes.
    gate.load_state_dict(incoming, strict=True)

    assert gate.weight.dtype is torch.bfloat16
    assert gate.e_score_correction_bias.dtype is torch.float32


# Revision of inclusionAI/Ling-mini-2.0 whose modeling_bailing_moe_v2.py the
# reference below was transcribed from. All three published Ling 2.0 checkpoints
# ship the same router code.
_REFERENCE_REVISION = "920c3fd9916e3d5e543fc4f609e827cad8a32983"


class _ReferenceBailingMoeV2Gate(nn.Module):
    """BailingMoeV2Gate, transcribed from the checkpoint-owned modeling file.

    Ling ships its modeling code inside the checkpoint and loads it with
    trust_remote_code, so there is no transformers.models.bailing_moe_v2 to
    import the way the DeepSeek policy test imports DeepseekV3MoE. The routing
    math is reimplemented rather than vendored so this test carries no
    third-party source; the revision it tracks is pinned above.

    The reference sigmoids with a redundant .float().type_as(logits) around it;
    logits are already fp32 there, so both casts are no-ops and are dropped here.
    """

    def __init__(self, *, num_experts, gating_dim, top_k, n_group, topk_group, routed_scaling_factor):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(num_experts, gating_dim))
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def group_limited_topk(self, scores):
        """Group-limited top-k over router scores.

        Args:
            scores: Tensor of shape [tokens, num_experts], the bias-adjusted sigmoid
                routing scores. Experts are laid out contiguously by group, so axis 1
                reshapes to [tokens, n_group, num_experts // n_group].

        Returns:
            Tuple of (probs, indices), each a Tensor of shape [tokens, top_k]. probs
            holds the masked scores of the selected experts (dtype follows ``scores``)
            and indices their int64 expert ids, both sorted by descending score.
        """
        num_tokens, _ = scores.size()
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked_scores, k=self.top_k, dim=-1)

    def forward(self, hidden_states):
        """Route tokens to experts.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading
                dimensions, which are flattened to a single ``tokens`` axis. Any
                floating dtype; the projection casts to fp32 internally.

        Returns:
            Tuple of (topk_idx, topk_weight, logits). topk_idx is an int64 Tensor of
            shape [tokens, top_k] holding selected expert ids; topk_weight is an fp32
            Tensor of shape [tokens, top_k] holding the normalized, ``route_scale``-
            scaled routing weights; logits is an fp32 Tensor of shape
            [tokens, num_experts]. None of the three alias ``hidden_states``.
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = torch.sigmoid(logits)
        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)
        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)
        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        return topk_idx, topk_weight * self.routed_scaling_factor, logits


def _paired_gates(cfg: MoEConfig, *, bias_mean: float, bias_std: float, seed: int = 0):
    """An Automodel Gate and a reference gate holding identical weights."""
    torch.manual_seed(seed)
    gate = Gate(cfg, gate_precision=torch.float32)
    gate.weight.data.normal_(std=0.02)
    gate.e_score_correction_bias.normal_(mean=bias_mean, std=bias_std)

    ref = _ReferenceBailingMoeV2Gate(
        num_experts=cfg.n_routed_experts,
        gating_dim=cfg.dim,
        top_k=cfg.n_activated_experts,
        n_group=cfg.n_expert_groups,
        topk_group=cfg.n_limited_groups,
        routed_scaling_factor=cfg.route_scale,
    )
    with torch.no_grad():
        ref.weight.copy_(gate.weight)  # bf16 -> fp32 is exact
        ref.expert_bias.copy_(gate.e_score_correction_bias)
    return gate, ref


# All three published Ling 2.0 checkpoints share one router geometry (256
# experts, top-8, 8 groups limited to 4) and one route_scale (2.5), so the shape
# axis carries one real case rather than three copies. Cases are (bias_mean,
# bias_std, router_overrides, weight_atol): the first two use the tiny 8-expert
# grouped router, the last matches the published routing shape at a reduced
# hidden dim so it stays a CPU test.
#
# weight_atol is None where the weights must match bitwise. Both sides now run a
# sorted top-k, so they normalize over the same order and every case measures an
# exact 0.0 delta -- including the published shape, whose 8-element sum would be
# order-sensitive if the orders ever diverged. The published shape keeps a small
# atol purely as insurance against topk tie-breaking; it is not covering a known
# error.
#
# Bias distributions are kept non-negative on purpose. The reference masks
# out-of-group experts with -inf while Automodel multiplies by a 0/1 mask; those
# agree only while the bias-adjusted sigmoid scores stay positive. A bias that
# drives scores negative would let a masked expert at 0 outrank an unmasked one
# -- a routing-semantics difference, not a precision one, and outside this PR.
_PARITY_CASES = (
    pytest.param(0.0, 0.0, {}, None, id="zero_bias"),
    pytest.param(0.2, 0.02, {}, None, id="positive_bias"),
    pytest.param(
        0.2,
        0.02,
        dict(n_routed_experts=256, n_activated_experts=8, n_expert_groups=8, n_limited_groups=4),
        1e-7,
        id="published_shape",
    ),
)


@pytest.mark.parametrize(("bias_mean", "bias_std", "router_overrides", "weight_atol"), _PARITY_CASES)
def test_ling_gate_matches_reference_router_grouped(bias_mean, bias_std, router_overrides, weight_atol):
    """Proj/Score/Out parity vs the pinned reference, grouped routing.

    The reference BailingMoeV2Gate does the fp32 projection and
    sigmoid/bias/group-mask/top-k/norm/scale. Automodel's Gate does the same,
    so the comparison drives both.
    """
    cfg = _router_config(**router_overrides)
    gate, ref = _paired_gates(cfg, bias_mean=bias_mean, bias_std=bias_std)
    gate.eval()
    ref.eval()

    x = torch.randn(64, cfg.dim, dtype=torch.bfloat16)
    w_am, i_am, _ = gate(x, torch.ones(64, dtype=torch.bool), None)

    with torch.no_grad():
        i_ref, w_ref, router_logits = ref(x)

    assert router_logits.dtype is torch.float32  # Proj: reference projects in fp32
    assert w_ref.dtype is torch.float32  # Out: reference never casts back
    assert w_am.dtype is torch.float32  # Out: this PR's default matches it

    # Both sides return sorted top-k, but sort each side's indices and apply the
    # same permutation to the weights so the comparison is order-independent.
    o_am, o_ref = i_am.argsort(dim=1), i_ref.argsort(dim=1)
    assert torch.equal(i_am.gather(1, o_am), i_ref.gather(1, o_ref))
    w_am_sorted, w_ref_sorted = w_am.gather(1, o_am), w_ref.gather(1, o_ref)
    if weight_atol is None:
        assert torch.equal(w_am_sorted, w_ref_sorted)
    else:
        # Measured max delta 0.0 at the published shape (max|w| 0.358). The
        # 8.94e-08 this tolerance was sized for came from the reference's
        # sorted=False expert top-k, which upstream does not use.
        assert torch.allclose(w_am_sorted, w_ref_sorted, atol=weight_atol, rtol=0)


def _weight_grad(cfg: MoEConfig, *, use_reference: bool, cotangent: torch.Tensor, x: torch.Tensor):
    """Gradient w.r.t. the gate weight under a fixed cotangent.

    The loss is (weights * g).sum() with a fixed g, not weights.sum(): with
    norm_topk_prob the selected weights sum to exactly route_scale per token, so
    weights.sum() is constant and its gradient is numerically zero. A test built
    on it would compare noise against noise and pass unconditionally.

    Only the gate weight carries gradient. The score-correction bias is a buffer
    on both sides.

    Args:
        cfg: Router geometry shared by both gates.
        use_reference: Select the pinned reference gate instead of Automodel's Gate.
        cotangent: Tensor of shape [tokens, n_activated_experts], the fixed upstream
            gradient applied to the index-sorted routing weights.
        x: Tensor of shape [tokens, dim], the router input. bf16 here on purpose, so
            the Proj/Score/Out casts under test are the ones that run.

    Returns:
        Tuple of (grad, indices). grad is an fp32 Tensor of shape
        [n_routed_experts, dim], the gradient w.r.t. the gate weight; indices is an
        int64 Tensor of shape [tokens, n_activated_experts], ascending per row so the
        two implementations' expert sets are directly comparable.
    """
    gate, ref = _paired_gates(cfg, bias_mean=0.2, bias_std=0.02)
    module = ref if use_reference else gate
    module.weight.requires_grad_(True)

    if use_reference:
        indices, weights, _ = module(x)
    else:
        weights, indices, _ = module(x, torch.ones(x.shape[0], dtype=torch.bool), None)

    order = indices.argsort(dim=1)
    (weights.gather(1, order) * cotangent).sum().backward()
    return module.weight.grad.detach().float(), indices.gather(1, order)


def test_ling_gate_gradients_match_reference():
    """The fp32 routing path reproduces the reference's gradient, not just its value.

    dtype=float32 here stores the gate weight in fp32, unlike the bf16 storage the
    Param stage pins above. That is deliberate: the reference's weight is fp32, and
    a bf16 grad on one side would force a tolerance and hide the exactness this
    asserts. The input is still bf16, so the Proj/Score/Out casts under test are the
    ones that run.
    """
    cfg = _router_config(dtype=torch.float32)
    torch.manual_seed(1)
    x = torch.randn(64, cfg.dim, dtype=torch.bfloat16)
    g = torch.randn(64, cfg.n_activated_experts)

    grad_ref, idx_ref = _weight_grad(cfg, use_reference=True, cotangent=g, x=x)
    grad_am, idx_am = _weight_grad(cfg, use_reference=False, cotangent=g, x=x)

    assert torch.equal(idx_am, idx_ref)  # same experts, or the grads are not comparable
    # Bit-identical to the reference, not merely close: measured max delta 0.0,
    # against 8.38e-03 with the policy off. torch.equal rather than allclose,
    # since a tolerance here would understate what the fp32 path achieves.
    assert torch.equal(grad_am, grad_ref)


def test_ling_gate_gradients_diverge_without_out_policy():
    """Negative control: without the Out policy the backward carries a bf16 truncation.

    Without this, the gradient comparison above could pass on a build where the
    policy does nothing at all.
    """
    torch.manual_seed(1)
    cfg_on = _router_config(dtype=torch.float32)
    cfg_off = _router_config(dtype=torch.float32, router_weights_fp32=False)
    x = torch.randn(64, cfg_on.dim, dtype=torch.bfloat16)
    g = torch.randn(64, cfg_on.n_activated_experts)

    grad_ref, _ = _weight_grad(cfg_on, use_reference=True, cotangent=g, x=x)
    grad_on, _ = _weight_grad(cfg_on, use_reference=False, cotangent=g, x=x)
    grad_off, _ = _weight_grad(cfg_off, use_reference=False, cotangent=g, x=x)

    delta_on = (grad_on - grad_ref).abs().max()
    delta_off = (grad_off - grad_ref).abs().max()

    # delta_on is 0.0 on this build, so a ratio test would reduce to "> 0" and
    # pass vacuously. Assert the two sides separately instead.
    assert delta_on == 0.0, f"policy-on path is no longer bit-exact: {delta_on:.3e}"
    assert delta_off > 1e-4, f"policy made no difference: off={delta_off:.3e}"


def test_ling_v2_has_a_single_construction_path():
    """No Ling sibling bypasses this __init__ the way DeepseekV32ForCausalLM does.

    V3.2 needed its own copy of the fp32 gate_precision default because it skips
    DeepseekV3ForCausalLM.__init__. If a Ling sibling class ever appears, it will
    need the same treatment, and this test fails at that moment.
    """
    from nemo_automodel._transformers.registry import ModelRegistry, resolve_custom_config_cls

    assert resolve_custom_config_cls("bailing_moe") is BailingMoeV2Config
    assert ModelRegistry.get_model_cls_from_model_arch("BailingMoeV2ForCausalLM") is BailingMoeV2ForCausalLM
