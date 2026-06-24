# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for Nemotron-V3 Multi-Token Prediction (MTP) wiring.

Covers the model-agnostic scaffolding (``roll_tensor``, pattern parsing,
loss aggregation) and the NemotronV3-specific glue (forward emits
``mtp_loss``, gradients flow into MTP params, state-dict carries the
expected ``mtp.*`` keys).
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.mtp import (
    MTPConfig,
    roll_tensor,
)
from nemo_automodel.components.models.nemotron_v3.mtp import (
    _resolve_block_types_per_sublayer,
    build_mtp_config_from_hf,
    parse_mtp_layer_pattern,
)


class MockNemotronV3Config:
    """Minimal NemotronV3-compatible config for CPU unit testing."""

    def __init__(self, **overrides):
        # Core dims
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 16
        self.hidden_size = 64
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP / MoE
        self.intermediate_size = 64
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"
        self.moe_intermediate_size = 32
        self.moe_shared_expert_intermediate_size = 32

        # Mamba (unused when no mamba layers, but referenced by config)
        self.mamba_num_heads = 2
        self.mamba_head_dim = 16
        self.ssm_state_size = 8
        self.n_groups = 1
        self.chunk_size = 64
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 2
        self.vocab_size = 64
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False
        self.layers_block_type = ["attention", "moe"]

        # MoE routing
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = False

        # MTP fields default to disabled; tests opt in via overrides.
        self.num_nextn_predict_layers = 0
        self.mtp_hybrid_override_pattern = ""

        for key, value in overrides.items():
            setattr(self, key, value)

    def to_dict(self):
        return vars(self)


# ---------------------------------------------------------------------------
# Pattern parsing & rolling
# ---------------------------------------------------------------------------


class TestPatternParsing:
    def test_parse_star_e(self):
        assert parse_mtp_layer_pattern("*E") == ["attention", "moe"]

    def test_parse_complex_pattern(self):
        assert parse_mtp_layer_pattern("M*-E") == ["mamba", "attention", "mlp", "moe"]

    def test_unknown_symbol_raises(self):
        with pytest.raises(ValueError, match="Unknown MTP layer symbol"):
            parse_mtp_layer_pattern("X")

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_mtp_layer_pattern("")

    def test_resolve_from_symbol_pattern(self):
        """Super-V3 path: ``mtp_hybrid_override_pattern`` is honored."""
        cfg = MockNemotronV3Config(num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*E")
        cfg.mtp_layers_block_type = None
        assert _resolve_block_types_per_sublayer(cfg) == ["attention", "moe"]

    def test_resolve_from_layers_block_type(self):
        """List-form path: ``mtp_layers_block_type`` list-of-strings is honored."""
        cfg = MockNemotronV3Config(num_nextn_predict_layers=1)
        cfg.mtp_hybrid_override_pattern = None
        cfg.mtp_layers_block_type = ["attention", "moe"]
        assert _resolve_block_types_per_sublayer(cfg) == ["attention", "moe"]

    def test_resolve_returns_none_when_both_absent(self):
        cfg = MockNemotronV3Config()
        cfg.mtp_hybrid_override_pattern = None
        cfg.mtp_layers_block_type = None
        assert _resolve_block_types_per_sublayer(cfg) is None

    def test_resolve_rejects_unknown_block_type_in_list(self):
        cfg = MockNemotronV3Config()
        cfg.mtp_hybrid_override_pattern = None
        cfg.mtp_layers_block_type = ["attention", "bogus"]
        with pytest.raises(ValueError, match="Unknown MTP block type"):
            _resolve_block_types_per_sublayer(cfg)

    def test_build_mtp_config_from_layers_block_type(self):
        """List-form config (no symbol pattern) yields an enabled MTPConfig.

        Regression for the ``mtp_layers_block_type`` fallback added so that
        checkpoints lacking ``mtp_hybrid_override_pattern`` can build MTP
        without raising ``MTP layer pattern is empty``.
        """
        cfg = MockNemotronV3Config(num_nextn_predict_layers=1)
        cfg.mtp_hybrid_override_pattern = None
        cfg.mtp_layers_block_type = ["attention", "moe"]
        mtp_config = build_mtp_config_from_hf(cfg)
        assert mtp_config.enabled
        assert mtp_config.num_layers == 1
        assert mtp_config.pattern_length == 2
        assert mtp_config.total_sublayers == 2


class TestRollTensor:
    def test_left_shift_zeros_trailing(self):
        t = torch.arange(8).unsqueeze(0)  # [1, 8]
        rolled = roll_tensor(t, shifts=-1, dim=-1)
        assert rolled.tolist() == [[1, 2, 3, 4, 5, 6, 7, 0]]

    def test_double_left_shift_zeros_two_trailing(self):
        t = torch.arange(8).unsqueeze(0)
        rolled1 = roll_tensor(t, shifts=-1, dim=-1)
        rolled2 = roll_tensor(rolled1, shifts=-1, dim=-1)
        assert rolled2.tolist() == [[2, 3, 4, 5, 6, 7, 0, 0]]

    def test_zero_shift_is_identity(self):
        t = torch.arange(8).unsqueeze(0)
        assert torch.equal(roll_tensor(t, shifts=0, dim=-1), t)

    def test_right_shift_zeros_leading(self):
        t = torch.arange(8).unsqueeze(0)
        rolled = roll_tensor(t, shifts=1, dim=-1)
        assert rolled.tolist() == [[0, 0, 1, 2, 3, 4, 5, 6]]


class TestMTPConfig:
    def test_disabled_when_zero_layers(self):
        c = MTPConfig(num_layers=0, layer_pattern="*E")
        assert not c.enabled

    def test_disabled_when_empty_pattern(self):
        c = MTPConfig(num_layers=1, layer_pattern="")
        assert not c.enabled

    def test_enabled_and_total_sublayers(self):
        c = MTPConfig(num_layers=2, layer_pattern="*E")
        assert c.enabled
        assert c.pattern_length == 2
        assert c.total_sublayers == 4

    def test_use_repeated_layer_defaults_false(self):
        """Regression: default behavior is unchanged when the new flag is unset."""
        c = MTPConfig(num_layers=2, layer_pattern="*E")
        assert c.use_repeated_layer is False
        assert c.num_physical_depths == c.num_layers
        assert c.total_sublayers == c.num_layers * c.pattern_length

    def test_use_repeated_layer_collapses_physical_depths(self):
        """When set, physical depth count is 1 regardless of num_layers."""
        c = MTPConfig(num_layers=5, layer_pattern="*E", use_repeated_layer=True)
        assert c.num_physical_depths == 1
        assert c.total_sublayers == 1 * c.pattern_length


# ---------------------------------------------------------------------------
# End-to-end forward / backward
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
    )


def _make_model(backend, *, mtp_layers=0, mtp_pattern="", **cfg_overrides):
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    config = MockNemotronV3Config(
        num_nextn_predict_layers=mtp_layers,
        mtp_hybrid_override_pattern=mtp_pattern,
        **cfg_overrides,
    )
    model = NemotronHForCausalLM(config, backend=backend)
    return model.to(torch.bfloat16), config


class TestMTPDisabled:
    @pytest.mark.run_only_on("GPU")
    def test_no_mtp_when_config_omits_fields(self, backend):
        """When num_nextn_predict_layers is 0, self.mtp must be None and
        forward must not emit per-depth hidden states."""
        model, config = _make_model(backend)
        assert model.mtp is None

        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert getattr(out, "mtp_per_depth_h", None) is None
        assert getattr(out, "mtp_loss_scaling_factor", None) is None
        assert out.loss is not None
        assert out.logits.shape == (2, 8, config.vocab_size)


class TestMTPEnabled:
    def test_module_built_with_correct_layer_count(self, backend):
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        assert model.mtp is not None
        # D=1, P=2 -> 2 flat sublayers; fusion on idx 0, final_layernorm on idx 1.
        assert len(model.mtp.layers) == 2
        assert model.mtp.layers[0].has_fusion is True
        assert model.mtp.layers[0].has_final_norm is False
        assert model.mtp.layers[1].has_fusion is False
        assert model.mtp.layers[1].has_final_norm is True
        assert model.mtp.layers[0].block_type == "attention"
        assert model.mtp.layers[1].block_type == "moe"

    def test_fusion_modules_present_on_first_sublayer_only(self, backend):
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        first = model.mtp.layers[0]
        last = model.mtp.layers[1]
        assert hasattr(first, "enorm") and hasattr(first, "hnorm") and hasattr(first, "eh_proj")
        assert not hasattr(last, "enorm")
        assert hasattr(last, "final_layernorm")
        assert not hasattr(first, "final_layernorm")

    @pytest.mark.run_only_on("GPU")
    def test_forward_train_emits_mtp_per_depth_h(self, backend):
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out.mtp_per_depth_h is not None
        assert isinstance(out.mtp_per_depth_h, list)
        # D=1 in this fixture
        assert len(out.mtp_per_depth_h) == 1
        # Each per-depth tensor must be on the autograd graph so the
        # recipe-side ``calculate_mtp_loss`` produces trainable gradients.
        assert all(h.requires_grad for h in out.mtp_per_depth_h)
        assert out.mtp_loss_scaling_factor == 0.1
        # Main loss is unaffected.
        assert out.loss is not None
        assert out.loss.requires_grad

    @pytest.mark.run_only_on("GPU")
    def test_forward_eval_skips_mtp_branch(self, backend):
        """At eval (or without labels) MTP must not run."""
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.mtp_per_depth_h is None
        assert out.mtp_loss_scaling_factor is None

    @pytest.mark.run_only_on("GPU")
    def test_mtp_backward_populates_mtp_grads(self, backend):
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        # Backprop through a sum of per-depth hidden states to isolate MTP grads
        # (the production path runs them through the configured loss class
        # via the recipe-side ``calculate_mtp_loss``).
        sum(h.sum() for h in out.mtp_per_depth_h).backward()
        assert model.mtp.layers[0].enorm.weight.grad is not None
        assert model.mtp.layers[0].hnorm.weight.grad is not None
        assert model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.mtp.layers[1].final_layernorm.weight.grad is not None


# ---------------------------------------------------------------------------
# State-dict adapter MTP key handling
# ---------------------------------------------------------------------------


class TestMTPRepeatedLayer:
    """Weight-tied repeated MTP iterations (Megatron --mtp-use-repeated-layer parity).

    Build with ``num_nextn_predict_layers > 1`` overridden via the kwarg and
    ``mtp_use_repeated_layer=True``. Only one physical depth's worth of
    sublayers should exist in the state_dict, while the forward emits one
    hidden state per iteration.
    """

    def _build_repeated(self, backend, *, iterations, pattern):
        """Construct a model whose HF config exposes a single physical MTP depth
        but whose runtime iterates ``iterations`` times through the shared head."""
        from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

        # HF config records 1 physical depth (this is what the checkpoint exposes);
        # the kwargs override drives the runtime iteration count and weight tying.
        config = MockNemotronV3Config(
            num_nextn_predict_layers=1,
            mtp_hybrid_override_pattern=pattern,
        )
        model = NemotronHForCausalLM(
            config,
            backend=backend,
            num_nextn_predict_layers=iterations,
            mtp_use_repeated_layer=True,
        )
        return model.to(torch.bfloat16), config

    def test_physical_sublayers_match_one_depth(self, backend):
        model, _ = self._build_repeated(backend, iterations=2, pattern="*E")
        # P=2; one physical depth backs all iterations.
        assert len(model.mtp.layers) == 2
        # Fusion on the first sublayer, final_layernorm on the last.
        assert model.mtp.layers[0].has_fusion is True
        assert model.mtp.layers[1].has_final_norm is True

    @pytest.mark.run_only_on("GPU")
    def test_forward_emits_one_h_per_iteration(self, backend):
        model, config = self._build_repeated(backend, iterations=2, pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.mtp_per_depth_h is not None
        # 2 iterations -> 2 hidden states returned, despite only 1 physical depth.
        assert len(out.mtp_per_depth_h) == 2
        assert all(h.requires_grad for h in out.mtp_per_depth_h)

    @pytest.mark.run_only_on("GPU")
    def test_backward_accumulates_on_shared_params(self, backend):
        """Calling the same physical sublayer twice must accumulate gradient
        from both iterations onto the shared parameters."""
        model, config = self._build_repeated(backend, iterations=2, pattern="*E")
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        # Each iteration's hidden state must influence the shared params.
        sum(h.sum() for h in out.mtp_per_depth_h).backward()
        # The single physical fusion-bearing sublayer must have a grad.
        assert model.mtp.layers[0].enorm.weight.grad is not None
        assert model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.mtp.layers[1].final_layernorm.weight.grad is not None

    def test_state_dict_layout_matches_one_depth(self, backend):
        """State dict must contain only the one physical depth's keys, so the
        HF checkpoint (which carries D=1 worth of mtp.* weights) loads cleanly."""
        model, _ = self._build_repeated(backend, iterations=3, pattern="*E")
        sd = model.state_dict()
        # Only mtp.layers.{0,1} should exist for P=2 with 1 physical depth.
        mtp_indices = {int(k.split(".")[2]) for k in sd if k.startswith("mtp.layers.")}
        assert mtp_indices == {0, 1}


class TestMTPLossDispatch:
    """``calculate_mtp_loss`` has two branches in ``train_ft.py``: one for
    ``FusedLinearCrossEntropy`` (passes ``hidden_states`` straight to the
    fused kernel) and one for standard loss classes (materializes logits via
    ``lm_head`` first). All other MTP tests + the verification YAMLs exercise
    the standard branch; this class covers the fused branch on CPU by
    monkey-patching ``cut_cross_entropy.linear_cross_entropy`` to a stub.
    """

    def test_fused_linear_ce_branch_dispatches(self, backend, monkeypatch):
        """When the loss is ``FusedLinearCrossEntropy``, ``calculate_mtp_loss``
        must route per-depth hidden states + lm_head weight into the fused
        kernel, NOT materialize logits via ``lm_head(h_k)``. We verify this
        by stubbing ``linear_cross_entropy`` and asserting it is called once
        per MTP depth with the expected kwargs."""
        from nemo_automodel.components.loss import linear_ce as linear_ce_mod
        from nemo_automodel.recipes.llm.train_ft import calculate_mtp_loss

        if not linear_ce_mod.HAVE_CUT_CROSS_ENTROPY:
            pytest.skip("cut_cross_entropy not installed")

        calls: list[dict] = []

        def fake_linear_cross_entropy(hidden_states, lm_weight, **kwargs):
            calls.append(
                {
                    "hidden_shape": tuple(hidden_states.shape),
                    "weight_shape": tuple(lm_weight.shape),
                    "labels_shape": tuple(kwargs["targets"].shape),
                    "ignore_index": kwargs.get("ignore_index"),
                    "reduction": kwargs.get("reduction"),
                }
            )
            return torch.tensor(1.0)

        monkeypatch.setattr(linear_ce_mod, "linear_cross_entropy", fake_linear_cross_entropy)

        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        # Two synthetic per-depth hidden states matching what model.mtp would
        # produce; shapes are [B, S, H].
        bsz, seq_len = 2, 8
        h = torch.randn(bsz, seq_len, config.hidden_size, dtype=torch.bfloat16)
        mtp_per_depth_h = [h.clone() for _ in range(model.mtp_config.num_layers)]
        labels = torch.randint(0, config.vocab_size, (bsz, seq_len))

        loss_fn = linear_ce_mod.FusedLinearCrossEntropy(reduction="sum")
        out = calculate_mtp_loss(
            loss_fn,
            mtp_per_depth_h=mtp_per_depth_h,
            labels=labels,
            model=model,
            scaling_factor=0.1,
            num_label_tokens=bsz * seq_len,
        )

        # One fused-kernel invocation per MTP depth (D=1 here).
        assert len(calls) == len(mtp_per_depth_h), (
            f"expected {len(mtp_per_depth_h)} fused-kernel calls, got {len(calls)}"
        )
        # Each call must receive the lm_head weight of shape [vocab, hidden].
        # FusedLinearCrossEntropy flattens hidden states and labels to a single
        # token axis ([N, H] / [N]) before the fused kernel so the loss is layout
        # agnostic (THD / context-parallel safe); here N == bsz * seq_len.
        for call in calls:
            assert call["weight_shape"] == (config.vocab_size, config.hidden_size)
            assert call["hidden_shape"] == (bsz * seq_len, config.hidden_size)
            assert call["labels_shape"] == (bsz * seq_len,)
            assert call["ignore_index"] == -100
        # Loss is finite scalar. The fused branch normalizes stub_loss by
        # num_label_tokens internally (linear_ce.py:169-171), then
        # calculate_mtp_loss scales by scaling_factor/D over the per-depth sum:
        #   per-depth = stub_loss / num_label_tokens = 1.0 / 16 = 0.0625
        #   total     = (0.1 / 1) * 0.0625 = 0.00625
        expected = 0.1 / len(mtp_per_depth_h) * (1.0 / (bsz * seq_len)) * len(mtp_per_depth_h)
        assert torch.isfinite(out).item()
        assert abs(out.item() - expected) < 1e-6, f"expected {expected}, got {out.item()}"


class TestMTPInputEmbeds:
    """Verify that NemotronHForCausalLM correctly uses pre-fused embeddings (inputs_embeds)
    as the MTP future-token signal instead of re-embedding from input_ids.

    This is the multimodal fix: in SALM, audio positions carry padding_id in mtp_input_ids
    but the correct audio embedding in inputs_embeds.  Without this path, embed_fn(padding_id)
    would be used at every audio position, injecting wrong future-token signal into MTP fusion.

    The fix lives in model.py: when inputs_embeds is provided, it is pre-rolled per depth
    and passed to MTPModule as embed_inputs, bypassing the embed_fn path entirely.
    """

    @pytest.mark.run_only_on("GPU")
    def test_embed_inputs_overrides_embed_fn(self, backend):
        """When embed_inputs is passed to MTPModule, the fusion sublayer receives it
        directly instead of calling embed_fn(rolled_input_ids)."""
        from nemo_automodel.components.models.common.mtp import MTPConfig, roll_tensor
        from nemo_automodel.components.models.nemotron_v3.mtp import build_nemotron_v3_mtp

        H = 64
        B, S = 2, 10
        config = MockNemotronV3Config(num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*")
        mtp_config = MTPConfig(num_layers=1, layer_pattern="*")
        mtp = build_nemotron_v3_mtp(
            config, mtp_config=mtp_config, backend=backend, moe_config=None, dtype=torch.bfloat16
        ).to(torch.bfloat16)

        hidden_states = torch.randn(B, S, H, dtype=torch.bfloat16)
        # Pre-fused embeddings (e.g. SALM audio+text): fill with a recognizable value.
        input_embeds = torch.full((B, S, H), 7.0, dtype=torch.bfloat16)
        # Pre-roll once for depth 0 (as model.py does before calling mtp).
        rolled = roll_tensor(input_embeds, shifts=-1, dim=-2)

        captured_embed_inputs = []
        original_first_sublayer_call = mtp.layers[0].forward

        def patched_first_sublayer(h, **kwargs):
            if "embed_input" in kwargs:
                captured_embed_inputs.append(kwargs["embed_input"].detach().clone())
            return original_first_sublayer_call(h, **kwargs)

        mtp.layers[0].forward = patched_first_sublayer

        # Pass via embed_inputs tuple (the path model.py takes for multimodal).
        mtp(hidden_states=hidden_states, embed_inputs=(rolled,))

        assert len(captured_embed_inputs) == 1
        assert torch.allclose(captured_embed_inputs[0], rolled), (
            "embed_input to fusion sublayer should equal the pre-rolled embed_inputs tensor"
        )

    @pytest.mark.run_only_on("GPU")
    def test_without_inputs_embeds_uses_embed_fn(self, backend):
        """Regression: when inputs_embeds is absent the original embed_fn path is unchanged."""
        from nemo_automodel.components.models.common.mtp import MTPConfig
        from nemo_automodel.components.models.nemotron_v3.mtp import build_nemotron_v3_mtp

        H = 64
        B, S = 2, 10
        config = MockNemotronV3Config(num_nextn_predict_layers=1, mtp_hybrid_override_pattern="*")
        mtp_config = MTPConfig(num_layers=1, layer_pattern="*")
        mtp = build_nemotron_v3_mtp(
            config, mtp_config=mtp_config, backend=backend, moe_config=None, dtype=torch.bfloat16
        ).to(torch.bfloat16)

        hidden_states = torch.randn(B, S, H, dtype=torch.bfloat16)
        input_ids = torch.randint(0, 64, (B, S))
        sentinel = torch.full((B, S, H), 5.0, dtype=torch.bfloat16)

        embed_fn_calls = []

        def embed_fn(ids):
            embed_fn_calls.append(ids.clone())
            return sentinel.clone()

        captured_embed_inputs = []
        original_first_sublayer_call = mtp.layers[0].forward

        def patched_first_sublayer(h, **kwargs):
            if "embed_input" in kwargs:
                captured_embed_inputs.append(kwargs["embed_input"].detach().clone())
            return original_first_sublayer_call(h, **kwargs)

        mtp.layers[0].forward = patched_first_sublayer

        mtp(hidden_states=hidden_states, input_ids=input_ids, embed_fn=embed_fn)

        assert len(embed_fn_calls) == 1, "embed_fn must be called when no embed_inputs provided"
        assert len(captured_embed_inputs) == 1
        assert torch.allclose(captured_embed_inputs[0], sentinel), (
            "embed_input must equal embed_fn output when embed_inputs is None"
        )

    @pytest.mark.run_only_on("GPU")
    def test_model_forward_passes_inputs_embeds_to_mtp(self, backend):
        """NemotronHForCausalLM.forward() must thread inputs_embeds into self.mtp() so
        that the per-depth hidden states differ from the inputs_embeds-absent case."""
        model, config = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()
        B, S = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        # Craft inputs_embeds that differ from what embed_tokens(input_ids) would give.
        inputs_embeds = torch.randn(B, S, config.hidden_size, dtype=torch.bfloat16)

        # Forward with input_ids only (embed_fn path).
        out_ids = model(input_ids=input_ids, labels=input_ids.clone())
        # Forward with inputs_embeds (rolled-embeds path), keeping input_ids for MTP token rolling.
        out_embs = model(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=input_ids.clone())

        assert out_embs.mtp_per_depth_h is not None
        # The two forwards used different future-token embeddings in MTP, so the
        # per-depth hidden states must differ.
        assert not torch.allclose(out_ids.mtp_per_depth_h[0], out_embs.mtp_per_depth_h[0]), (
            "mtp_per_depth_h should differ when inputs_embeds differ from embed_tokens(input_ids)"
        )


class TestMTPStateDictAdapter:
    def test_mtp_keys_in_state_dict(self, backend):
        """Internal model state_dict must contain the expected mtp.* keys
        (matches the HF flat layout used by Super V3)."""
        model, _ = _make_model(backend, mtp_layers=1, mtp_pattern="*E")
        sd = model.state_dict()
        # Fusion modules on first sublayer (layer index 0 of the depth).
        assert "mtp.layers.0.enorm.weight" in sd
        assert "mtp.layers.0.hnorm.weight" in sd
        assert "mtp.layers.0.eh_proj.weight" in sd
        # final_layernorm on last sublayer (layer index 1).
        assert "mtp.layers.1.final_layernorm.weight" in sd
        # Per-sublayer block norm.
        assert "mtp.layers.0.norm.weight" in sd
        assert "mtp.layers.1.norm.weight" in sd
