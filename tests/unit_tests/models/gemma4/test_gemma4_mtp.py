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

"""Unit tests for the Gemma4 Multi-Token Prediction (MTP) head.

Two layers of coverage:

* CPU-only structural tests (run anywhere) cover the four invariants that
  distinguish a working Gemma4 MTP retrofit from a broken one:
    1. The MTP-pruned text config disables per-layer-input / MoE /
       KV-sharing / sliding attention so the wrapped
       ``Gemma4TextDecoderLayer`` runs as a plain dense full-attention block.
    2. ``build_gemma4_mtp`` / ``MTPModule`` produce the expected number of
       sublayers, with fusion modules on the first sublayer of each depth
       and ``final_layernorm`` on the last sublayer of each depth.
    3. The sublayer's forward consumes rolled ``position_ids`` and produces
       an output with the same ``[B, S, H]`` shape as its input — verifying
       we did not accidentally re-introduce the ``per_layer_input`` need.
    4. ``apply_parameter_freezing(freeze_base_for_mtp=True)`` freezes every
       parameter outside the ``mtp.*`` namespace.
    5. ``Gemma4WithMTPCausalLMOutput`` declares ``mtp_per_depth_h`` and
       ``mtp_loss_scaling_factor`` as real dataclass fields so they survive
       FSDP2's mixed-precision output cast (PR #2161 commit 0b2889ab).

* GPU integration tests build a tiny ``Gemma4ForConditionalGeneration`` end
  to end on a single CUDA device and exercise: train/eval gating of the MTP
  forward path, the auxiliary CE loss + backward, and the
  ``freeze_base_for_mtp`` parameter-freezing recipe knob.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
gemma4_modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
gemma4_config_module = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")

from nemo_automodel.components.models.common.mtp import MTPConfig
from nemo_automodel.components.models.gemma4_moe.mtp import (
    Gemma4MTPSublayer,
    _make_mtp_text_config,
    build_gemma4_mtp,
    build_mtp_config_from_kwargs,
)
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing


def _make_text_config(**overrides):
    """Build a tiny Gemma4TextConfig that exercises the variant-specific knobs.

    Defaults reproduce a pared-down E4B-shaped config: ``hidden_size_per_layer_input``
    is set (so we can verify it gets disabled by ``_make_mtp_text_config``),
    KV-sharing is on, ``layer_types`` mixes full and sliding, and ``rope_parameters``
    has both branches.
    """
    Gemma4TextConfig = gemma4_config_module.Gemma4TextConfig
    defaults = dict(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=4,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        enable_moe_block=False,
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=16,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
        # E4B-style features that MTP must neutralize:
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. _make_mtp_text_config — pruning invariants
# ---------------------------------------------------------------------------
class TestMakeMTPTextConfig:
    def test_disables_per_layer_input(self):
        cfg = _make_text_config()
        assert cfg.hidden_size_per_layer_input == 8  # baseline assumption
        mtp_cfg = _make_mtp_text_config(cfg)
        assert mtp_cfg.hidden_size_per_layer_input is None

    def test_disables_moe_block(self):
        cfg = _make_text_config(enable_moe_block=True, num_experts=2, top_k_experts=1, moe_intermediate_size=32)
        mtp_cfg = _make_mtp_text_config(cfg)
        assert mtp_cfg.enable_moe_block is False

    def test_disables_kv_sharing(self):
        cfg = _make_text_config()
        assert cfg.num_kv_shared_layers == 2
        mtp_cfg = _make_mtp_text_config(cfg)
        assert mtp_cfg.num_kv_shared_layers == 0

    def test_forces_full_attention(self):
        cfg = _make_text_config()
        mtp_cfg = _make_mtp_text_config(cfg)
        assert mtp_cfg.layer_types == ["full_attention"]

    def test_prunes_rope_parameters_to_full_attention(self):
        cfg = _make_text_config()
        if not isinstance(getattr(cfg, "rope_parameters", None), dict):
            pytest.skip("Gemma4TextConfig in this transformers build does not expose rope_parameters as dict")
        assert "full_attention" in cfg.rope_parameters
        mtp_cfg = _make_mtp_text_config(cfg)
        assert set(mtp_cfg.rope_parameters.keys()) == {"full_attention"}

    def test_does_not_mutate_input_config(self):
        cfg = _make_text_config()
        original_per_layer = cfg.hidden_size_per_layer_input
        _ = _make_mtp_text_config(cfg)
        assert cfg.hidden_size_per_layer_input == original_per_layer  # shallow copy


# ---------------------------------------------------------------------------
# 2. build_mtp_config_from_kwargs — runtime knobs
# ---------------------------------------------------------------------------
class TestBuildMTPConfigFromKwargs:
    def test_disabled_by_default(self):
        cfg = build_mtp_config_from_kwargs()
        assert cfg.enabled is False
        assert cfg.num_layers == 0

    def test_enabled_with_num_layers(self):
        cfg = build_mtp_config_from_kwargs(mtp_num_layers=2, mtp_layer_pattern="*", mtp_loss_scaling_factor=0.25)
        assert cfg.enabled is True
        assert cfg.num_layers == 2
        assert cfg.layer_pattern == "*"
        assert cfg.loss_scaling_factor == 0.25

    def test_zero_layers_disables(self):
        cfg = build_mtp_config_from_kwargs(mtp_num_layers=0, mtp_layer_pattern="*")
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# 3. Gemma4MTPSublayer + build_gemma4_mtp — structural / forward invariants
# ---------------------------------------------------------------------------
class TestGemma4MTPSublayer:
    @pytest.fixture
    def text_config(self):
        return _make_text_config()

    def test_sublayer_owns_rotary_embedding(self, text_config):
        mtp_cfg = _make_mtp_text_config(text_config)
        sublayer = Gemma4MTPSublayer(mtp_cfg, has_fusion=True, has_final_norm=True, dtype=torch.float32)
        assert isinstance(sublayer.rotary_emb, gemma4_modeling.Gemma4TextRotaryEmbedding)
        assert sum(p.numel() for p in sublayer.rotary_emb.parameters()) == 0

    def test_fusion_modules_present_when_requested(self, text_config):
        mtp_cfg = _make_mtp_text_config(text_config)
        sublayer = Gemma4MTPSublayer(mtp_cfg, has_fusion=True, has_final_norm=False, dtype=torch.float32)
        assert hasattr(sublayer, "enorm") and hasattr(sublayer, "hnorm") and hasattr(sublayer, "eh_proj")
        assert sublayer.eh_proj.weight.shape == (mtp_cfg.hidden_size, 2 * mtp_cfg.hidden_size)
        assert not hasattr(sublayer, "final_layernorm")

    def test_final_layernorm_present_when_requested(self, text_config):
        mtp_cfg = _make_mtp_text_config(text_config)
        sublayer = Gemma4MTPSublayer(mtp_cfg, has_fusion=False, has_final_norm=True, dtype=torch.float32)
        assert hasattr(sublayer, "final_layernorm")
        assert not hasattr(sublayer, "enorm")

    def test_forward_shape_matches_input(self, text_config):
        torch.manual_seed(0)
        mtp_cfg = _make_mtp_text_config(text_config)
        sublayer = Gemma4MTPSublayer(mtp_cfg, has_fusion=True, has_final_norm=True, dtype=torch.float32)
        sublayer.eval().to(torch.float32)
        # cast hidden states (created by Gemma4TextDecoderLayer in fp32 by default
        # for tests but we wire the LayerNorm dtypes through .to())
        sublayer = sublayer.to(torch.float32)

        B, S, H = 2, 4, mtp_cfg.hidden_size
        hidden_states = torch.randn(B, S, H, dtype=torch.float32)
        embed_input = torch.randn(B, S, H, dtype=torch.float32)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, S)

        out = sublayer(
            hidden_states,
            embed_input=embed_input,
            position_ids=position_ids,
            attention_mask=None,
        )
        assert out.shape == hidden_states.shape


class TestBuildGemma4MTP:
    def test_layer_count_matches_pattern(self):
        text_cfg = _make_text_config()
        mtp_cfg = MTPConfig(num_layers=2, layer_pattern="*", loss_scaling_factor=0.1)
        mtp = build_gemma4_mtp(text_cfg, mtp_cfg, dtype=torch.float32)
        assert len(mtp.layers) == mtp_cfg.total_sublayers == 2

    def test_first_sublayer_per_depth_has_fusion(self):
        text_cfg = _make_text_config()
        mtp_cfg = MTPConfig(num_layers=2, layer_pattern="*", loss_scaling_factor=0.1)
        mtp = build_gemma4_mtp(text_cfg, mtp_cfg, dtype=torch.float32)
        # pattern_length=1 → every sublayer is both first and last in its depth
        for sublayer in mtp.layers:
            assert sublayer.has_fusion
            assert sublayer.has_final_norm

    def test_unsupported_block_type_raises(self):
        text_cfg = _make_text_config()
        # "E" maps to "moe" which Gemma4 MTP does not implement.
        bad_cfg = MTPConfig(num_layers=1, layer_pattern="E", loss_scaling_factor=0.1)
        with pytest.raises(NotImplementedError):
            build_gemma4_mtp(text_cfg, bad_cfg, dtype=torch.float32)

    def test_disabled_config_raises(self):
        text_cfg = _make_text_config()
        with pytest.raises(ValueError):
            build_gemma4_mtp(text_cfg, MTPConfig(), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. apply_parameter_freezing(freeze_base_for_mtp=True)
# ---------------------------------------------------------------------------
class _MockModelWithMTP(nn.Module):
    """Mimics the parameter naming of Gemma4ForConditionalGeneration with MTP."""

    def __init__(self, dim: int = 8, num_mtp_sublayers: int = 1):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Linear(dim, dim, bias=False)
        self.model.language_model.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(2)])
        self.lm_head = nn.Linear(dim, dim, bias=False)
        self.mtp = nn.Module()
        self.mtp.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_mtp_sublayers)])

    @property
    def config(self):
        from types import SimpleNamespace

        return SimpleNamespace(model_type="gemma4")


class TestFreezeBaseForMTP:
    def test_only_mtp_params_remain_trainable(self):
        model = _MockModelWithMTP(num_mtp_sublayers=2)
        apply_parameter_freezing(
            model,
            freeze_config={
                "freeze_vision_tower": False,
                "freeze_audio_tower": False,
                "freeze_language_model": False,
                "freeze_base_for_mtp": True,
            },
        )
        for name, param in model.named_parameters():
            if name.startswith("mtp.") or ".mtp." in name:
                assert param.requires_grad, f"MTP param {name!r} should be trainable"
            else:
                assert not param.requires_grad, f"Non-MTP param {name!r} should be frozen"

    def test_freeze_flag_off_leaves_params_trainable(self):
        model = _MockModelWithMTP()
        apply_parameter_freezing(
            model,
            freeze_config={
                "freeze_vision_tower": False,
                "freeze_audio_tower": False,
                "freeze_language_model": False,
                "freeze_base_for_mtp": False,
            },
        )
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name!r} should remain trainable when freeze_base_for_mtp=False"


# ---------------------------------------------------------------------------
# 5. Gemma4WithMTPCausalLMOutput dataclass-fields invariant (CPU-only)
#
# FSDP2's mixed-precision output cast rebuilds ModelOutput instances from
# DECLARED dataclass fields only. Auxiliary fields tacked on with setattr()
# get silently dropped on the way out. This is the regression that caused
# NemotronV3 MTP losses to become NaN (PR #2161, commit 0b2889ab) — the
# `mtp_per_depth_h` list disappeared after FSDP2 reconstructed the output
# and the recipe's MTP loss saw an empty list. We replicate the same
# guarantee here.
# ---------------------------------------------------------------------------
class TestMTPOutputDataclass:
    def test_mtp_output_declares_required_fields(self):
        import dataclasses

        from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

        field_names = {f.name for f in dataclasses.fields(Gemma4WithMTPCausalLMOutput)}
        assert {"logits", "mtp_per_depth_h", "mtp_loss_scaling_factor"}.issubset(field_names), (
            "Gemma4WithMTPCausalLMOutput must declare mtp_per_depth_h and "
            "mtp_loss_scaling_factor as dataclass fields so FSDP2's "
            "mixed-precision output cast preserves them."
        )

    def test_mtp_output_roundtrip_through_dataclass_constructor(self):
        import dataclasses

        from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

        out = Gemma4WithMTPCausalLMOutput(
            logits=torch.zeros(1, 2, 3),
            hidden_states=None,
            mtp_per_depth_h=[torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)],
            mtp_loss_scaling_factor=0.25,
        )
        rebuilt = type(out)(**{f.name: getattr(out, f.name) for f in dataclasses.fields(out)})
        assert isinstance(rebuilt.mtp_per_depth_h, list)
        assert len(rebuilt.mtp_per_depth_h) == 2
        assert rebuilt.mtp_loss_scaling_factor == 0.25


# ---------------------------------------------------------------------------
# 6. End-to-end GPU integration tests
#
# These exercise MTP through ``Gemma4ForConditionalGeneration`` with a real
# tiny dense Gemma4 backbone. They use the same skip-on-CPU pattern as
# ``test_gemma4_model.py`` so the file as a whole still runs on CPU; only
# this class is gated behind CUDA.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGemma4MTPIntegration:
    """End-to-end tests on a tiny dense Gemma4 with MTP enabled.

    Uses the dense path (``enable_moe_block=False``) because that's what the
    Gemma4 4B variant the user is targeting actually runs and because the
    dense path's ``initialize_weights`` does not require a CUDA-bound MoE
    expert init pass — keeping the test fast.
    """

    @staticmethod
    def _build_dense_text_config():
        Gemma4TextConfig = gemma4_config_module.Gemma4TextConfig
        return Gemma4TextConfig(
            vocab_size=64,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=2,
            intermediate_size=64,
            rms_norm_eps=1e-6,
            max_position_embeddings=64,
            enable_moe_block=False,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=16,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="bfloat16",
        )

    @staticmethod
    def _build_model(*, mtp_num_layers: int = 0, scaling_factor: float = 0.1):
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.gemma4_moe.model import (
            _GEMMA4_HF_AVAILABLE,
            Gemma4ForConditionalGeneration,
        )

        if not _GEMMA4_HF_AVAILABLE:
            pytest.skip("transformers.models.gemma4 not available")

        text_cfg = TestGemma4MTPIntegration._build_dense_text_config()
        cfg = gemma4_config_module.Gemma4Config(text_config=text_cfg)
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        model = Gemma4ForConditionalGeneration.from_config(
            cfg,
            backend=backend,
            mtp_num_layers=mtp_num_layers,
            mtp_layer_pattern="*",
            mtp_loss_scaling_factor=scaling_factor,
        )
        device = torch.device("cuda")
        model.initialize_weights(buffer_device=device, dtype=torch.float32)
        return model.to(device), text_cfg

    def test_mtp_disabled_constructs_no_mtp_attribute(self):
        model, _ = self._build_model(mtp_num_layers=0)
        assert model.mtp is None
        assert model.mtp_config.enabled is False

    def test_mtp_enabled_constructs_expected_sublayer_count(self):
        D = 2
        model, _ = self._build_model(mtp_num_layers=D)
        assert model.mtp is not None
        assert len(model.mtp.layers) == D
        # Every sublayer should carry both fusion + final_layernorm because
        # pattern "*" has length 1 (each sublayer is both first and last).
        for sl in model.mtp.layers:
            assert sl.has_fusion and sl.has_final_norm

    def test_mtp_disabled_forward_returns_plain_output(self):
        model, text_cfg = self._build_model(mtp_num_layers=0)
        model.train()
        device = next(model.parameters()).device
        B, S = 2, 6
        input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)
        out = model(input_ids=input_ids)
        # When MTP is off the dense path returns the unmodified HF output;
        # importantly, it is NOT a Gemma4WithMTPCausalLMOutput.
        from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

        assert not isinstance(out, Gemma4WithMTPCausalLMOutput)

    def test_train_mode_returns_mtp_output_with_correct_shapes(self):
        D = 2
        model, text_cfg = self._build_model(mtp_num_layers=D, scaling_factor=0.3)
        model.train()
        device = next(model.parameters()).device
        B, S = 2, 6
        input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)

        out = model(input_ids=input_ids)

        from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

        assert isinstance(out, Gemma4WithMTPCausalLMOutput)
        assert out.logits.shape == (B, S, text_cfg.vocab_size)
        assert isinstance(out.mtp_per_depth_h, list)
        assert len(out.mtp_per_depth_h) == D
        for h in out.mtp_per_depth_h:
            assert h.shape == (B, S, text_cfg.hidden_size)
        # Scaling factor must be carried verbatim from the MTPConfig.
        assert out.mtp_loss_scaling_factor == 0.3

    def test_eval_mode_skips_mtp_path(self):
        D = 2
        model, text_cfg = self._build_model(mtp_num_layers=D)
        model.eval()
        device = next(model.parameters()).device
        B, S = 1, 4
        input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        from nemo_automodel.components.models.gemma4_moe.model import Gemma4WithMTPCausalLMOutput

        assert not isinstance(out, Gemma4WithMTPCausalLMOutput)

    def test_backward_produces_gradients_on_mtp_params(self):
        D = 2
        model, text_cfg = self._build_model(mtp_num_layers=D, scaling_factor=0.5)
        model.train()
        device = next(model.parameters()).device
        B, S = 2, 6
        input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)
        labels = input_ids.clone()

        out = model(input_ids=input_ids)
        # Use the same loss the recipe would (MaskedCrossEntropy) to keep
        # the autograd graph identical to production.
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
        from nemo_automodel.recipes.vlm.finetune import calculate_mtp_loss

        loss_fn = MaskedCrossEntropy()
        num_tokens = int((labels != -100).sum().item())

        # Main next-token CE on the materialized logits.
        logits = out.logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        main_loss = loss_fn(logits=logits, labels=shifted_labels, num_label_tokens=num_tokens)

        mtp_loss = calculate_mtp_loss(
            loss_fn,
            mtp_per_depth_h=out.mtp_per_depth_h,
            labels=labels,
            model=model,
            scaling_factor=out.mtp_loss_scaling_factor,
            num_label_tokens=num_tokens,
        )
        total = main_loss + mtp_loss
        assert torch.isfinite(total)
        total.backward()

        mtp_grad_seen = False
        for name, p in model.named_parameters():
            if name.startswith("mtp.") or ".mtp." in name:
                if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
                    mtp_grad_seen = True
                    break
        assert mtp_grad_seen, "Expected at least one MTP parameter to have non-zero gradients after backward()"

    def test_freeze_base_for_mtp_blocks_backbone_grads(self):
        D = 1
        model, text_cfg = self._build_model(mtp_num_layers=D, scaling_factor=0.5)
        model.train()

        # Apply the recipe's parameter-freezing knob the same way the recipe
        # does it. We only test the freeze_base_for_mtp flag here; vision /
        # audio / language_model freezing has its own tests.
        apply_parameter_freezing(
            model,
            freeze_config={
                "freeze_vision_tower": False,
                "freeze_audio_tower": False,
                "freeze_language_model": False,
                "freeze_base_for_mtp": True,
            },
        )

        device = next(model.parameters()).device
        B, S = 1, 5
        input_ids = torch.randint(0, text_cfg.vocab_size, (B, S), device=device)
        labels = input_ids.clone()

        out = model(input_ids=input_ids)
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
        from nemo_automodel.recipes.vlm.finetune import calculate_mtp_loss

        loss_fn = MaskedCrossEntropy()
        num_tokens = int((labels != -100).sum().item())

        # Build the loss chain: when the backbone is frozen but lm_head is
        # also frozen, the main CE has no trainable inputs. We rely on the
        # MTP loss path (which only needs mtp params to be trainable, and
        # uses lm_head's frozen weights as a fixed projection) to drive
        # gradient flow.
        mtp_loss = calculate_mtp_loss(
            loss_fn,
            mtp_per_depth_h=out.mtp_per_depth_h,
            labels=labels,
            model=model,
            scaling_factor=out.mtp_loss_scaling_factor,
            num_label_tokens=num_tokens,
        )
        assert torch.isfinite(mtp_loss)
        mtp_loss.backward()

        for name, p in model.named_parameters():
            if name.startswith("mtp.") or ".mtp." in name:
                assert p.requires_grad, f"MTP param {name!r} must remain trainable under freeze_base_for_mtp"
            else:
                assert not p.requires_grad, f"Non-MTP param {name!r} must be frozen under freeze_base_for_mtp"
                # Safety: a frozen param cannot accumulate gradient.
                assert p.grad is None or float(p.grad.abs().sum().item()) == 0.0
