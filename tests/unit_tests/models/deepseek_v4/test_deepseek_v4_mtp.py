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

"""Unit tests for DeepSeek V4 Multi-Token Prediction (MTP) support.

All tests run on CPU with a tiny model config.

Run:
    PYTHONPATH=/path/to/Automodel_lao python -m pytest \
        tests/unit_tests/models/deepseek_v4/test_deepseek_v4_mtp.py -v -s
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
from nemo_automodel.components.models.deepseek_v4.mtp import build_mtp_config_from_hf

# MoE.forward unconditionally creates a torch.cuda.Stream() for shared experts.
# Gate the tests that actually call model.forward() on CUDA availability.
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MoE.forward unconditionally allocates torch.cuda.Stream() for shared experts",
)


def _tiny_config(**overrides) -> DeepseekV4Config:
    """Tiny V4 config for MTP tests: small enough to run fast on CPU."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        hc_mult=4,
        num_hash_layers=0,
        compress_ratios=[0, 0],
        sliding_window=16,
        num_nextn_predict_layers=0,  # disabled by default
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _make_model(config: DeepseekV4Config) -> DeepseekV4ForCausalLM:
    """Build a tiny model with no HF state dict adapter."""
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)
    model = model.float()
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.zero_()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTPConfig:
    def test_mtp_config_disabled(self):
        """num_nextn_predict_layers=0 -> mtp_config.enabled == False."""
        cfg = _tiny_config(num_nextn_predict_layers=0)
        mtp_config = build_mtp_config_from_hf(cfg)
        assert not mtp_config.enabled
        assert mtp_config.num_layers == 0
        assert mtp_config.layer_pattern == ""

    def test_mtp_config_enabled(self):
        """num_nextn_predict_layers=1 -> mtp_config.enabled == True, pattern == '*E'."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        mtp_config = build_mtp_config_from_hf(cfg)
        assert mtp_config.enabled
        assert mtp_config.num_layers == 1
        assert mtp_config.layer_pattern == "*E"
        assert mtp_config.pattern_length == 2  # one attn + one MoE sublayer


class TestModelMTPConstruction:
    def test_model_has_no_mtp_by_default(self):
        """Default config (num_nextn_predict_layers=0) -> model.mtp is None."""
        cfg = _tiny_config(num_nextn_predict_layers=0)
        model = _make_model(cfg)
        assert model.mtp is None

    def test_model_has_mtp_when_configured(self):
        """With num_nextn_predict_layers=1 -> model.mtp is not None.

        Pattern '*E' has 2 sublayers per depth, 1 depth -> 2 total sublayers.
        """
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        assert model.mtp is not None
        assert len(model.mtp.layers) == 2  # pattern_length=2, num_layers=1


class TestMTPForward:
    @_REQUIRES_CUDA
    def test_forward_eval_no_mtp_output(self):
        """In eval mode, mtp_per_depth_h should be None."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.eval()

        bsz, seq = 1, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        with torch.no_grad():
            out = model(input_ids)

        assert out.mtp_per_depth_h is None
        assert isinstance(out.logits, torch.Tensor)
        assert out.logits.shape == (bsz, seq, cfg.vocab_size)

    @_REQUIRES_CUDA
    def test_forward_train_mtp_output(self):
        """In train mode, mtp_per_depth_h is list of length 1, each [B, S, hidden]."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        out = model(input_ids)

        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1  # 1 MTP depth
        h = out.mtp_per_depth_h[0]
        assert h.shape == (bsz, seq, cfg.hidden_size), f"unexpected shape: {h.shape}"

    @_REQUIRES_CUDA
    def test_mtp_gradient_backprop(self):
        """MTP hidden states are differentiable; eh_proj gradient is non-None."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()

        bsz, seq = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        out = model(input_ids)

        # Backward through MTP head only.
        out.mtp_per_depth_h[0].sum().backward()

        # The first sublayer (sublayer 0) owns eh_proj (fusion layer).
        eh_proj_weight = model.mtp.layers[0].eh_proj.weight
        assert eh_proj_weight.grad is not None, "eh_proj.weight.grad is None after backward"

    @_REQUIRES_CUDA
    def test_logits_is_tensor(self):
        """out.logits must be a raw torch.Tensor, not wrapped."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(input_ids)

        assert isinstance(out.logits, torch.Tensor), f"expected Tensor, got {type(out.logits)}"


class TestMTPStateDict:
    def test_state_dict_has_mtp_keys(self):
        """mtp.layers.0.input_layernorm.weight should be in model.state_dict()."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        sd = model.state_dict()
        assert "mtp.layers.0.input_layernorm.weight" in sd, (
            f"MTP key not found; state_dict keys with 'mtp': {[k for k in sd if 'mtp' in k]}"
        )

    def test_rotary_not_in_state_dict(self):
        """_rotary_emb stored on sublayer must NOT appear in state_dict."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        # The sublayer stores rotary refs via object.__setattr__ to avoid registration.
        sublayer = model.mtp.layers[0]
        param_names = dict(sublayer.named_parameters())
        assert "_rotary_emb" not in param_names, "_rotary_emb should not be a registered parameter"
        assert "_rotary_emb_compress" not in param_names, "_rotary_emb_compress should not be a registered parameter"

    def test_rotary_not_in_model_state_dict(self):
        """MTP rotary refs should not pollute the top-level model state dict."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        sd = model.state_dict()
        mtp_rotary_keys = [k for k in sd if "mtp" in k and "rotary" in k]
        assert not mtp_rotary_keys, f"Unexpected MTP rotary keys in state_dict: {mtp_rotary_keys}"


if __name__ == "__main__":
    import sys

    suite = [
        ("MTPConfig disabled", TestMTPConfig().test_mtp_config_disabled),
        ("MTPConfig enabled", TestMTPConfig().test_mtp_config_enabled),
        ("Model no MTP by default", TestModelMTPConstruction().test_model_has_no_mtp_by_default),
        ("Model has MTP when configured", TestModelMTPConstruction().test_model_has_mtp_when_configured),
        ("State dict has MTP keys", TestMTPStateDict().test_state_dict_has_mtp_keys),
        ("Rotary not in state dict", TestMTPStateDict().test_rotary_not_in_state_dict),
        ("Rotary not in model state dict", TestMTPStateDict().test_rotary_not_in_model_state_dict),
    ]

    if torch.cuda.is_available():
        fwd = TestMTPForward()
        suite += [
            ("Forward eval no MTP output", fwd.test_forward_eval_no_mtp_output),
            ("Forward train MTP output", fwd.test_forward_train_mtp_output),
            ("MTP gradient backprop", fwd.test_mtp_gradient_backprop),
            ("Logits is tensor", fwd.test_logits_is_tensor),
        ]

    failed = []
    for name, fn in suite:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(suite)} tests")
        sys.exit(1)
    else:
        print(f"All {len(suite)} tests passed.")
