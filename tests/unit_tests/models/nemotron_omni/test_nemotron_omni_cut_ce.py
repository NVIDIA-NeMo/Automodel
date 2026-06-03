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

"""Memory-efficient fused cross-entropy (cut-CE) support for ``NemotronOmni``.

The fused-linear-cross-entropy path in the FT recipe only fires when:
  (a) ``_supports_logits_to_keep(model)`` is True — i.e. the top-level
      ``*ForConditionalGeneration.forward`` has a ``logits_to_keep`` parameter;
  (b) ``model(logits_to_keep=1, output_hidden_states=True, **batch)`` returns an
      output where ``"hidden_states" in out`` (or ``out.hidden_states is not
      None``), the hidden states span the FULL sequence, and the logits cover
      only the last token.

Otherwise the recipe SILENTLY falls back to MaskedCrossEntropy.

``NemotronOmniForConditionalGeneration.forward`` is a thin VLM wrapper that
delegates the LM body (lm_head gating + final hidden states) to its inner
``NemotronHForCausalLM`` language model. These tests build a *real* tiny
NemotronH LLM on CPU (attention+mlp layers only, no Mamba/MoE kernels) and wire
it into a stub VLM via ``object.__new__`` — exercising the wrapper forward's
``logits_to_keep`` / ``output_hidden_states`` plumbing end-to-end without
loading the 30B vision/audio towers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
)
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

IMG_TOKEN_ID = 18
SOUND_TOKEN_ID = 27
VIDEO_TOKEN_ID = 131081


class MockNemotronV3Config:
    """Tiny NemotronV3 LLM config (attention+mlp only) runnable on CPU.

    Mirrors ``tests/unit_tests/models/nemotron_v3/test_nemotron_v3_model.py`` so
    the inner ``NemotronHForCausalLM`` builds without Mamba/MoE GPU kernels.
    """

    def __init__(self, **overrides):
        # Attention configuration
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 256
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP/MoE configuration
        self.intermediate_size = 512
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"

        # Mamba configuration (unused with attention+mlp layers, but required attrs)
        self.mamba_num_heads = 4
        self.mamba_head_dim = 32
        self.ssm_state_size = 16
        self.n_groups = 1
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General configuration
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 2
        self.vocab_size = 100
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Hybrid architecture: attention + mlp only (CPU-friendly).
        self.layers_block_type = ["attention", "mlp"]

        # MoE configuration (unused with the above block types).
        self.n_routed_experts = 4
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 128
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 128

        for key, value in overrides.items():
            setattr(self, key, value)

    def to_dict(self):
        return vars(self)


class _OmniConfig:
    """Minimal stand-in for the NemotronOmni config: forward only reads
    ``self.config.llm_config`` (to resolve ``output_hidden_states``)."""

    def __init__(self, llm_config):
        self.llm_config = llm_config


def _build_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build_omni(llm_config) -> NemotronOmniForConditionalGeneration:
    """Build a NemotronOmni wrapper with a real tiny inner LLM and only the
    attributes the text-only forward path touches.

    Vision/video/sound towers are NOT created — for a text-only batch every
    multimodal block is gated off (pixel_values / pixel_values_videos /
    sound_features are all None), so the forward goes straight to the LLM.
    """
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    inner = NemotronHForCausalLM(llm_config, backend=_build_backend()).to(torch.bfloat16)

    self = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(self)
    self.img_context_token_id = IMG_TOKEN_ID
    self.video_context_token_id = VIDEO_TOKEN_ID
    self.sound_context_token_id = SOUND_TOKEN_ID
    self.sound_encoder = None
    self.language_model = inner
    self.config = _OmniConfig(llm_config)
    return self


# -----------------------------------------------------------------------------
# (a) _supports_logits_to_keep
# -----------------------------------------------------------------------------


def test_forward_signature_supports_logits_to_keep():
    """The FT recipe gates cut-CE on a ``logits_to_keep`` forward parameter."""
    model = _build_omni(MockNemotronV3Config())
    assert _supports_logits_to_keep(model) is True


# -----------------------------------------------------------------------------
# (b) logits_to_keep=1 + output_hidden_states=True
# -----------------------------------------------------------------------------


def test_cut_ce_contract_hidden_states_and_last_token_logits():
    """With ``logits_to_keep=1, output_hidden_states=True`` the output must carry
    FULL-sequence hidden states and last-token-only logits."""
    config = MockNemotronV3Config()
    model = _build_omni(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    out = model(input_ids=input_ids, logits_to_keep=1, output_hidden_states=True)

    # Contract checked by the recipe (train_ft.py ~1444): hidden states present.
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # Final hidden states span the FULL sequence (lm_head input, pre-slice).
    hs = out.hidden_states
    hs = hs[0] if isinstance(hs, (tuple, list)) else hs
    assert hs.shape == (batch_size, seq_len, config.hidden_size)

    # logits_to_keep=1 => logits cover only the last token.
    assert out.logits.shape == (batch_size, 1, config.vocab_size)


def test_cut_ce_via_config_output_hidden_states():
    """``output_hidden_states`` resolves from the text sub-config when the
    forward arg is omitted (the recipe toggles it on ``llm_config``)."""
    config = MockNemotronV3Config(output_hidden_states=True)
    model = _build_omni(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = model(input_ids=input_ids, logits_to_keep=1)

    assert ("hidden_states" in out) or (out.hidden_states is not None)
    hs = out.hidden_states
    hs = hs[0] if isinstance(hs, (tuple, list)) else hs
    assert hs.shape == (2, 8, config.hidden_size)


# -----------------------------------------------------------------------------
# (c) default behavior preserved
# -----------------------------------------------------------------------------


def test_default_forward_yields_full_length_logits():
    """Default ``model(input_ids)`` (logits_to_keep=0, no output_hidden_states)
    must still produce full-length logits and not leak hidden states."""
    config = MockNemotronV3Config()
    model = _build_omni(config)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    out = model(input_ids=input_ids)

    assert out.logits.shape == (batch_size, seq_len, config.vocab_size)
    # output_hidden_states defaults to llm_config's value (False here).
    assert out.hidden_states is None


def test_default_logits_match_logits_to_keep_zero():
    """``logits_to_keep=0`` is the implicit default and must equal full
    projection (no slicing) bit-for-bit."""
    config = MockNemotronV3Config()
    model = _build_omni(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        default_logits = model(input_ids=input_ids).logits
        explicit_zero = model(input_ids=input_ids, logits_to_keep=0).logits

    assert default_logits.shape == explicit_zero.shape
    assert torch.equal(default_logits, explicit_zero)

    # And the last-token slice equals logits_to_keep=1.
    with torch.no_grad():
        last_token = model(input_ids=input_ids, logits_to_keep=1).logits
    torch.testing.assert_close(last_token, default_logits[:, -1:, :])


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
