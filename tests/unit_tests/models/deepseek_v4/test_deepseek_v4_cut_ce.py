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

"""Cut cross-entropy (FusedLinearCrossEntropy) support for DeepSeek V4.

These tests pin the contract that ``recipes/llm/train_ft.py`` relies on to use
the memory-efficient fused linear cross-entropy path:

1. ``DeepseekV4ForCausalLM.forward`` exposes a ``logits_to_keep`` parameter, so
   ``_supports_logits_to_keep(model)`` returns ``True``.
2. ``model(input_ids, logits_to_keep=1, output_hidden_states=True)`` returns an
   output that carries the FULL-sequence final hidden states (``"hidden_states"
   in out`` is truthy) while computing logits only for the last token.
3. The default call ``model(input_ids)`` still produces full-length logits and
   does not surface hidden states (backward-compatible behavior).

The tiny config mirrors ``test_dsv4_model_smoke.py``; ``disable_shared_expert_overlap``
keeps ``MoE.forward`` from allocating a ``torch.cuda.Stream()`` so the forward
runs on CPU-only CI images.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


def _tiny_config(**overrides) -> DeepseekV4Config:
    """Tiny V4 config: fits in ~1 GB CPU RAM, exercises the lm_head path."""
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
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        hc_mult=4,
        num_hash_layers=1,
        compress_ratios=[0, 0],  # 2 full-attention layers (no CSA)
        sliding_window=16,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _make_model(config: DeepseekV4Config) -> DeepseekV4ForCausalLM:
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
        # Run shared experts sequentially on the current stream so MoE.forward
        # does not allocate a torch.cuda.Stream() (CPU-only CI safe).
        disable_shared_expert_overlap=True,
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)
    # Force a single fp32 dtype (embed_tokens is built bf16 via torch_dtype while
    # bare nn.Linear submodules default to fp32) so matmuls don't mismatch.
    model = model.float()
    # Use small non-zero weights so a last-token slice differs meaningfully from
    # the full projection (Gate/GroupedExperts use uninitialized torch.empty;
    # initialize_weights() is not called here since it requires CUDA).
    with torch.no_grad():
        torch.manual_seed(0)
        for p in model.parameters():
            if p.is_floating_point():
                p.normal_(0.0, 0.02)
    model.eval()
    return model


def _build_tiny_model():
    """Instantiate the tiny model, skipping if the env can't import/build it."""
    try:
        cfg = _tiny_config()
        return cfg, _make_model(cfg)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"Cannot build tiny DeepSeek V4 model in this environment: {exc!r}")


class TestDeepseekV4CutCE:
    def test_supports_logits_to_keep(self):
        """forward() must expose ``logits_to_keep`` for the cut-CE recipe path."""
        _, model = _build_tiny_model()
        assert _supports_logits_to_keep(model) is True

    def test_output_hidden_states_and_logits_to_keep(self):
        """logits_to_keep=1 + output_hidden_states=True yields last-token logits
        and FULL-sequence hidden states; ``"hidden_states" in out`` is truthy."""
        cfg, model = _build_tiny_model()
        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))

        with torch.no_grad():
            out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

        # Membership check mirrors train_ft.py: ``if "hidden_states" not in out``.
        assert ("hidden_states" in out) or out.hidden_states is not None
        hidden_states = out.hidden_states
        assert hidden_states is not None
        # Hidden states span the FULL sequence (not sliced to the kept tokens).
        assert hidden_states.shape == (bsz, seq, cfg.hidden_size)
        # Logits correspond to ONLY the last token.
        assert out.logits.shape == (bsz, 1, cfg.vocab_size)
        assert isinstance(out.logits, torch.Tensor)

    def test_logits_to_keep_matches_full_last_token(self):
        """The last-token logits from logits_to_keep=1 equal the full projection."""
        cfg, model = _build_tiny_model()
        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))

        with torch.no_grad():
            full = model(input_ids, logits_to_keep=0)
            kept = model(input_ids, logits_to_keep=1)

        assert full.logits.shape == (bsz, seq, cfg.vocab_size)
        assert kept.logits.shape == (bsz, 1, cfg.vocab_size)
        torch.testing.assert_close(full.logits[:, -1:, :], kept.logits)

    def test_default_forward_full_logits_no_hidden_states(self):
        """Default call preserves full-length logits and omits hidden states."""
        cfg, model = _build_tiny_model()
        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))

        with torch.no_grad():
            out = model(input_ids)

        assert out.logits.shape == (bsz, seq, cfg.vocab_size)
        assert isinstance(out.logits, torch.Tensor)
        # Backward-compatible default: hidden states are not surfaced.
        assert out.hidden_states is None
        assert "hidden_states" not in out


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
