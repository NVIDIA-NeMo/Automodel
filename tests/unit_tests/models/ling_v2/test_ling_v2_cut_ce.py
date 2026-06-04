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

"""Memory-efficient fused cross-entropy (cut-CE / FusedLinearCrossEntropy) support.

The training recipe (``nemo_automodel/recipes/llm/train_ft.py``) only routes through
``FusedLinearCrossEntropy`` when (a) ``_supports_logits_to_keep(model)`` is True (the
forward exposes a ``logits_to_keep`` parameter) AND (b) ``model(logits_to_keep=1, ...)``
returns an output where ``"hidden_states" in out`` and ``get_final_hidden_states(out)``
yields the final (full-sequence) hidden states. Otherwise it silently falls back to
``MaskedCrossEntropy``. These tests pin that contract for ``BailingMoeV2ForCausalLM``.

The tiny ``torch``-backend MoE model runs on CPU, so no GPU is required.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.models.ling_v2.model import BailingMoeV2ForCausalLM
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


def _tiny_cfg() -> BailingMoeV2Config:
    return BailingMoeV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        rope_fusion=False,
    )


def _moe_cfg(cfg: BailingMoeV2Config) -> MoEConfig:
    return MoEConfig(
        dim=cfg.hidden_size,
        inter_dim=cfg.intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.num_experts,
        n_shared_experts=cfg.num_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=cfg.n_group,
        n_limited_groups=cfg.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        force_e_score_correction_bias=True,
        score_func=cfg.score_function,
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=cfg.norm_topk_prob,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        shared_expert_inter_dim=cfg.moe_intermediate_size,
        shared_expert_activation="swiglu",
        softmax_before_topk=False,
        dtype=torch.float32,
    )


def _build_model() -> tuple[BailingMoeV2ForCausalLM, BailingMoeV2Config]:
    cfg = _tiny_cfg()
    model = BailingMoeV2ForCausalLM(cfg, moe_config=_moe_cfg(cfg), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.eval(), cfg


def test_supports_logits_to_keep():
    """Recipe gate (a): forward must expose a ``logits_to_keep`` parameter."""
    model, _ = _build_model()
    assert _supports_logits_to_keep(model) is True


def test_logits_to_keep_and_hidden_states():
    """Recipe gate (b): with logits_to_keep=1 + output_hidden_states=True the output
    must carry full-sequence hidden states while logits cover only the last token."""
    model, cfg = _build_model()

    seq_len = 16
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # train_ft.py checks ``"hidden_states" not in out`` to reject the fused path.
    assert ("hidden_states" in out) or out.hidden_states is not None

    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    # Hidden states span the FULL sequence (fused CE consumes all positions).
    assert hidden_states.shape == (1, seq_len, cfg.hidden_size)

    # Logits correspond to only the last token.
    logits = out.logits
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_default_forward_full_logits():
    """Default call (logits_to_keep=0, no output_hidden_states) yields full-length
    logits and no hidden states, preserving the pre-change behavior."""
    model, cfg = _build_model()

    seq_len = 16
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    logits = out.logits
    assert logits.shape == (1, seq_len, cfg.vocab_size)
    # Hidden states are gated off by default.
    assert ("hidden_states" not in out) and out.hidden_states is None


def test_default_logits_match_logits_to_keep_slice():
    """The last-token logits from logits_to_keep=1 must equal the last position of
    the full (logits_to_keep=0) logits — i.e. slicing does not change the values."""
    model, cfg = _build_model()

    seq_len = 16
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    with torch.no_grad():
        full = model(input_ids).logits
        last = model(input_ids, logits_to_keep=1).logits

    torch.testing.assert_close(last[:, -1, :], full[:, -1, :])
