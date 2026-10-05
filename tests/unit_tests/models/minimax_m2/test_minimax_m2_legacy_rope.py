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

"""Checkpoint-config and independent mathematical coverage of MiniMax M2 partial RoPE."""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2Model


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch",
        enable_hf_state_dict_adapter=False,
    )


@pytest.mark.parametrize("parameters_present", [False, True])
@pytest.mark.parametrize(
    ("rotary_dim", "explicit_factor", "expected_dim"),
    [(64, None, 64), (None, None, 128), (64, 0.25, 32), (64, 0.5, 64), (64, 1.0, 128)],
)
def test_checkpoint_rotary_dimension_and_roundtrip(
    tmp_path: Path,
    parameters_present: bool,
    rotary_dim: int | None,
    explicit_factor: float | None,
    expected_dim: int,
) -> None:
    # Keep the released checkpoint's head_dim=128 / rotary_dim=64. Reduce only
    # unrelated dimensions so this tests the real HF config loader without weights.
    payload = {
        "model_type": "minimax_m2",
        "architectures": ["MiniMaxM2ForCausalLM"],
        "vocab_size": 32,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "max_position_embeddings": 32,
        "rope_theta": 5000000.0,
    }
    if rotary_dim is not None:
        payload["rotary_dim"] = rotary_dim
    if parameters_present:
        payload["rope_parameters"] = {"rope_theta": 5000000.0, "rope_type": "default"}
        if explicit_factor is not None:
            payload["rope_parameters"]["partial_rotary_factor"] = explicit_factor
    elif explicit_factor is not None:
        payload["partial_rotary_factor"] = explicit_factor
    (tmp_path / "config.json").write_text(json.dumps(payload))
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    model = MiniMaxM2Model(config, _backend())
    assert model.rotary_emb.rotary_dim == expected_dim
    assert model.rotary_emb.partial_rotary_factor == expected_dim / 128
    assert config.rope_parameters.get("partial_rotary_factor", 1.0) == expected_dim / 128
    assert config.rope_parameters["rope_theta"] == 5000000.0
    assert config.rope_parameters["rope_type"] == "default"

    saved = tmp_path / "saved"
    config.save_pretrained(saved)
    reloaded = AutoConfig.from_pretrained(saved, local_files_only=True)
    assert MiniMaxM2Model(reloaded, _backend()).rotary_emb.rotary_dim == expected_dim


def test_legacy_partial_rope_matches_independent_forward_and_gradient() -> None:
    config = AutoConfig.for_model(
        "minimax_m2",
        vocab_size=32,
        bos_token_id=0,
        eos_token_id=1,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        rotary_dim=64,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        rope_theta=5000000.0,
    )
    model = MiniMaxM2Model(config, _backend())
    model.rotary_emb.device = torch.device("cpu")
    torch.manual_seed(91)
    q = torch.randn(1, 8, 2, 128, requires_grad=True)
    k = torch.randn(1, 8, 1, 128, requires_grad=True)
    actual_q, actual_k = model.rotary_emb(q, k)

    # Independent half-split rotation of exactly the first 64 channels.
    # This does not use the production frequency builder or rotary helper.
    angle = torch.tensor(
        [[position / (5000000.0 ** (2 * channel / 64)) for channel in range(32)] for position in range(8)]
    )[None, :, None, :]
    expected = []
    for value in (q, k):
        first = value[..., :32]
        second = value[..., 32:64]
        expected.append(
            torch.cat(
                (
                    first * angle.cos() - second * angle.sin(),
                    second * angle.cos() + first * angle.sin(),
                    value[..., 64:],
                ),
                dim=-1,
            )
        )
    torch.testing.assert_close(actual_q, expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_k, expected[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_q[..., 64:], q[..., 64:], rtol=0, atol=0)
    torch.testing.assert_close(actual_q[:, 0], q[:, 0], rtol=0, atol=0)
    assert not torch.equal(actual_q[:, 1:, :, :64], q[:, 1:, :, :64])

    q_grad = torch.randn_like(actual_q)
    k_grad = torch.randn_like(actual_k)
    actual_grad = torch.autograd.grad((actual_q, actual_k), (q, k), (q_grad, k_grad))
    expected_grad = torch.autograd.grad(tuple(expected), (q, k), (q_grad, k_grad))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
