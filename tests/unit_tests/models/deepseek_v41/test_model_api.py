# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Model API policy, FP32 readout, and independent shifted-label loss checks."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM


def _model(dtype: str = "float32") -> DeepseekV41ForCausalLM:
    config = DeepseekV41Config(
        text_config=DeepseekV41TextConfig(
            vocab_size=17,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[],
            dtype=dtype,
        ),
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
    )
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")
    model = DeepseekV41ForCausalLM(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=getattr(torch, dtype))
    return model


def test_default_policy_is_explicit_and_routing_correction_is_fixed() -> None:
    config = _model().config
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
    assert model.config_class is DeepseekV41Config and model.base_model_prefix == "model"
    assert model.backend.attn == "tilelang"
    assert model.backend.linear == "torch" and model.backend.rms_norm == "torch_fp32"
    assert model.backend.dispatcher == "hybridep" and model.backend.experts == "torch_mm"
    assert model.moe_config.combine_in_fp32 and model.moe_config.gate_bias_update_factor == 0


def test_bf16_backbone_returns_unrounded_fp32_logits():
    torch.manual_seed(31)
    model = _model("bfloat16")
    with torch.no_grad():
        model.lm_head.weight.copy_(
            torch.linspace(-0.04321, 0.05137, model.lm_head.weight.numel()).view_as(model.lm_head.weight)
        )
    result = model(torch.tensor([[3, 5, 7, 9]]), return_hidden_states=True)
    assert result.hidden_states.dtype == torch.bfloat16
    assert result.logits.dtype == torch.float32
    expected = F.linear(result.hidden_states.float(), model.lm_head.weight)
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
    assert torch.count_nonzero(result.logits != result.logits.bfloat16().float()) > 0
    result.logits.square().mean().backward()
    assert model.lm_head.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_labels_match_independent_shifted_logprob_and_head_gradient():
    torch.manual_seed(17)
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    labels = ids.clone()
    labels[0, 2] = -100
    original = labels.clone()
    result = model(ids, labels=labels, return_hidden_states=True)
    # Compare API variants under the same grad mode and before backward.
    plain = model(ids, return_hidden_states=True)
    assert plain.loss is None
    torch.testing.assert_close(plain.logits, result.logits, rtol=0, atol=0)
    reference = result.logits.detach().clone().requires_grad_()
    targets = labels[:, 1:]
    valid = targets != -100
    probabilities = reference[:, :-1].log_softmax(-1)
    expected = -probabilities.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid].mean()
    torch.testing.assert_close(result.loss, expected, rtol=1e-6, atol=1e-7)
    assert result.loss.dtype == torch.float32 and result.loss.ndim == 0
    assert result.hidden_states.shape == (2, 4, model.config.text_config.hidden_size)
    assert "loss" in result and torch.equal(labels, original)
    result.logits.retain_grad()
    result.loss.backward()
    expected.backward()
    torch.testing.assert_close(result.logits.grad, reference.grad, rtol=1e-6, atol=1e-8)
    assert torch.count_nonzero(result.logits.grad[:, -1]) == 0
    assert torch.count_nonzero(result.logits.grad[0, 1]) == 0
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize(
    "metadata",
    [
        {"qkv_format": "thd"},
        {"packed_seq_ids": torch.ones(1, 4, dtype=torch.long)},
        {"seq_lens": torch.tensor([2, 2])},
        {"cu_seqlens": torch.tensor([0, 2, 4])},
        {"cu_seqlens_q": torch.tensor([0, 2, 4])},
    ],
)
def test_labels_reject_packing_before_numerical_forward(metadata):
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(ids, labels=ids, **metadata)


@pytest.mark.parametrize("logits_to_keep", [1, torch.tensor([0, 2])])
def test_labels_require_full_ordered_logits(logits_to_keep):
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(ValueError, match="logits for every input position"):
        model(ids, labels=ids, logits_to_keep=logits_to_keep)


def test_labels_shape_and_inputs_embeds_contract() -> None:
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    with pytest.raises(ValueError, match="logits for every input position"):
        model(ids, labels=ids[:, :-1])
    with pytest.raises(TypeError, match="unexpected keyword"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids), labels=ids)


def test_captured_streams_and_final_hidden_states_have_distinct_contracts() -> None:
    model = _model()
    ids = torch.tensor([[3, 4, 5, 6]])
    final = model(ids, return_hidden_states=True)
    captured = model(ids, output_hidden_states=True)
    assert final.hidden_states.shape == (1, 4, 16)
    assert len(captured.hidden_states) == 1
    assert captured.hidden_states[0].shape == (1, 4, 4, 16)
    torch.testing.assert_close(final.logits, captured.logits, rtol=0, atol=0)
