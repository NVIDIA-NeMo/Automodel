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

"""Native NemotronForDecision: bidirectional reads, parity with the stock encoder, checkpoint names, registry."""

from __future__ import annotations

import pytest
import torch
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

from nemo_automodel.components.models.nemotron_decision.model import NemotronForDecision


def _config(**overrides):
    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        mask_token_id=3,
        architectures=["NemotronForDecision"],
        attn_implementation="sdpa",
    )
    return Ministral3Config(**{**values, **overrides})


@pytest.fixture
def model():
    torch.manual_seed(0)
    return NemotronForDecision(_config()).eval()


def test_reads_are_bidirectional(model):
    ids = torch.tensor([[5, 6, 7, 8, 3]])
    changed = ids.clone()
    changed[0, -2] = 9
    with torch.no_grad():
        a = model(input_ids=ids, output_last_hidden_states_only=True).last_hidden_state
        b = model(input_ids=changed, output_last_hidden_states_only=True).last_hidden_state
    assert not model.config.is_causal
    assert not torch.allclose(a[0, 0], b[0, 0])  # the first position sees a later token


def test_matches_the_stock_encoder_with_causal_attention_disabled(model):
    config = _config()
    config.is_causal = False
    reference = Ministral3Model(config).eval()
    for layer in reference.layers:
        layer.self_attn.is_causal = False
    reference.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in model.state_dict().items() if k.startswith("encoder.")}
    )
    ids = torch.tensor([[5, 6, 7, 8, 3]])
    with torch.no_grad():
        expected = reference(input_ids=ids, position_ids=torch.arange(5)[None], use_cache=False).last_hidden_state
        hidden = model(input_ids=ids, output_last_hidden_states_only=True).last_hidden_state
        logits = model(input_ids=ids).logits
    torch.testing.assert_close(hidden, expected)
    torch.testing.assert_close(logits, model.diffusion_head(hidden))


def test_parameter_names_match_the_checkpoint_layout(model):
    names = set(model.state_dict())
    assert "diffusion_head.weight" in names and "encoder.embed_tokens.weight" in names
    assert all(name.startswith(("encoder.", "diffusion_head.")) for name in names)
    assert model.get_output_embeddings() is model.diffusion_head
    assert model.get_input_embeddings() is model.encoder.embed_tokens


def _bf16_model(**kwargs):
    torch.manual_seed(0)
    return NemotronForDecision(_config(), **kwargs).to(torch.bfloat16).eval()


def test_logits_dtype_sets_the_dtype_of_the_forward_logits():
    ids = torch.tensor([[5, 6, 7, 8, 3]])
    with torch.no_grad():
        fp32_model = _bf16_model(logits_dtype=torch.float32)
        hidden = fp32_model(input_ids=ids, output_last_hidden_states_only=True).last_hidden_state
        logits = fp32_model(input_ids=ids).logits
        default_logits = _bf16_model()(input_ids=ids).logits
    # Float64 oracle over the same BF16 operands; the head weight stays BF16.
    expected = hidden.double() @ fp32_model.diffusion_head.weight.double().T
    assert fp32_model.diffusion_head.weight.dtype == torch.bfloat16
    assert hidden.dtype == torch.bfloat16 and logits.dtype == torch.float32
    torch.testing.assert_close(logits.double(), expected, rtol=1e-5, atol=1e-5)
    assert default_logits.dtype == torch.bfloat16
    torch.testing.assert_close(default_logits.double(), expected, rtol=2**-7, atol=2**-7)


def test_logits_dtype_unset_matches_a_plain_linear_bit_for_bit(model):
    ids = torch.tensor([[5, 6, 7, 8, 3]])
    with torch.no_grad():
        hidden = model(input_ids=ids, output_last_hidden_states_only=True).last_hidden_state
        logits = model(input_ids=ids).logits
    assert torch.equal(logits, torch.nn.functional.linear(hidden, model.diffusion_head.weight))
    assert model.diffusion_head.bias is None


def test_registry_routes_the_decision_architecture_to_the_native_class():
    from nemo_automodel._transformers.registry import ModelRegistry

    assert ModelRegistry.resolve_custom_model_cls("NemotronForDecision", _config()) is NemotronForDecision
    assert not NemotronForDecision.supports_config(_config(quantization_config={"quant_method": "modelopt"}))
