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

"""Tiny real model consumers verify cross-document isolation and causal attention."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.attention import adapt_model_inputs
from nemo_automodel.components.models.common.packing import configure_packing


@pytest.mark.parametrize("attention_backend", ["sdpa", "eager"])
def test_block_causal_real_model_has_no_cross_document_leakage(attention_backend):
    torch.manual_seed(1)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attn_implementation=attention_backend,
            use_cache=False,
        )
    ).eval()
    configure_packing(attention_backend, model=model)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    original = {
        "input_ids": ids,
        "labels": ids.clone(),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
        "document_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "padding_mask": torch.zeros_like(ids, dtype=torch.bool),
    }
    batch = adapt_model_inputs(original)
    with torch.no_grad():
        packed = model(
            input_ids=ids, position_ids=batch["position_ids"], _packed_seq_ids=batch["attention_document_ids"]
        ).logits
        isolated = model(input_ids=ids[:, 3:]).logits
        altered = ids.clone()
        altered[:, :3] = 9
        perturbed = model(
            input_ids=altered, position_ids=batch["position_ids"], _packed_seq_ids=batch["attention_document_ids"]
        ).logits
    torch.testing.assert_close(packed[:, 3:], isolated)
    torch.testing.assert_close(packed[:, 3:], perturbed[:, 3:])


@pytest.mark.parametrize("attention_backend", ["sdpa", "eager"])
def test_xdoc_real_model_influence_causality_and_batch_independence(attention_backend, monkeypatch):
    from nemo_automodel.components.datasets.zephon.attention import (
        enable_xdoc_attention,
    )

    torch.manual_seed(3)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=48,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attn_implementation=attention_backend,
            use_cache=False,
        )
    ).eval()
    configure_packing(attention_backend, model=model)
    ids = torch.tensor([[2, 3, 4, 5, 6, 1], [12, 13, 14, 15, 16, 1]])
    positions = torch.tensor([[0, 1, 2, 0, 1, 0], [0, 1, 2, 0, 1, 0]])
    padding = torch.tensor([[False, False, False, False, False, True]]).expand(2, -1)
    isolated = adapt_model_inputs(
        {
            "input_ids": ids,
            "labels": ids.clone(),
            "position_ids": positions,
            "padding_mask": padding,
            "document_ids": torch.tensor([[1, 1, 1, 2, 2, 0]] * 2),
        },
    )
    joined = enable_xdoc_attention(isolated, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT)
    for key in ("input_ids", "labels", "padding_mask", "document_ids"):
        assert joined[key] is isolated[key]
    assert joined["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0]] * 2
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 2, 0, 1, 0]] * 2))
    assert joined is isolated
    joined_positions = joined["position_ids"]
    assert joined["attention_document_ids"].tolist() == [[1, 1, 1, 1, 1, 0]] * 2
    assert "attention_mask" not in joined
    observed_masks = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def observe_sdpa(query, key, value, attn_mask=None, **kwargs):
        """Observe SDPA without changing its inputs.

        Args:
            query: Tensor of shape [batch, heads, sequence, head_dim].
            key: Tensor of shape [batch, heads, sequence, head_dim].
            value: Tensor of shape [batch, heads, sequence, head_dim].
            attn_mask: Optional tensor broadcastable to [batch, heads, sequence, sequence].
            **kwargs: Remaining SDPA arguments.

        Returns:
            Tensor of shape [batch, heads, sequence, head_dim].
        """
        observed_masks.append(attn_mask)
        return sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe_sdpa)
    with torch.no_grad():
        baseline = model(
            input_ids=ids,
            position_ids=joined_positions,
            attention_mask=joined.get("attention_mask"),
            _packed_seq_ids=joined.get("attention_document_ids"),
        ).logits
        ordinary = model(input_ids=ids, attention_mask=~padding).logits
        torch.testing.assert_close(baseline[:, :5], ordinary[:, :5])
        altered = ids.clone()
        altered[0, :3] = torch.tensor([20, 21, 22])
        changed = model(
            input_ids=altered,
            position_ids=joined_positions,
            attention_mask=joined.get("attention_mask"),
            _packed_seq_ids=joined.get("attention_document_ids"),
        ).logits
        assert not torch.allclose(baseline[0, 3:5], changed[0, 3:5])
        torch.testing.assert_close(baseline[1, :5], changed[1, :5])
        future = ids.clone()
        future[:, 4:] = 30
        future_logits = model(
            input_ids=future,
            position_ids=joined_positions,
            attention_mask=joined.get("attention_mask"),
            _packed_seq_ids=joined.get("attention_document_ids"),
        ).logits
        torch.testing.assert_close(baseline[:, :4], future_logits[:, :4])
    if attention_backend == "sdpa":
        # The real model uses ordinary causal SDPA without a quadratic mask.
        assert observed_masks and observed_masks[0] is None
    logits = model(
        input_ids=ids,
        position_ids=joined_positions,
        attention_mask=joined.get("attention_mask"),
        _packed_seq_ids=joined.get("attention_document_ids"),
    ).logits
    loss = torch.nn.functional.cross_entropy(logits[:, :5].reshape(-1, 48), ids[:, :5].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad()
    ordinary = model(input_ids=ids, attention_mask=~padding).logits
    ordinary_loss = torch.nn.functional.cross_entropy(ordinary[:, :5].reshape(-1, 48), ids[:, :5].reshape(-1))
    torch.testing.assert_close(loss, ordinary_loss)
    ordinary_loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-4, msg=name)
