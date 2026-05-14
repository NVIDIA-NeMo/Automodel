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

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.target import (
    HFEagle3TargetModel,
    _shift_left_with_zero,
)


def _build_tiny_target(num_hidden_layers: int = 4) -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=32,
    )
    config.torch_dtype = torch.float32
    return LlamaForCausalLM(config).to(torch.float32).eval()


def test_default_aux_layer_ids_returns_three_unique_for_normal_models():
    target = HFEagle3TargetModel(_build_tiny_target(num_hidden_layers=16))
    ids = target.aux_layer_ids
    assert ids == [1, 7, 12]
    assert len(set(ids)) == len(ids)


def test_default_aux_layer_ids_deduplicates_on_shallow_models():
    """Models with fewer than ~7 layers must not collide on the default recipe.

    With ``num_layers=5`` the raw recipe yields ``[1, 1, 1]``; with
    ``num_layers=6`` it yields ``[1, 2, 2]``. Both must be deduplicated
    (order-preserving) so ``generate_batch`` does not register multiple
    hooks on the same module and fail the post-forward length check.
    """
    target_5 = HFEagle3TargetModel(_build_tiny_target(num_hidden_layers=5))
    assert target_5.aux_layer_ids == [1]

    target_6 = HFEagle3TargetModel(_build_tiny_target(num_hidden_layers=6))
    assert target_6.aux_layer_ids == [1, 2]


def test_generate_batch_aux_hidden_states_shape_and_layer_capture():
    """``aux_hidden_states`` must be the per-layer hiddens concatenated along H.

    Run the target end-to-end, capture the hidden states via the public
    ``generate_batch`` API, and verify the result is exactly
    ``concat([h_layer_1, h_layer_2, h_layer_3], dim=-1)`` recomputed via
    HF's ``output_hidden_states=True``.
    """
    torch.manual_seed(0)
    model = _build_tiny_target(num_hidden_layers=8)
    target = HFEagle3TargetModel(model, aux_layer_ids=[1, 3, 5])

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    batch = target.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)

    hidden = model.config.hidden_size
    assert batch.aux_hidden_states.shape == (batch_size, seq_len, hidden * 3)

    # Reference: HF's own ``output_hidden_states``. ``hidden_states[0]``
    # is the input embedding and ``hidden_states[k+1]`` is the output of
    # layer ``k``, so capturing layers 1/3/5 must match indices 2/4/6.
    with torch.no_grad():
        ref = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    expected = torch.cat([ref.hidden_states[2], ref.hidden_states[4], ref.hidden_states[6]], dim=-1)
    torch.testing.assert_close(batch.aux_hidden_states, expected)


def test_generate_batch_shifts_logits_input_ids_and_loss_mask():
    """Targets must be left-shifted with zero tail to align with next-token labels."""
    torch.manual_seed(0)
    model = _build_tiny_target(num_hidden_layers=4)
    target = HFEagle3TargetModel(model, aux_layer_ids=[1, 2, 3])

    batch_size, seq_len = 2, 5
    input_ids = torch.randint(1, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.long)

    batch = target.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)

    torch.testing.assert_close(batch.input_ids, _shift_left_with_zero(input_ids))
    torch.testing.assert_close(batch.loss_mask, _shift_left_with_zero(loss_mask))
    # attention_mask is NOT shifted -- it tracks padding positions, not
    # the next-token target offset.
    torch.testing.assert_close(batch.attention_mask, attention_mask)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    torch.testing.assert_close(batch.logits, _shift_left_with_zero(ref.logits))


def test_generate_batch_raises_for_out_of_bounds_aux_layer_id():
    model = _build_tiny_target(num_hidden_layers=4)
    target = HFEagle3TargetModel(model, aux_layer_ids=[1, 2, 99])

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)
    try:
        target.generate_batch(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-bounds aux layer id")
