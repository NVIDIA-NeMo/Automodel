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

"""Unit tests for dual-stream model expansion.

The property the whole design rests on is that an expanded model reproduces its parent
*bit-exactly* until the expansion weights learn. Several tests below deliberately assert
bit-exactness rather than a tolerance: an expanded layer that merely comes close is a bug,
because zero-initialized output projections make the two paths algebraically identical.

Two traps are worth knowing when adding tests here.

* **Zero-init hides stream-B bugs.** At initialization the output projections discard
  whatever stream B computed, so a function-preservation check cannot see an error inside
  stream B. ``test_no_state_leaks_between_passes`` perturbs the expansion weights first,
  which is what makes such errors observable.
* **Streams stay identical until an expanded layer diverges them.** Expanding only layer 1
  leaves layer 1's input identical on both streams, because layer 0 runs in skip mode. Any
  test that needs genuinely diverged streams has to expand an *upstream* layer too.
"""

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from nemo_automodel.components.expansion import (
    ExpandedLinear,
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
    freeze_non_expansion_parameters,
    is_expansion_parameter,
)

VOCAB, HIDDEN, LAYERS, SEQ, BATCH = 64, 32, 4, 8, 2


def _build(layers=None, merge_weight=0.5, dtype=torch.float32):
    """A tiny Llama, optionally expanded. Deterministic across calls."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ * 2,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config).to(dtype).eval()
    if layers is not None:
        apply_expansion(model, ExpansionConfig(enabled=True, layers=layers, merge_weight=merge_weight))
    return model


@pytest.fixture
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (BATCH, SEQ))


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


@pytest.mark.parametrize("merge_weight", [0.0, 0.5, 1.0])
def test_expanded_model_reproduces_parent_bit_exactly(input_ids, merge_weight):
    """The expanded model is its parent until the expansion weights learn.

    Holds for any merge weight because the two streams are seeded identically and the
    output projections start at zero, so the merge has nothing to interpolate between.
    """
    parent = _logits(_build(), input_ids)
    expanded = _logits(_build(layers=[1, 2], merge_weight=merge_weight), input_ids)
    assert torch.equal(expanded, parent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_function_preservation_holds_in_reduced_precision(input_ids, dtype):
    """Precision does not weaken the guarantee, so bf16 and fp16 runs start from the parent.

    The output projections start at exactly zero, and zero is representable in every float
    format, so stream B contributes nothing no matter how coarse the mantissa. The two
    places that could still drift -- the skip-mode delta and the final merge -- accumulate
    in fp32 before casting back, which is why this is bit-exact rather than merely close.
    """
    parent = _logits(_build(dtype=dtype), input_ids)
    expanded = _logits(_build(layers=[1, 2], dtype=dtype), input_ids)
    assert torch.equal(expanded, parent)


def test_skip_mode_equals_a_zero_expansion_weight(input_ids):
    """A non-expanded layer computes exactly what an expanded layer with ``W_b == 0`` does.

    This is the claim that lets unexpanded layers run once instead of twice. Comparing a
    fully expanded model whose expansion weights are zeroed against a partially expanded
    one exercises both code paths on the same weights.
    """
    fully_expanded = _build(layers=list(range(LAYERS)))
    partially_expanded = _build(layers=[1])
    for model in (fully_expanded, partially_expanded):
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.zero_()

    assert torch.equal(_logits(fully_expanded, input_ids), _logits(partially_expanded, input_ids))


def test_only_expansion_weights_are_trainable(input_ids):
    """Freezing the pretrained weights keeps stream A out of the autograd graph."""
    model = _build(layers=[1, 2])
    trainable, frozen = freeze_non_expansion_parameters(model)
    assert trainable > 0 and frozen > 0

    model.train()
    model(input_ids=input_ids, use_cache=False).logits.square().mean().backward()
    with_grad = [name for name, param in model.named_parameters() if param.grad is not None]
    assert with_grad
    assert all(is_expansion_parameter(name) for name in with_grad)


def test_only_output_projections_have_gradient_at_step_zero(input_ids):
    """Zero-initialized output projections gate their layer's input-side linears.

    At initialization the gradient of ``q_proj``'s expansion weight flows through
    ``o_proj``'s, which is zero, so it is zero too. Input-side weights start learning only
    once the output-side ones move. A regression here means the zero-init policy changed.
    """
    model = _build(layers=[1, 2])
    freeze_non_expansion_parameters(model)
    model.train()
    model(input_ids=input_ids, use_cache=False).logits.square().mean().backward()

    nonzero = [
        name for name, param in model.named_parameters() if param.grad is not None and param.grad.abs().sum() > 0
    ]
    assert len(nonzero) == 2 * len(ExpansionConfig().zero_init_modules)
    assert all("o_proj" in name or "down_proj" in name for name in nonzero)


def test_module_paths_and_state_dict_keys_are_preserved(input_ids):
    """Patching in place keeps tensor-parallel plans and ``to_hf``/``from_hf`` working.

    TP plans match module paths segment-wise and checkpoint conversion matches state-dict
    keys, so a wrapper that inserted a level would silently break both.
    """
    parent, expanded = _build(), _build(layers=[1, 2])

    parent_paths = {name for name, _ in parent.named_modules()}
    expanded_paths = {name for name, _ in expanded.named_modules()}
    assert not parent_paths - expanded_paths
    assert not {name for name in expanded_paths - parent_paths if not name.endswith("expansion")}

    parent_keys, expanded_keys = set(parent.state_dict()), set(expanded.state_dict())
    assert not parent_keys - expanded_keys
    assert not {key for key in expanded_keys - parent_keys if not is_expansion_parameter(key)}


def test_no_state_leaks_between_passes(input_ids):
    """Running the layer twice must not let the passes share mutable state.

    The KV cache was the first instance: the A pass appended its keys and the B pass
    appended its own, so attention ran against twice the keys. Zero-init hid it from every
    function-preservation check, so the expansion weights are perturbed first here.
    """
    model = _build(layers=[1, 2])
    with torch.no_grad():
        for _, param in expansion_parameters(model):
            param.normal_(0.0, 0.05)

    assert torch.equal(_logits(model, input_ids), _logits(model, input_ids))


def _greedy_without_cache(model, input_ids, steps):
    """Decode greedily with a full forward per step and no cache anywhere.

    Args:
        model: A causal LM.
        input_ids: Token ids of shape ``[batch, sequence]``.
        steps: How many tokens to append.

    Returns:
        Token ids of shape ``[batch, sequence + steps]``.
    """
    for _ in range(steps):
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        input_ids = torch.cat([input_ids, logits[:, -1:].argmax(-1)], dim=1)
    return input_ids


def test_cached_generation_is_refused(input_ids):
    """Decoding from a cache must fail loudly rather than return plausible wrong tokens.

    Stream B's keys and values differ from stream A's once the expansion weights learn, so
    one cache cannot serve both. Nothing about the output looks wrong when it is: the
    tokens are well-formed, just computed against the wrong history.
    """
    model = _build(layers=[1, 2])
    with torch.no_grad():
        for _, param in expansion_parameters(model):
            param.normal_(0.0, 0.05)

    with pytest.raises(NotImplementedError, match="use_cache=False"):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=True)


def test_uncached_generation_is_correct(input_ids):
    """The path the refusal recommends has to actually produce the right tokens.

    Compared against a decode loop written out by hand, so this checks the recommendation
    rather than merely that ``generate`` returns something. The expansion weights are
    perturbed first: at their initial values any generation check passes, because the
    expanded model is its parent.
    """
    model = _build(layers=[1, 2])
    with torch.no_grad():
        for _, param in expansion_parameters(model):
            param.normal_(0.0, 0.05)

    expected = _greedy_without_cache(model, input_ids, steps=4)
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=False)
    assert torch.equal(generated, expected)


def test_expanded_model_generates_exactly_like_its_parent(input_ids):
    """Function preservation, observed through the decoding loop rather than the logits."""
    with torch.no_grad():
        parent = _build().generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=False)
        expanded = _build(layers=[1, 2]).generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=False)
    assert torch.equal(expanded, parent)


def test_expanded_linear_replaces_the_module_in_place():
    """The expanded linear must remain an ``nn.Linear`` holding the pretrained weight."""
    model = _build(layers=[0])
    q_proj = model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, ExpandedLinear)
    assert isinstance(q_proj, nn.Linear)
    assert q_proj.expansion.weight.shape == q_proj.weight.shape


def test_zero_init_applies_only_to_output_projections():
    """Copy-initialized input-side weights, zero-initialized output-side ones."""
    model = _build(layers=[0])
    for name, param in expansion_parameters(model):
        is_output_projection = "o_proj" in name or "down_proj" in name
        assert bool(param.abs().sum() == 0) is is_output_projection, name


def test_disabled_config_is_a_no_op(input_ids):
    """``enabled=False`` must leave the model untouched."""
    model = _build()
    before = _logits(model, input_ids)
    apply_expansion(model, ExpansionConfig(enabled=False))
    assert not list(expansion_parameters(model))
    assert torch.equal(_logits(model, input_ids), before)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"merge_weight": 1.5}, "merge_weight"),
        ({"zero_init_modules": ["not_a_projection"]}, "zero_init_modules"),
    ],
)
def test_config_rejects_invalid_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExpansionConfig(enabled=True, **kwargs)


def test_out_of_range_layer_is_rejected():
    with pytest.raises(ValueError, match="outside the decoder stack"):
        apply_expansion(_build(), ExpansionConfig(enabled=True, layers=[LAYERS]))


def test_unsupported_architecture_is_rejected():
    """A model without a decoder layer list needs explicit support, not a silent no-op."""

    class Shapeless(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(HIDDEN, HIDDEN)

    with pytest.raises(AttributeError, match="decoder layer list"):
        apply_expansion(Shapeless(), ExpansionConfig(enabled=True))


def test_freeze_without_expansion_is_rejected():
    """Freezing a model that was never expanded would leave nothing to train."""
    with pytest.raises(ValueError, match="apply_expansion"):
        freeze_non_expansion_parameters(_build())
