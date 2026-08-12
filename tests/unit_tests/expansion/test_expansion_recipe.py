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

"""Model expansion as the recipe applies it, through ``apply_model_infrastructure``.

The recipe hands an ``ExpansionConfig`` down the same route ``freeze_config`` takes, and
``apply_model_infrastructure`` splits the work in two: it allocates the expansion weights
beside PEFT, before sharding, and gives them their values after the checkpoint load. That
split is invisible from outside -- what a caller can observe is that the model comes back
expanded, frozen everywhere but the expansion weights, and still numerically its parent.

PEFT is refused outright, because it patches the same linears expansion does. The refusal
is checked before anything is applied, so a bad configuration fails on the configuration
rather than partway through the build. Pipeline parallelism needs no refusal: the two
streams travel between layers concatenated on the hidden axis, so a stage boundary carries
one ordinary tensor.
"""

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.expansion import (
    ExpansionConfig,
    expansion_parameters,
    is_expansion_parameter,
)

VOCAB, HIDDEN, LAYERS, SEQ, BATCH = 64, 32, 4, 8, 2
EXPANDED_LAYERS = [1, 2]


def _build() -> LlamaForCausalLM:
    """A tiny Llama on CPU, deterministic across calls."""
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
    return LlamaForCausalLM(config).eval()


def _apply(model: LlamaForCausalLM, expansion_config: ExpansionConfig | None, **kwargs) -> LlamaForCausalLM:
    """Run the infrastructure step the recipe runs, single-process and without a checkpoint."""
    return apply_model_infrastructure(
        model=model,
        is_meta_device=False,
        device=torch.device("cpu"),
        mesh=MeshContext(),
        expansion_config=expansion_config,
        load_base_model=False,
        **kwargs,
    )


def _logits(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(1))


def test_recipe_path_expands_freezes_and_preserves_the_parent(input_ids):
    """One call has to leave the model expanded, frozen, initialized and unchanged.

    Bit-exactness is the assertion that catches a half-finished initialization: expansion
    weights left at whatever the allocation produced would still give the parent's logits
    only if the output projections really are zero, and the trainable-parameter check
    catches the freezing half.
    """
    expected = _logits(_build(), input_ids)
    model = _apply(_build(), ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS))

    expansion = dict(expansion_parameters(model))
    assert expansion, "the recipe path produced no expansion weights"

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable and all(is_expansion_parameter(name) for name in trainable)
    assert len(trainable) == len(expansion)

    zeroed = [name for name, param in expansion.items() if param.abs().sum() == 0]
    assert len(zeroed) == len(EXPANDED_LAYERS) * len(ExpansionConfig().zero_init_modules)

    assert torch.equal(_logits(model, input_ids), expected)


def test_disabled_expansion_config_is_a_no_op(input_ids):
    """``enabled: false`` in the YAML must not touch the model."""
    expected = _logits(_build(), input_ids)
    model = _apply(_build(), ExpansionConfig(enabled=False))

    assert not list(expansion_parameters(model))
    assert torch.equal(_logits(model, input_ids), expected)


def test_no_expansion_config_is_a_no_op(input_ids):
    """Configs without an ``expansion:`` block go through the pre-existing path."""
    expected = _logits(_build(), input_ids)
    model = _apply(_build(), None)

    assert not list(expansion_parameters(model))
    assert torch.equal(_logits(model, input_ids), expected)


def test_combining_with_peft_is_refused():
    """Refused before anything is applied.

    The sentinel object here carries none of the attributes a real PEFT config does, so a
    refusal raised too late would surface as an ``AttributeError`` from inside PEFT rather
    than as the ``NotImplementedError`` under test.
    """
    with pytest.raises(NotImplementedError, match="PEFT"):
        _apply(_build(), ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS), peft_config=object())


def test_expanded_model_declares_its_inter_stage_shapes():
    """Pipeline parallelism precomputes stage shapes; expansion has to correct them.

    ``functional._precompute_stage_shapes`` sizes the inter-stage tensor from
    ``config.hidden_size`` instead of inferring it, so without this hook stage 0 emits the
    doubled carrier into a slot declared one stream wide and the schedule raises
    ``PipeliningShapeError`` at the first step.
    """
    model = _apply(_build(), ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS))
    hook = getattr(model, "get_pipeline_stage_metas", None)
    assert callable(hook), "the pipeline shape hook was not attached"

    inputs, outputs = hook(is_first=True, microbatch_size=2, seq_len=SEQ, dtype=torch.float32)
    assert inputs[0].shape == (2, SEQ)
    assert outputs[0].shape == (2, SEQ, VOCAB), "a stage owning the LM head emits logits"

    model.lm_head = None  # what pipeline splitting leaves on a non-final stage
    inputs, outputs = hook(is_first=False, microbatch_size=2, seq_len=SEQ, dtype=torch.float32)
    assert inputs[0].shape == (2, SEQ, 2 * HIDDEN), "a later stage receives the two-stream carrier"
    assert outputs[0].shape == (2, SEQ, 2 * HIDDEN), "a non-final stage emits it too"


def test_an_unexpanded_model_declares_nothing():
    """The hook must not appear on models that are not expanded."""
    assert getattr(_apply(_build(), None), "get_pipeline_stage_metas", None) is None
