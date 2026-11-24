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

"""Model FLOPs of a text-only training step of Nemotron 3.5 Super VL.

A text batch runs the language model only: 88 layers (40 Mamba, 40 MoE, 8 attention),
``lm_head`` and the MTP depths the model was built with. The recipe takes the count from
the model (``MFUConfig.build(model=...)``), because the checkpoint config is a composite
without a FLOPs formula and keeps one MTP depth where training runs two through one
shared block.

The expected values are derived in this file from the layer shapes of ``llm_config`` in
``nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16``, not from the FLOPs helpers. A
GEMM counts six FLOPs per multiply-accumulate (forward, input gradient and weight
gradient, two each). Two conventions of the Nemotron-H count are taken as given: a query
attends half of the sequence on average under the causal mask, and the state-space scan,
which is not a GEMM, counts 21 FLOPs per element of its state.
"""

import pytest
import torch
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from nemo_automodel._transformers.mfu import MFUConfig
from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniConfig,
    NemotronOmniForConditionalGeneration,
)
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
from nemo_automodel.components.models.nemotron_v3.mtp import build_mtp_config_from_hf

pytestmark = pytest.mark.cpu_only

SEQ_LEN = 8192

# One symbol per layer of the language model: M Mamba, E MoE, * attention.
LAYER_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
NUM_MAMBA, NUM_MOE, NUM_ATTENTION = 40, 40, 8

HIDDEN = 4096
VOCAB = 131072

# Mamba block: 128 heads of 64 channels, 8 groups of state size 128.
MAMBA_INNER = 128 * 64
MAMBA_IN_PROJ = HIDDEN * (2 * MAMBA_INNER + 2 * 8 * 128 + 128)  # hidden -> gate, x, B, C and dt
MAMBA_OUT_PROJ = MAMBA_INNER * HIDDEN
MAMBA_SCAN_STATE = MAMBA_INNER * 128

# MoE block: 22 of 512 routed experts in a latent space of 1024, one shared expert on the hidden state.
TOP_K = 22
EXPERT = 2 * 1024 * 2688  # latent -> 2688 -> latent
MOE_LATENT_PROJ = 2 * HIDDEN * 1024  # hidden -> latent and latent -> hidden
MOE_SHARED_EXPERT = 2 * HIDDEN * 5376  # hidden -> 5376 -> hidden
MOE_ROUTER = HIDDEN * 512
MOE = MOE_LATENT_PROJ + TOP_K * EXPERT + MOE_SHARED_EXPERT + MOE_ROUTER

# Attention block: 32 query heads and 2 key/value heads of 128 channels.
ATTENTION_PROJ = HIDDEN * (32 * 128) + 2 * HIDDEN * (2 * 128) + (32 * 128) * HIDDEN  # q, k and v, o
ATTENTION_CORE = 2 * (SEQ_LEN // 2) * 32 * 128  # scores and weighted values over SEQ_LEN / 2 keys
ATTENTION = ATTENTION_PROJ + ATTENTION_CORE

LM_HEAD = HIDDEN * VOCAB
MTP_FUSION = 2 * HIDDEN * HIDDEN  # [embedding, hidden state] -> hidden

BACKBONE_FLOPS_PER_TOKEN = (
    6 * (NUM_MAMBA * (MAMBA_IN_PROJ + MAMBA_OUT_PROJ) + NUM_MOE * MOE + NUM_ATTENTION * ATTENTION + LM_HEAD)
    + 21 * NUM_MAMBA * MAMBA_SCAN_STATE
)
# One MTP depth: the fusion projection, one attention and one MoE block, and lm_head for its loss.
MTP_DEPTH_FLOPS_PER_TOKEN = 6 * (MTP_FUSION + ATTENTION + MOE + LM_HEAD)


def _llm_config(**overrides: int) -> NemotronHConfig:
    """Build the language-model config of the checkpoint, with the fields that set layer shapes."""
    block_type = {"M": "mamba", "E": "moe", "*": "attention"}
    fields = {
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "layers_block_type": [block_type[symbol] for symbol in LAYER_PATTERN],
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 2688,
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "n_routed_experts": 512,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": 2688,
        "moe_shared_expert_intermediate_size": 5376,
        "moe_latent_size": 1024,
        # The checkpoint keeps one MTP depth: an attention block and an MoE block.
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
    }
    fields.update(overrides)
    return NemotronHConfig(**fields)


def _language_model(config: NemotronHConfig, *, mtp_depths: int, shared_mtp_block: bool = True) -> NemotronHForCausalLM:
    """Build the language model without its weights: the attributes its FLOPs count reads."""
    model = NemotronHForCausalLM.__new__(NemotronHForCausalLM)
    # nn.Module.__setattr__ needs the module machinery; bypass it for a model without weights.
    object.__setattr__(model, "config", config)
    object.__setattr__(
        model,
        "mtp_config",
        build_mtp_config_from_hf(config, num_nextn_predict_layers=mtp_depths, use_repeated_layer=shared_mtp_block),
    )
    return model


def _vl_model(llm_config: NemotronHConfig, *, mtp_depths: int) -> NemotronOmniForConditionalGeneration:
    """Build the vision-language model without its weights, around the language model above."""
    model = NemotronOmniForConditionalGeneration.__new__(NemotronOmniForConditionalGeneration)
    # A RADIO ViT-H tower: 32 layers of width 1280 that a text batch never runs.
    vision_config = {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16, "patch_size": 16}
    object.__setattr__(model, "config", NemotronOmniConfig(llm_config=llm_config, vision_config=vision_config))
    object.__setattr__(model, "language_model", _language_model(llm_config, mtp_depths=mtp_depths))
    return model


def test_derivation_covers_the_layers_of_the_checkpoint():
    assert len(LAYER_PATTERN) == 88
    assert [LAYER_PATTERN.count(symbol) for symbol in "ME*"] == [NUM_MAMBA, NUM_MOE, NUM_ATTENTION]
    assert BACKBONE_FLOPS_PER_TOKEN == 75_874_959_360
    assert MTP_DEPTH_FLOPS_PER_TOKEN == 4_891_607_040


@pytest.mark.parametrize("mtp_depths", [0, 2])
def test_language_model_counts_backbone_and_the_mtp_depths_it_runs(mtp_depths):
    model = _language_model(_llm_config(), mtp_depths=mtp_depths)

    flops = model.model_flops(gbs=1, seq_len=SEQ_LEN)

    assert flops == SEQ_LEN * (BACKBONE_FLOPS_PER_TOKEN + mtp_depths * MTP_DEPTH_FLOPS_PER_TOKEN)


@pytest.mark.parametrize("mtp_depths", [0, 2])
def test_recipe_calculator_counts_the_language_model_of_the_vl_model(mtp_depths):
    """The recipe builds its calculator from the VL model; a text batch then counts the language model only."""
    calculator = MFUConfig(peak_tflops=2500.0).build(model=_vl_model(_llm_config(), mtp_depths=mtp_depths))
    input_ids = torch.zeros((8, SEQ_LEN), dtype=torch.long)

    flops = calculator.get_flops(input_ids)

    assert flops == 8 * SEQ_LEN * (BACKBONE_FLOPS_PER_TOKEN + mtp_depths * MTP_DEPTH_FLOPS_PER_TOKEN)


def test_mtp_depths_are_those_of_the_model_not_of_the_checkpoint_config():
    config = _llm_config()
    assert config.num_nextn_predict_layers == 1

    two_depths = _language_model(config, mtp_depths=2).model_flops(gbs=1, seq_len=SEQ_LEN)
    one_depth = _language_model(config, mtp_depths=1).model_flops(gbs=1, seq_len=SEQ_LEN)

    assert two_depths - one_depth == SEQ_LEN * MTP_DEPTH_FLOPS_PER_TOKEN


def test_two_mtp_depths_cost_the_same_with_and_without_a_shared_block():
    config = _llm_config()

    shared = _language_model(config, mtp_depths=2, shared_mtp_block=True).model_flops(gbs=1, seq_len=SEQ_LEN)
    separate = _language_model(config, mtp_depths=2, shared_mtp_block=False).model_flops(gbs=1, seq_len=SEQ_LEN)

    assert shared == separate == SEQ_LEN * (BACKBONE_FLOPS_PER_TOKEN + 2 * MTP_DEPTH_FLOPS_PER_TOKEN)


def test_one_routed_expert_fewer_per_token_lowers_the_count_by_that_expert():
    """A wrong expert top-k does not go unnoticed: each routed expert is 6 x 2 x 1024 x 2688 FLOPs per token."""
    model = _language_model(_llm_config(num_experts_per_tok=TOP_K - 1), mtp_depths=2)

    flops = model.model_flops(gbs=1, seq_len=SEQ_LEN)

    expected = SEQ_LEN * (BACKBONE_FLOPS_PER_TOKEN + 2 * MTP_DEPTH_FLOPS_PER_TOKEN)
    assert expected - flops == SEQ_LEN * 6 * EXPERT * (NUM_MOE + 2)
