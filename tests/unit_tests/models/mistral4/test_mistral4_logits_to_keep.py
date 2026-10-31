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

"""``logits_to_keep`` on the Mistral 4 VLM forward, the contract the fused losses need.

The recipe decides whether a memory-efficient loss is usable by inspecting the
forward signature (``_supports_logits_to_keep``), so absorbing the kwarg into
``**kwargs`` silently downgrades the configured ``loss_fn`` to a default
``MaskedCrossEntropy`` -- along with its ``fp32_upcast`` setting -- and
materialises the ``[tokens, vocab_size]`` logits tensor.

These live outside ``test_mistral4_model.py`` because that module is gated on
``torch.cuda.is_available()``; the contract here is CPU-checkable.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.model import _HF_MISTRAL3_AVAILABLE
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not _HF_MISTRAL3_AVAILABLE, reason="transformers mistral3 not available")

HIDDEN_SIZE = 64
VOCAB_SIZE = 256


@pytest.fixture
def backend():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def multimodal_config():
    """Small Mistral3Config wrapping a Mistral4Config text config."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from nemo_automodel.components.models.mistral4.configuration import Mistral4Config

    if "mistral4" not in CONFIG_MAPPING:
        AutoConfig.register("mistral4", Mistral4Config)

    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    text_config = Mistral4Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
    )
    return Mistral3Config(
        text_config=text_config.to_dict(),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 4,
        },
        image_token_index=10,
        spatial_merge_size=2,
    )


@pytest.fixture
def model(multimodal_config, backend):
    from nemo_automodel.components.models.mistral4.model import Mistral3ForConditionalGeneration

    m = Mistral3ForConditionalGeneration(multimodal_config, backend=backend)
    # Without this the parameters are uninitialised memory, so the finite-value
    # assertion below passes or fails depending on allocator reuse.
    m.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return m


def test_recipe_detects_logits_to_keep_support(model):
    """The probe the trainer actually runs before keeping a fused loss."""
    assert _supports_logits_to_keep(model)


def test_logits_to_keep_returns_hidden_states_and_skips_lm_head(model):
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1)

    # The recipe raises unless the mapping carries `hidden_states`, and
    # FusedLinearCrossEntropy reads it via get_final_hidden_states(out).
    assert "hidden_states" in out
    hidden = out["hidden_states"]
    assert hidden.shape == (1, 8, HIDDEN_SIZE)
    assert hidden.shape[-1] != VOCAB_SIZE, "lm_head was applied; the logits tensor was materialised"
    assert torch.isfinite(hidden).all()


def test_without_logits_to_keep_still_returns_logits(model):
    """Default path is unchanged: callers that want logits keep getting them."""
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))

    with torch.no_grad():
        logits = model(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 8, VOCAB_SIZE)
