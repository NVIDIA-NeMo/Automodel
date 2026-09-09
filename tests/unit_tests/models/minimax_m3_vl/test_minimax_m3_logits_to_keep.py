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

"""``logits_to_keep`` on the M3 VLM forward, the contract the fused losses need.

The recipe decides whether a memory-efficient loss is usable by inspecting the
forward signature (``_supports_logits_to_keep``), so absorbing the kwarg into
``**kwargs`` silently downgrades the configured ``loss_fn`` to a default
``MaskedCrossEntropy`` -- along with its ``fp32_upcast`` setting -- and
materialises the ``[tokens, vocab_size]`` logits tensor.
"""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

from .conftest import (
    IMAGE_TOKEN_INDEX,
    SPARSE_ATTENTION_CONFIG,
    TINY_CFG,
    VIDEO_TOKEN_INDEX,
    VISION_CONFIG,
)


@pytest.fixture
def mtp_vlm_model(backend):
    """``vlm_model`` with one MTP module, which the fused-loss path must reject.

    The shared ``mtp_model`` fixture builds ``MiniMaxM3SparseForCausalLM``; the
    ``logits_to_keep`` branch lives on the conditional-generation class.
    """
    cfg = MiniMaxM3VLConfig(
        vision_config=dict(VISION_CONFIG),
        text_config={
            **TINY_CFG,
            "torch_dtype": "float32",
            "sparse_attention_config": dict(SPARSE_ATTENTION_CONFIG),
            "num_mtp_modules": 1,
        },
        image_token_index=IMAGE_TOKEN_INDEX,
        video_token_index=VIDEO_TOKEN_INDEX,
        projector_hidden_size=TINY_CFG["hidden_size"],
    )
    m = MiniMaxM3SparseForConditionalGeneration(cfg, backend=backend)
    m.initialize_weights(dtype=torch.float32)
    return m


def test_recipe_detects_logits_to_keep_support(vlm_model):
    """The probe the trainer actually runs before keeping a fused loss."""
    assert _supports_logits_to_keep(vlm_model)


def test_logits_to_keep_returns_hidden_states_and_skips_lm_head(vlm_model):
    hidden_size = vlm_model.config.text_config.hidden_size
    vocab = vlm_model.config.text_config.vocab_size
    ids = torch.randint(2, 99, (1, 16))

    with torch.no_grad():
        out = vlm_model(ids, logits_to_keep=1)

    # FusedLinearCrossEntropy reads this via `"hidden_states" not in out` then
    # get_final_hidden_states(out), so the mapping matters as much as the tensor.
    assert "hidden_states" in out
    hidden = out["hidden_states"]
    assert hidden.shape == (1, 16, hidden_size)
    assert hidden.shape[-1] != vocab, "lm_head was applied; the logits tensor was materialised"
    assert torch.isfinite(hidden).all()


def test_without_logits_to_keep_still_returns_logits(vlm_model):
    """Default path is unchanged: callers that want logits keep getting them."""
    vocab = vlm_model.config.text_config.vocab_size
    ids = torch.randint(2, 99, (1, 16))

    with torch.no_grad():
        logits = vlm_model(ids)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 16, vocab)


def test_logits_to_keep_with_mtp_raises(mtp_vlm_model):
    """MTP needs full logits, so the fused path must fail loudly rather than
    hand back a dict ``mtp_logits`` cannot consume."""
    mtp_vlm_model.train()
    ids = torch.randint(2, 99, (1, 16))

    with pytest.raises(NotImplementedError, match="num_mtp_modules"):
        mtp_vlm_model(ids, logits_to_keep=1)


def test_mtp_without_logits_to_keep_is_unaffected(mtp_vlm_model):
    """The guard is scoped to the fused path; MTP training itself still runs."""
    mtp_vlm_model.train()
    ids = torch.randint(2, 99, (1, 16))

    out = mtp_vlm_model(ids)

    assert out.logits.shape == (1, 16, mtp_vlm_model.config.text_config.vocab_size)
    assert out.mtp_per_depth_logits is not None
