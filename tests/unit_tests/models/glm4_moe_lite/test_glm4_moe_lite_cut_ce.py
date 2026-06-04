# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tiny-config tests for memory-efficient fused cross-entropy support.

These verify that ``Glm4MoeLiteForCausalLM.forward`` exposes the contract the
training recipe (``nemo_automodel/recipes/llm/train_ft.py``) relies on when the
loss is ``FusedLinearCrossEntropy``:

- ``forward`` accepts a ``logits_to_keep`` parameter (checked by
  ``_supports_logits_to_keep``), and
- ``model(logits_to_keep=1, output_hidden_states=True)`` returns an output
  whose ``hidden_states`` field is populated with the *final* hidden states
  spanning the full sequence, while ``logits`` cover only the last token.

Otherwise the recipe silently falls back to MaskedCrossEntropy.
"""

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.glm4_moe_lite.model import Glm4MoeLiteForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cut-CE path requires CUDA")


@dataclass
class TinyGlm4MoeLiteConfig:
    """Minimal CPU-friendly config combining GLM4 MoE with MLA-specific fields."""

    # Basic model config
    vocab_size: int = 256
    hidden_size: int = 64
    num_hidden_layers: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 256
    rope_theta: float = 5000.0
    # float32 so the model runs on CPU without bf16 matmul gaps.
    torch_dtype: str = "float32"

    rope_parameters: dict = None

    # MLA config
    num_attention_heads: int = 4
    q_lora_rank: int = 16
    kv_lora_rank: int = 8
    qk_nope_head_dim: int = 8
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    rope_scaling: dict = None

    # MoE config
    intermediate_size: int = 128
    moe_intermediate_size: int = 64
    n_routed_experts: int = 4
    n_shared_experts: int = 0
    num_experts_per_tok: int = 2
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True

    # Layer types (first layer dense, rest sparse)
    mlp_layer_types: list = None

    def __post_init__(self):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def model(backend_config):
    cfg = TinyGlm4MoeLiteConfig()
    model = Glm4MoeLiteForCausalLM(cfg, backend=backend_config)
    # Populate weights/buffers (incl. the MoE gate correction bias) so logits are
    # finite; the bare constructor leaves some buffers uninitialized.
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_supports_logits_to_keep(model):
    # train_ft.py gates FusedLinearCrossEntropy on this being True.
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_outputs_hidden_states_and_last_token_logits(model):
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # (a) hidden states must be present: the recipe does `"hidden_states" in out`
    # (ModelOutput __contains__ is True only for populated/non-None fields).
    assert ("hidden_states" in out) or out.hidden_states is not None

    # (b) the final hidden states span the FULL sequence ...
    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    assert hidden_states.shape == (batch, seq_len, model.config.hidden_size)

    # ... and the logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, model.config.vocab_size)


def test_default_forward_returns_full_length_logits(model):
    batch, seq_len = 2, 6
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        out = model(input_ids)

    # (c) default forward still yields full-length logits.
    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq_len, model.config.vocab_size)

    # Default keeps prior behavior: hidden states are not emitted.
    assert ("hidden_states" not in out) or out.hidden_states is None


def test_logits_to_keep_zero_matches_full_projection(model):
    batch, seq_len = 1, 5
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))

    with torch.no_grad():
        explicit_zero = model(input_ids, logits_to_keep=0)
        default = model(input_ids)

    # logits_to_keep=0 must be identical to the default (all positions, no slice).
    torch.testing.assert_close(explicit_zero.logits, default.logits)
