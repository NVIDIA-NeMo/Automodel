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

"""CPU tiny-config tests for Step3p5 memory-efficient fused cross-entropy support.

These verify the contract the training recipe relies on to enable
``FusedLinearCrossEntropy`` (cut-CE): ``Step3p5ForCausalLM.forward`` must expose a
``logits_to_keep`` parameter and, when ``output_hidden_states`` is set, carry the
FULL-sequence final hidden states on the returned model output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.model import Step3p5ForCausalLM
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep


@dataclass
class TinyStepConfig:
    """Minimal Step3p5 config that instantiates and runs a forward on CPU."""

    vocab_size: int = 16
    hidden_size: int = 8
    intermediate_size: int = 16
    num_hidden_layers: int = 0
    num_attention_heads: int = 2
    num_attention_groups: int = 1
    max_position_embeddings: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_parameters: dict | None = None
    partial_rotary_factors: list | None = None
    layer_types: list | None = None
    attention_other_setting: dict | None = None
    sliding_window: int | None = None
    use_head_wise_attn_gate: bool = False
    use_rope_layers: list | None = None
    head_dim: int = 4
    attention_bias: bool = False
    torch_dtype: str = "float32"
    moe_layers_enum: tuple = ()
    moe_num_experts: int = 2
    moe_top_k: int = 1
    moe_intermediate_size: int = 4
    moe_router_activation: str = "softmax"
    moe_router_scaling_factor: float = 1.0
    use_moe_router_bias: bool = False
    share_expert_dims: int = 4
    swiglu_limits: list | None = None
    swiglu_limits_shared: list | None = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": self.rope_theta, "rope_type": "default"}
        if self.layer_types is None:
            self.layer_types = ["full_attention"]
        if self.attention_other_setting is None:
            self.attention_other_setting = {"num_attention_heads": 2, "num_attention_groups": 1}
        if self.swiglu_limits is None:
            self.swiglu_limits = [None]
        if self.swiglu_limits_shared is None:
            self.swiglu_limits_shared = [None]


def _tiny_backend(**kwargs):
    values = dict(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    values.update(kwargs)
    return BackendConfig(**values)


def _build_model(config: TinyStepConfig) -> Step3p5ForCausalLM:
    model = Step3p5ForCausalLM(config, backend=_tiny_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    return model


def test_supports_logits_to_keep():
    """The recipe gates cut-CE on a ``logits_to_keep`` parameter in forward."""
    model = _build_model(TinyStepConfig())
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_outputs_hidden_states_and_last_token_logits():
    """logits_to_keep=1 must return last-token logits while carrying full-length hidden states."""
    config = TinyStepConfig()
    model = _build_model(config)

    batch, seq = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    out = model(input_ids, logits_to_keep=1, output_hidden_states=True)

    # The recipe checks `"hidden_states" not in out`; ModelOutput excludes None
    # fields from membership, so the field must be populated here.
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    hidden_states = get_final_hidden_states(out)
    assert hidden_states is not None
    # Hidden states span the FULL sequence (not sliced to the kept logits).
    assert hidden_states.shape == (batch, seq, config.hidden_size)

    # Logits correspond to only the last token.
    assert out.logits.shape == (batch, 1, config.vocab_size)


def test_default_forward_yields_full_length_logits():
    """Default call (logits_to_keep=0, no output_hidden_states) keeps prior behavior."""
    config = TinyStepConfig()
    model = _build_model(config)

    batch, seq = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))

    out = model(input_ids)

    # Downstream callers read getattr(out, "logits", out); logits span all positions.
    logits = getattr(out, "logits", out)
    assert logits.shape == (batch, seq, config.vocab_size)
    # No hidden states requested -> not carried (fused-CE path not triggered).
    assert "hidden_states" not in out
