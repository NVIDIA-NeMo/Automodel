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

"""CPU regression tests for the Step5 conditional model's loss controls."""

import inspect
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5.configuration import Step5Config, Step5VLConfig
from nemo_automodel.components.models.step5.model import Step5ForCausalLM, Step5ForConditionalGeneration
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep
from nemo_automodel.recipes.llm.train_ft import _maybe_downgrade_loss_fn

pytestmark = pytest.mark.cpu_only


def test_conditional_explicit_capability_and_defaults():
    parameters = inspect.signature(Step5ForConditionalGeneration.forward).parameters
    assert parameters["logits_to_keep"].default == 0
    assert parameters["output_hidden_states"].default is False
    assert parameters["compute_logits"].default is True
    probe = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
    nn.Module.__init__(probe)
    assert _supports_logits_to_keep(probe)
    assert not probe.ModelCapabilities.supports_pp  # Not new multimodal PP support.


@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"logits_to_keep": 2},
        {"output_hidden_states": True},
        {"compute_logits": False, "output_hidden_states": True},
    ],
)
def test_conditional_forwards_loss_controls(monkeypatch, controls):
    model = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
    nn.Module.__init__(model)
    sentinel = object()
    parent = Mock(return_value=sentinel)
    monkeypatch.setattr(Step5ForCausalLM, "forward", parent)
    tokens, positions = object(), object()
    assert model.forward(tokens, position_ids=positions, **controls) is sentinel
    parent.assert_called_once_with(
        tokens,
        inputs_embeds=None,
        logits_to_keep=controls.get("logits_to_keep", 0),
        output_hidden_states=controls.get("output_hidden_states", False),
        compute_logits=controls.get("compute_logits", True),
        position_ids=positions,
    )


@pytest.mark.parametrize("pp_enabled", [False, True])
def test_conditional_retains_configured_fused_loss(pp_enabled):
    probe = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
    nn.Module.__init__(probe)
    loss = FusedLinearCrossEntropy()
    assert _maybe_downgrade_loss_fn(loss, probe, pp_enabled) is loss
    masked = MaskedCrossEntropy()
    assert _maybe_downgrade_loss_fn(masked, probe, pp_enabled) is masked


def test_kwargs_alone_does_not_advertise_loss_controls():
    class Unsupported(nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return input_ids

    model = Unsupported()
    assert not _supports_logits_to_keep(model)
    assert isinstance(_maybe_downgrade_loss_fn(FusedLinearCrossEntropy(), model, True), MaskedCrossEntropy)


@pytest.fixture
def conditional_model():
    text = Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[],
        dtype="float32",
        num_nextn_predict_layers=0,
        sparse_config={
            "proxy_dim": 16,
            "sparse_indexer_rope_dim": 4,
            "sparse_indexer_num_heads": 4,
            "num_provider_groups": 2,
            "topk": 2,
            "region_block_size": 2,
        },
    )
    config = Step5VLConfig(
        text_config=text,
        vision_config={"width": 16, "layers": 1, "heads": 2, "image_size": 28, "patch_size": 14, "mlp_ratio": 2},
        image_token_id=63,
    )
    torch.manual_seed(137)
    return Step5ForConditionalGeneration(
        config, backend=BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm")
    ).eval()


def test_conditional_public_outputs_keep_default_logits(conditional_model):
    model = conditional_model
    tokens = torch.tensor([[0, 1, 2, 3]])
    default = model(tokens)
    explicit = model(tokens, logits_to_keep=0, output_hidden_states=False)
    retained = model(tokens, logits_to_keep=2, output_hidden_states=True)
    assert default.logits.shape == (1, 4, 64)
    assert default.hidden_states is None
    torch.testing.assert_close(default.logits, explicit.logits)
    torch.testing.assert_close(retained.logits, default.logits[:, -2:, :])
    assert retained.hidden_states.shape == (1, 4, 32)
    skipped = model(tokens, output_hidden_states=True, compute_logits=False)
    assert skipped.logits.shape == (1, 4, 0)
    torch.testing.assert_close(skipped.hidden_states, retained.hidden_states)
    indices = torch.tensor([0, 2])
    indexed = model(tokens, logits_to_keep=indices)
    torch.testing.assert_close(indexed.logits, default.logits[:, indices, :])


def test_conditional_native_pipeline_hidden_output_skips_head(conditional_model, monkeypatch):
    model = conditional_model
    model._pp_return_hidden_states = True
    head = Mock(side_effect=AssertionError("native pipeline hidden-state path must skip lm_head"))
    monkeypatch.setattr(model.lm_head, "forward", head)
    hidden = model(torch.tensor([[0, 1, 2, 3]]))
    assert hidden.shape == (1, 4, 32)
    head.assert_not_called()
    hidden.square().mean().backward()
    assert model.model.embed_tokens.weight.grad is not None
