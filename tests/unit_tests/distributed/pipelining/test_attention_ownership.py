# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stages must build the same attention masks as the unsplit model."""

from copy import deepcopy

import pytest
import torch

from nemo_automodel.components.distributed.pipelining.hf_utils import create_pipeline_forward_inner

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("first_stage", [False, True])
@pytest.mark.parametrize("mask_kind", ["causal", "padding", "packed"])
def test_stage_owned_masks_match_model_outputs_and_gradients(
    architecture: str, first_stage: bool, mask_kind: str
) -> None:
    """Compare full/sliding attention, padding, and document isolation without a collate wrapper."""
    if architecture == "llama":
        from transformers import LlamaConfig, LlamaModel

        config_cls, model_cls = LlamaConfig, LlamaModel
        extra = {}
    else:
        from transformers import Qwen2Config, Qwen2Model

        config_cls, model_cls = Qwen2Config, Qwen2Model
        extra = {
            "use_sliding_window": True,
            "sliding_window": 2,
            "layer_types": ["sliding_attention", "full_attention"],
        }
    config = config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_cache=False,
        **extra,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    reference = model_cls(config).eval()
    stage = deepcopy(reference)
    if not first_stage:
        stage.embed_tokens = None
        # A later stage retains global ModuleDict keys, even when its first
        # remaining layer is not layer zero in the model's attention plan.
        stage.layers = torch.nn.ModuleDict({"1": stage.layers[1]})
        reference.layers = torch.nn.ModuleList([reference.layers[1]])
        reference.config.num_hidden_layers = 1
        if architecture == "qwen2":
            reference.config.layer_types = ["full_attention"]
            reference.has_sliding_layers = False
    forward = create_pipeline_forward_inner("PipelineStage")
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])

    # Change the padding/document boundaries without changing shape: an unsafe
    # causal-mask cache would silently reuse the first batch's exclusions.
    for boundary in (2, 4):
        attention_mask = None
        if mask_kind == "padding":
            attention_mask = torch.ones_like(input_ids)
            attention_mask[0, :boundary] = 0
            attention_mask[1, boundary:] = 0
        elif mask_kind == "packed":
            documents = torch.ones_like(input_ids)
            documents[:, boundary:] = 2
            allowed = (documents[:, :, None] == documents[:, None, :]).tril().unsqueeze(1)
            attention_mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(
                ~allowed, torch.finfo(torch.float32).min
            )
        inputs = {"input_ids": input_ids}
        if not first_stage:
            inputs = {"inputs_embeds": reference.embed_tokens(input_ids).detach()}
        wanted = reference(**inputs, attention_mask=attention_mask).last_hidden_state
        actual = forward(stage, **inputs, attention_mask=attention_mask)
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        wanted.square().sum().backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in stage.named_parameters():
            reference_name = name if first_stage else name.replace("layers.1.", "layers.0.", 1)
            expected = reference_parameters[reference_name].grad
            if expected is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-5)
        stage.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
