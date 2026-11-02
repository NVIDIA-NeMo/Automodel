# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding must suppress expert dispatch without changing supervised token math."""

import copy

import pytest
import torch

from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config


def _padding_model() -> MuseSpark1xOmniForCausalLM:
    """Build two tiny decoder layers for padding and gradient checks."""
    config = tiny_muse_spark_config()
    config.num_hidden_layers = 2
    model = MuseSpark1xOmniForCausalLM(config, backend=tiny_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


@pytest.mark.runtime_budget(
    10, reason="two real model forward/backward passes compare all gradients; cold CPU initialization measured 6.3s"
)
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_preserves_valid_logits_and_parameter_gradients(padding_side: str) -> None:
    """Skipping padded expert rows preserves valid logits and every trainable gradient."""
    torch.manual_seed(29)
    model = _padding_model().train()
    reference = copy.deepcopy(model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    valid = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, False, False]])
    if padding_side == "left":
        valid = valid.flip(-1)
    dispatched_rows = []
    handles = []
    for layer in model.model.layers:
        handles.append(
            layer.feed_forward.post_expert_norm.register_forward_pre_hook(
                lambda _module, args: dispatched_rows.append(args[0].shape[0])
            )
        )
    try:
        logits = model(input_ids=input_ids, attention_mask=valid).logits
    finally:
        for handle in handles:
            handle.remove()
    # Explicitly retain the previous implementation's routing of padding.
    expected = reference(input_ids=input_ids, attention_mask=valid, padding_mask=torch.zeros_like(valid)).logits
    torch.testing.assert_close(logits[valid], expected[valid], rtol=1e-5, atol=1e-7)
    gradient = torch.randn_like(logits[valid])
    logits[valid].backward(gradient)
    expected[valid].backward(gradient)
    for name, parameter in model.named_parameters():
        expected_grad = reference.get_parameter(name).grad
        if expected_grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-5, atol=1e-7, msg=name)
    topk = model.config.moe["topk"]
    assert dispatched_rows == [int(valid.sum()) * topk] * model.config.num_hidden_layers


@pytest.mark.parametrize("mask_kind", ["explicit", "absent", "causal_4d", "all_padding"])
def test_expert_padding_mask_contract(mask_kind: str) -> None:
    """Explicit padding wins; absent/causal masks do not become token padding."""
    torch.manual_seed(31)
    model = _padding_model().train()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    kwargs = {}
    expected_tokens = 4
    if mask_kind == "explicit":
        kwargs["attention_mask"] = torch.ones_like(input_ids)
        kwargs["padding_mask"] = torch.tensor([[False, False, True, True]])
        expected_tokens = 2
    elif mask_kind == "causal_4d":
        kwargs["attention_mask"] = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    elif mask_kind == "all_padding":
        kwargs["attention_mask"] = torch.zeros_like(input_ids)
        expected_tokens = 0
    masks = []
    handles = [
        layer.feed_forward.experts.register_forward_pre_hook(
            lambda _module, args: masks.append(args[1].detach().clone())
        )
        for layer in model.model.layers
    ]
    try:
        logits = model(input_ids=input_ids, **kwargs).logits
        logits.square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(logits).all()
    assert [int(mask.sum()) for mask in masks] == [expected_tokens] * model.config.num_hidden_layers


def test_cp_preparation_preserves_padding_before_attention_mask_is_removed() -> None:
    """CP preparation emits a global padding mask for the existing auxiliary sharder."""
    model = _padding_model()
    valid = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]])
    batch = {"input_ids": torch.ones_like(valid), "labels": torch.ones_like(valid), "attention_mask": valid}
    original = valid.clone()
    prepared = model.prepare_model_inputs_for_cp(batch, num_chunks=2)
    torch.testing.assert_close(prepared["padding_mask"], ~valid.bool())
    torch.testing.assert_close(batch["attention_mask"], original)
    assert "padding_mask" not in batch
    explicit = torch.zeros_like(valid, dtype=torch.bool)
    batch["padding_mask"] = explicit
    prepared = model.prepare_model_inputs_for_cp(batch, num_chunks=2)
    assert prepared.get("padding_mask", explicit) is explicit


def test_received_pipeline_residual_uses_its_microbatch_padding() -> None:
    """A later PP stage derives padding from its own local microbatch metadata."""
    model = _padding_model().eval()
    model.model.embed_tokens = None
    model.lm_head = None
    residual = torch.randn(1, 6, model.config.hidden_size)
    valid = torch.tensor([[True, True, False, False, False, False]])
    with torch.no_grad():
        output = model(input_ids=residual, attention_mask=valid).logits
        reference = model(input_ids=residual, attention_mask=valid, padding_mask=~valid).logits
    torch.testing.assert_close(output, reference)
