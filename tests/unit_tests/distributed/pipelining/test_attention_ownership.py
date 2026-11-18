# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stages must build the same attention masks as the unsplit model."""

from copy import deepcopy

import pytest
import torch

from nemo_automodel.components.distributed.pipelining.hf_utils import create_pipeline_forward_inner

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("architecture", ["llama", "qwen2"])
@pytest.mark.parametrize("first_stage", [False, True])
@pytest.mark.parametrize("mask_kind", ["causal", "padding", "packed"])
def test_stage_owned_masks_match_model_outputs_and_gradients(
    architecture: str, first_stage: bool, mask_kind: str, backend: str
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
    config._attn_implementation = backend
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


def test_explicit_padding_mask_has_no_scalar_reads() -> None:
    """Mask materialization must not read tensor values back to the host."""
    from types import SimpleNamespace

    from torch.utils._python_dispatch import TorchDispatchMode
    from transformers import LlamaConfig

    from nemo_automodel.components.distributed.pipelining.hf_utils import _build_or_reuse_pp_causal_mask

    class RejectScalarReads(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func != torch.ops.aten._local_scalar_dense.default, "Mask creation reads a tensor scalar"
            return func(*args, **(kwargs or {}))

    config = LlamaConfig()
    config._attn_implementation = "sdpa"
    module = SimpleNamespace(config=config, has_sliding_layers=False)
    padding = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    with RejectScalarReads():
        result = _build_or_reuse_pp_causal_mask(
            module, torch.zeros(2, 4, 8), padding, torch.arange(4), torch.arange(4)[None]
        )["full_attention"]
    expected = torch.ones(4, 4, dtype=torch.bool).tril()[None, None] & padding[:, None, None, :].bool()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("first_stage", [True, False])
def test_reset_positions_do_not_reuse_document_masks(backend: str, first_stage: bool) -> None:
    """Changing implicit document boundaries must match an independent explicit mask."""
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = backend
    torch.manual_seed(42)
    reference = LlamaModel(config).eval()
    stage = deepcopy(reference)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    inputs = {"input_ids": tokens}
    if not first_stage:
        stage.embed_tokens = None
        inputs = {"inputs_embeds": reference.embed_tokens(tokens).detach()}
    forward = create_pipeline_forward_inner("PipelineStage")
    for positions in ([0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1]):
        position_ids = torch.tensor([positions, positions])
        documents = (position_ids == 0).cumsum(-1)
        allowed = (documents[:, :, None] == documents[:, None, :]).tril().unsqueeze(1)
        mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
        expected = reference(**inputs, position_ids=position_ids, attention_mask=mask).last_hidden_state
        actual = forward(stage, **inputs, position_ids=position_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in stage.named_parameters():
            expected_grad = reference_parameters[name].grad
            if expected_grad is not None:
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-4, atol=1e-5)
        stage.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
