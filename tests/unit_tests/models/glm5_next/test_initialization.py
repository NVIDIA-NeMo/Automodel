# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native checkpoint-owned initialization instead of calling model init directly."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.moe import experts as moe_experts
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


def _construct_model(device: str) -> Glm5NextForConditionalGeneration:
    """Construct the same custom-model class used by the native from-config path."""
    with torch.device(device):
        return Glm5NextForConditionalGeneration(tiny_glm5_next_config(), backend=tiny_backend())


def _assert_initialized(model: Glm5NextForConditionalGeneration) -> None:
    """Check parameter semantics, including deliberate zero-valued state."""
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu", name
        assert torch.isfinite(parameter).all(), name
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0, msg=name)
    text = model.model.language_model
    torch.testing.assert_close(text.embed_tokens.weight[0], torch.zeros(16), rtol=0, atol=0)
    for layer in text.layers.values():
        for hc in (layer.attn_hc, layer.ffn_hc):
            torch.testing.assert_close(hc.base, torch.zeros_like(hc.base), rtol=0, atol=0)
            torch.testing.assert_close(hc.scale, torch.ones_like(hc.scale), rtol=0, atol=0)
            assert hc.fn.std() > 0
    random_weights = (
        text.embed_tokens.weight[1:],
        model.lm_head.weight,
        text.layers["0"].self_attn.q_proj.weight,
        text.layers["0"].mlp.down_proj.weight,
        text.layers["3"].self_attn.q_a_proj.weight,
        text.layers["3"].mlp.gate.weight,
        text.layers["3"].mlp.experts.gate_and_up_projs,
        text.layers["3"].mlp.experts.down_projs,
        text.layers["3"].mlp.shared_experts.down_proj.weight,
    )
    for weight in random_weights:
        assert weight.std() > 0
    gate_bias = text.layers["3"].mlp.gate.e_score_correction_bias
    torch.testing.assert_close(gate_bias, torch.zeros_like(gate_bias), rtol=0, atol=0)


def test_checkpointer_materializes_and_initializes_meta_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native lifecycle must populate from-config meta shells without a checkpoint."""
    # Keep this CPU unit test on CPU even when the test runner has visible GPUs.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(71)
    model = _construct_model("meta")
    assert all(parameter.is_meta for parameter in model.parameters())
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    _assert_initialized(model)


def test_checkpointer_replaces_poisoned_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """A skipped initializer deterministically fails instead of depending on empty memory."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(73)
    model = _construct_model("cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    _assert_initialized(model)


def test_native_initialized_model_has_nonconstant_logits_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check real CPU forward/backward after the same native initialization boundary."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Exercise native initialization and real MoE arithmetic without compiling
    # an unrelated import-time activation wrapper on the cold CPU test path.
    monkeypatch.setattr(moe_experts, "swiglu_clamped_deepep", inspect.unwrap(moe_experts.swiglu_clamped_deepep))
    torch.manual_seed(79)
    model = _construct_model("meta")
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    _assert_initialized(model)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
    assert torch.isfinite(logits).all()
    assert (logits.std(dim=-1) > 0).all()
    labels = torch.tensor([[2, 3, 4, 5, 6, 7]])
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    assert torch.isfinite(loss)
    loss.backward()
    text = model.model.language_model
    differentiable_weights = (
        text.embed_tokens.weight,
        text.layers["0"].self_attn.q_proj.weight,
        text.layers["0"].attn_hc.fn,
        text.layers["3"].self_attn.q_a_proj.weight,
        text.layers["3"].mlp.gate.weight,
        model.lm_head.weight,
    )
    for weight in differentiable_weights:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0
    # Hard top-k selection intentionally does not backpropagate into its indexer.
    assert text.layers["3"].self_attn.indexer.wq_b.weight.grad is None
