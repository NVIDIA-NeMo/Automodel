# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU arithmetic checks for packed MLA layouts, outputs and gradients."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata
from nemo_automodel.components.models.kimi_linear import flash_attention as flash
from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel.components.models.kimi_linear.cp import (
    KimiPackedContext,
    build_document_causal_mask,
)
from nemo_automodel.components.models.kimi_linear.model import (
    KimiLinear48BForCausalLM,
    KimiMLAAttention,
)
from nemo_automodel.shared.import_utils import UnavailableError
from tests.unit_tests.models.kimi_linear.test_model import (
    _backend_config,
    _require_fla,
    _tiny_kimi_config,
)


def _reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, ids: torch.Tensor) -> torch.Tensor:
    """Compute dense per-document causal attention.

    Args:
        query: Tensor of shape [batch, heads, sequence, qk_dim].
        key: Tensor of shape [batch, heads, sequence, qk_dim].
        value: Tensor of shape [batch, heads, sequence, value_dim].
        ids: Integer tensor of shape [batch, sequence] with zero padding.

    Returns:
        Tensor of shape [batch, heads, sequence, value_dim]; padding outputs are zero.
    """
    mask = build_document_causal_mask(ids, ids, q_global_start=0, dtype=query.dtype)
    probabilities = ((query @ key.transpose(-1, -2)) * query.shape[-1] ** -0.5 + mask).softmax(-1)
    return (probabilities @ value) * (ids > 0)[:, None, :, None]


def _fake_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, None]:
    """CPU replacement for only the external CUDA kernel.

    Args:
        query: Tensor of shape [valid_tokens, heads, qk_dim], grouped in cu_seqlens_q order.
        key: Tensor of shape [valid_tokens, heads, qk_dim].
        value: Tensor of shape [valid_tokens, heads, value_dim], with qk_dim >= value_dim
            after the wrapper pads narrower Q/K.
        cu_seqlens_q: Integer tensor of shape [documents + 1].
        cu_seqlens_k: Integer tensor of shape [documents + 1].
        max_seqlen_q: Maximum document length.
        max_seqlen_k: Maximum document length.
        softmax_scale: Original QK scale, independent of head_dim.
        causal: Must be true.

    Returns:
        Pair of output tensor [valid_tokens, heads, value_dim] and unused LSE (None).
    """
    assert causal
    assert torch.equal(cu_seqlens_q, cu_seqlens_k) and max_seqlen_q == max_seqlen_k
    outputs = []
    for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        q, k, v = (tensor[start:end].transpose(0, 1) for tensor in (query, key, value))
        scores = (q @ k.transpose(-1, -2)) * softmax_scale
        future = torch.ones((end - start, end - start), dtype=torch.bool).triu(1)
        outputs.append((scores.masked_fill(future, -torch.inf).softmax(-1) @ v).transpose(0, 1))
    return torch.cat(outputs), None


@pytest.fixture
def fake_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flash, "_load_fa4_varlen", lambda: _fake_varlen)


@pytest.mark.parametrize("qk_dim,value_dim", [(16, 8), (8, 16), (192, 128)])
@pytest.mark.parametrize(
    "ids",
    [
        [[1, 1, 2, 2, 0], [0, 1, 1, 2, 0]],
        [[2, 1, 2, 0, 1], [1, 0, 1, 0, 0]],
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    ],
)
def test_outputs_and_all_qkv_gradients_match_dense_reference(fake_flash, ids, qk_dim, value_dim) -> None:
    torch.manual_seed(3)
    ids = torch.tensor(ids, dtype=torch.int32)
    q = torch.randn(2, 2, 5, qk_dim, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(2, 2, 5, value_dim, requires_grad=True)
    expected = _reference(q, k, v, ids=ids)
    actual = flash.packed_flash_attention(q, k, v, layout=flash.build_flash_layout(ids), scale=qk_dim**-0.5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    for left, right in zip(actual_grads, expected_grads):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_all_padding_fp16_outputs_and_qkv_gradients_are_finite_zeros(fake_flash) -> None:
    q, k, v = (torch.ones(1, 1, 1024, 64, dtype=torch.float16, requires_grad=True) for _ in range(3))
    layout = flash.build_flash_layout(torch.zeros(1, 1024, dtype=torch.int32))

    actual = flash.packed_flash_attention(q, k, v, layout=layout, scale=64**-0.5)

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0
    actual.backward(torch.randn_like(actual))
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


@pytest.mark.parametrize("mask_kind", ["indexed", "binary", "padding_only", "none"])
def test_backbone_outputs_and_parameter_gradients_without_dense_mask(fake_flash, monkeypatch, mask_kind) -> None:
    torch.manual_seed(9)
    config = _tiny_kimi_config()
    eager = KimiLinear48BForCausalLM(config, backend=_backend_config())
    eager.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    fast = KimiLinear48BForCausalLM(config, backend=replace(_backend_config(), attn="fa4"))
    fast.load_state_dict(eager.state_dict(), strict=True)
    ids = torch.tensor([[1, 1, 2, 2, 0], [0, 1, 1, 1, 0]], dtype=torch.int32)
    if mask_kind in ("binary", "padding_only"):
        ids = (ids > 0).int()
    elif mask_kind == "none":
        ids = torch.ones_like(ids)
    tokens = torch.tensor([[3, 4, 5, 6, 0], [0, 7, 8, 9, 0]])
    expected = eager(input_ids=tokens, attention_mask=ids).logits

    def reject_dense_mask(
        inputs_embeds: torch.Tensor,
        packed_context: KimiPackedContext | None,
        *,
        dtype: torch.dtype,
    ) -> None:
        """Reject dense-mask construction.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            packed_context: Optional document map [batch, sequence].
            dtype: Floating mask dtype.
        """
        raise AssertionError("Flash MLA must not allocate a sequence-square mask")

    monkeypatch.setattr(
        "nemo_automodel.components.models.kimi_linear.model._make_causal_mask",
        reject_dense_mask,
    )
    options = (
        {"padding_mask": ids == 0}
        if mask_kind == "padding_only"
        else ({"attention_mask": ids} if mask_kind != "none" else {})
    )
    actual = fast(input_ids=tokens, **options).logits
    valid = ids > 0
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-6, rtol=1e-5)
    expected[valid].square().sum().backward()
    actual[valid].square().sum().backward()
    for (name, left), (other, right) in zip(eager.named_parameters(), fast.named_parameters()):
        assert name == other
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-6, rtol=1e-4, msg=name)


def test_layer_requires_a_prepared_flash_layout() -> None:
    config = _tiny_kimi_config()
    layer = KimiMLAAttention(config, 0, backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="requires flash_layout"):
        layer(torch.randn(1, 4, config.hidden_size))


def test_cp_keeps_its_existing_attention_path(fake_flash, monkeypatch) -> None:
    config = _tiny_kimi_config()
    layer = KimiMLAAttention(config, 0, backend=replace(_backend_config(), attn="fa4"))
    context = KimiPackedContext(torch.ones(1, 4, dtype=torch.int32), cp_size=2)
    calls = []

    def cp_forward(hidden_states: torch.Tensor, packed_context: KimiPackedContext) -> torch.Tensor:
        """Stand in for the unchanged CP kernel.

        Args:
            hidden_states: Tensor of shape [batch, local_sequence, hidden].
            packed_context: Global document map [batch, sequence].

        Returns:
            The input tensor [batch, local_sequence, hidden], aliased unchanged.
        """
        calls.append(packed_context)
        return hidden_states

    monkeypatch.setattr(layer, "_forward_with_cp", cp_forward)
    x = torch.randn(1, 2, config.hidden_size)
    assert layer(x, packed_context=context) is x
    assert calls == [context]


def test_missing_optional_dependency_and_explicit_dense_mask(monkeypatch) -> None:
    flash._load_fa4_varlen.cache_clear()
    monkeypatch.setattr(flash, "safe_import_from", lambda *args, **kwargs: (False, None))
    q = torch.randn(1, 1, 2, 8)
    layout = flash.build_flash_layout(torch.ones(1, 2, dtype=torch.int32))
    with pytest.raises(UnavailableError, match="flash_attn.cute"):
        flash.packed_flash_attention(q, q, q, layout=layout, scale=8**-0.5)
    config = _tiny_kimi_config()
    layer = KimiMLAAttention(config, 0, backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="additive attention_mask"):
        layer(
            torch.randn(1, 2, config.hidden_size),
            attention_mask=torch.zeros(1, 1, 2, 2),
        )


def test_checkpoint_config_does_not_persist_the_execution_backend(monkeypatch) -> None:
    config = _tiny_kimi_config()
    fast = KimiLinear48BForCausalLM(config, backend=replace(_backend_config(), attn="fa4"))
    fast.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    saved = fast.config.to_dict()
    restored = KimiLinear48BForCausalLM(KimiLinear48BConfig.from_dict(saved), backend=_backend_config())
    assert fast.backend.attn == "fa4"
    assert restored.backend.attn == "eager"
    restored.load_state_dict(fast.state_dict(), strict=True)

    def reject_flash():
        raise AssertionError("A restored checkpoint must run without the optional FA4 kernel")

    monkeypatch.setattr(flash, "_load_fa4_varlen", reject_flash)
    assert torch.isfinite(restored(input_ids=torch.tensor([[3, 4]])).logits).all()


@pytest.mark.parametrize("use_kda", [False, True])
def test_thd_document_spans_respect_physical_padding(fake_flash, monkeypatch, use_kda) -> None:
    if use_kda:
        _require_fla()
    torch.manual_seed(20)
    config = _tiny_kimi_config(use_kda=use_kda)
    reference = KimiLinear48BForCausalLM(config, backend=_backend_config())
    reference.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    fast = KimiLinear48BForCausalLM(config, backend=replace(_backend_config(), attn="fa4"))
    fast.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.tensor([[3, 4, 0, 5, 0, 0]])
    ids = torch.tensor([[1, 1, 0, 2, 0, 0]], dtype=torch.int32)
    reference_metadata = build_packed_sequence_metadata(ids) if use_kda else {}
    if use_kda:
        # Isolate packing/gradient behavior from the unchanged GPU-only FLA core.
        def recurrent_reference(hidden_states: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
            """Compute segmented prefix sums as a deterministic recurrent oracle.

            Args:
                hidden_states: Tensor of shape [1, valid_tokens, hidden].
                cu_seqlens: Logical boundaries of shape [documents + 1].

            Returns:
                Prefix sums of shape [1, valid_tokens, hidden], reset at each document.
            """
            assert hidden_states.shape[1] == 3
            assert cu_seqlens.tolist() == [0, 2, 3]
            cuts = cu_seqlens.tolist()
            return torch.cat([hidden_states[:, start:end].cumsum(1) for start, end in zip(cuts, cuts[1:])], dim=1)

        for model in (reference, fast):
            for layer in model.model.layers.values():
                if layer.is_linear_attn:
                    monkeypatch.setattr(layer.self_attn, "_kda_core", recurrent_reference)
    expected = reference(input_ids=tokens, attention_mask=ids, **reference_metadata).logits
    physical_metadata = {"packed_token_indices": torch.tensor([0, 1, 3])} if use_kda else {}
    actual = fast(
        input_ids=tokens,
        padding_mask=ids == 0,
        qkv_format="thd",
        position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
        cu_seqlens=torch.tensor([[0, 2, 3]], dtype=torch.int32),
        cu_seqlens_padded=torch.tensor([[0, 3, 6]], dtype=torch.int32),
        **physical_metadata,
    ).logits
    valid = ids > 0
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-6, rtol=1e-5)
    expected[valid].square().sum().backward()
    actual[valid].square().sum().backward()
    for (name, a), (other, b) in zip(fast.named_parameters(), reference.named_parameters()):
        assert name == other
        if b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4, msg=name)


def test_backbone_rejects_dense_mask_in_flash_mode() -> None:
    config = _tiny_kimi_config()
    model = KimiLinear48BForCausalLM(config, backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="binary/document attention_mask"):
        model(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.zeros(1, 1, 2, 2))


def test_fa4_rejects_attention_dropout(fake_flash) -> None:
    q = torch.randn(1, 1, 2, 8)
    layout = flash.build_flash_layout(torch.ones(1, 2, dtype=torch.int32))
    with pytest.raises(ValueError, match="attention_dropout=0"):
        flash.packed_flash_attention(q, q, q, layout=layout, scale=8**-0.5, dropout_p=0.1)


def test_nonempty_fa4_rejects_deterministic_algorithms(fake_flash) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        q = torch.ones(1, 1, 2, 8, requires_grad=True)
        layout = flash.build_flash_layout(torch.ones(1, 2, dtype=torch.int32))
        with pytest.raises(RuntimeError, match="nondeterministic attention backward"):
            flash.packed_flash_attention(q, q, q, layout=layout, scale=8**-0.5)
        empty = flash.packed_flash_attention(
            q, q, q, layout=flash.build_flash_layout(torch.zeros(1, 2, dtype=torch.int32)), scale=8**-0.5
        )
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0
        grad = torch.autograd.grad(empty, q, torch.randn_like(empty))[0]
        assert torch.count_nonzero(grad) == 0
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def test_hybrid_physical_thd_padding_requires_unpadded_indices(fake_flash) -> None:
    _require_fla()
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(use_kda=True), backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="requires logical cu_seqlens and packed_token_indices"):
        model(
            input_ids=torch.tensor([[3, 4, 0, 5, 0, 0]]),
            padding_mask=torch.tensor([[False, False, True, False, True, True]]),
            qkv_format="thd",
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]]),
            cu_seqlens=torch.tensor([[0, 2, 3]], dtype=torch.int32),
            cu_seqlens_padded=torch.tensor([[0, 3, 6]], dtype=torch.int32),
        )


def test_flat_thd_boundaries_reject_a_mismatched_batch_layout(fake_flash) -> None:
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="document map must match"):
        model(
            input_ids=torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
