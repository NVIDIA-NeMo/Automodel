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

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata
from nemo_automodel.components.models.common.packing import flatten_packed_sequence_metadata
from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel.components.models.kimi_linear.cp import (
    KimiPackedContext,
    build_document_causal_mask,
)
from nemo_automodel.components.models.kimi_linear.model import (
    KimiLinear48BForCausalLM,
    KimiMLAAttention,
)
from tests.unit_tests.models.kimi_linear.test_model import (
    _backend_config,
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
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """CPU replacement for only the external CUDA kernel.

    Args:
        query: Tensor of shape [valid_tokens, heads, qk_dim], grouped in cu_seqlens_q order.
        key: Tensor of shape [valid_tokens, heads, qk_dim].
        value: Tensor of shape [valid_tokens, heads, value_dim].
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
    """Replace only the external CUDA kernels; keep shared preprocessing and restore."""

    def dense(q, k, v, **kwargs):
        lengths = torch.arange(0, (q.shape[0] + 1) * q.shape[1], q.shape[1], dtype=torch.int32)
        output, _ = _fake_varlen(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens_q=lengths,
            cu_seqlens_k=lengths,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=q.shape[1],
            **kwargs,
        )
        return output.reshape(*v.shape), None

    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute",
        SimpleNamespace(
            flash_attn_func=dense,
            flash_attn_varlen_func=_fake_varlen,
        ),
    )


def _metadata(ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Use the dataset's production metadata builder and model-entry normalization."""
    if not ids.bool().any():
        return {}
    metadata = build_packed_sequence_metadata(ids)
    indices, boundaries = flatten_packed_sequence_metadata(
        metadata["packed_token_indices"],
        metadata["cu_seqlens"],
        batch_size=ids.shape[0],
        sequence_length=ids.shape[1],
    )
    return {**metadata, "packed_token_indices": indices, "cu_seqlens": boundaries}


def _shared_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Exercise the same shared attention path as Kimi MLA."""
    _, attention = initialize_attn_module_and_func("fa4", q.shape[1], q.shape[-1], v.shape[-1], q.shape[-1] ** -0.5)
    q, k, v, kwargs = preprocess_args_and_kwargs_for_attn(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        ids,
        "fa4",
        **_metadata(ids),
    )
    return attention(q, k, v, **kwargs).transpose(1, 2)


@pytest.mark.parametrize("qk_dim,value_dim", [(16, 8), (192, 128)])
@pytest.mark.parametrize(
    "ids",
    [
        [[1, 1, 2, 2, 0], [0, 1, 1, 2, 0]],
        [[1, 1, 0, 2, 2], [1, 0, 1, 0, 0]],
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
    actual = _shared_attention(q, k, v, ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    for left, right in zip(actual_grads, expected_grads):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_all_padding_fp16_outputs_and_qkv_gradients_are_finite_zeros(fake_flash) -> None:
    q, k, v = (torch.ones(1, 1, 1024, 64, dtype=torch.float16, requires_grad=True) for _ in range(3))
    actual = _shared_attention(q, k, v, torch.zeros(1, 1024, dtype=torch.int32))

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
    actual = fast(input_ids=tokens, **options, **(_metadata(ids) if mask_kind == "indexed" else {})).logits
    valid = ids > 0
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-6, rtol=1e-5)
    expected[valid].square().sum().backward()
    actual[valid].square().sum().backward()
    for (name, left), (other, right) in zip(eager.named_parameters(), fast.named_parameters()):
        assert name == other
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(right.grad, left.grad, atol=1e-6, rtol=1e-4, msg=name)


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


def test_missing_optional_dependency(monkeypatch) -> None:
    monkeypatch.setattr("nemo_automodel.components.attention.utils.safe_import", lambda *args, **kwargs: (False, None))
    with pytest.raises(ImportError, match="flash_attn.cute"):
        KimiMLAAttention(_tiny_kimi_config(), 0, backend=replace(_backend_config(), attn="fa4"))


def test_checkpoint_config_does_not_persist_the_execution_backend(fake_flash, monkeypatch) -> None:
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

    monkeypatch.setattr("nemo_automodel.components.attention.utils.safe_import", reject_flash)
    assert torch.isfinite(restored(input_ids=torch.tensor([[3, 4]])).logits).all()


def test_backbone_rejects_dense_mask_in_flash_mode(fake_flash) -> None:
    config = _tiny_kimi_config()
    model = KimiLinear48BForCausalLM(config, backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="binary/document attention_mask"):
        model(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.zeros(1, 1, 2, 2))


def test_fa4_rejects_attention_dropout(fake_flash) -> None:
    config = _tiny_kimi_config()
    config.attention_dropout = 0.1
    layer = KimiMLAAttention(config, 0, backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="attention_dropout=0"):
        layer(torch.randn(1, 2, config.hidden_size))


@pytest.mark.parametrize("warn_only", [False, True])
def test_shared_fa4_forwards_deterministic_mode(fake_flash, monkeypatch, warn_only) -> None:
    seen = []

    def kernel(*args, **kwargs):
        seen.append(kwargs["deterministic"])
        return _fake_varlen(*args, **kwargs)

    monkeypatch.setattr(sys.modules["flash_attn.cute"], "flash_attn_varlen_func", kernel)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        q = torch.randn(1, 1, 3, 16, requires_grad=True)
        ids = torch.ones(1, 3, dtype=torch.int32)
        for enabled in (False, True):
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            _shared_attention(q, q, q, ids).sum().backward()
        assert seen == [False, True]
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.parametrize(
    "metadata",
    [
        {"qkv_format": "thd"},
        {"cu_seqlens_padded": torch.tensor([0, 2, 4], dtype=torch.int32)},
    ],
)
def test_shared_fa4_rejects_physical_thd(fake_flash, metadata) -> None:
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="Use NEAT packing"):
        model.model(input_ids=torch.tensor([[3, 4, 5, 6]]), **metadata)


def test_indexed_fa4_requires_dataset_metadata(fake_flash) -> None:
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=replace(_backend_config(), attn="fa4"))
    with pytest.raises(ValueError, match="dataset-provided"):
        model(input_ids=torch.tensor([[3, 4, 5, 6]]), attention_mask=torch.tensor([[1, 1, 2, 2]]))


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("pack_size", [8, 10])
def test_recipe_neat_shared_fa4_document_isolation(fake_flash, split: str, training: bool, pack_size: int) -> None:
    """Exercise dataset metadata, the native model-entry hook, and the shared FA4 path."""
    from tests.unit_tests.models.kimi_linear.test_recipe_packing import _assert_recipe_document_isolation

    _assert_recipe_document_isolation(
        device="cpu", use_kda=False, training=training, pack_size=pack_size, split=split, attn="fa4"
    )
