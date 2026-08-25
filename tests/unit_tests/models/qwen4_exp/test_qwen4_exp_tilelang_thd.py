# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contract and H100 parity tests for direct-THD Qwen4 sparse GQA."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import math
import sys
from collections.abc import Callable, Sequence

import pytest
import torch
from torch import nn

_QUERY_HEADS = 24
_KV_HEADS = 2
_HEAD_DIM = 256
_SCALE = _HEAD_DIM**-0.5
_MAX_TOPK = 2051

Operator = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ForwardBackwardResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _can_run_h100_tilelang() -> bool:
    """Return whether this process has TileLang and an exact H100/SM90 target."""
    if importlib.util.find_spec("tilelang") is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0) and "H100" in torch.cuda.get_device_name()


requires_h100_tilelang = pytest.mark.skipif(
    not _can_run_h100_tilelang(),
    reason="requires TileLang on an H100/SM90 CUDA device",
)


def _cpu_inputs(tokens: int = 3) -> TensorTuple:
    """Create valid CPU tensors for fail-before-launch contract tests.

    Returns:
        Tuple containing BF16 query ``[tokens, 24, 256]``, key/value
        ``[tokens, 2, 256]``, and int32 routes ``[tokens, 3]``.
    """
    query = torch.zeros(tokens, _QUERY_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    key = torch.zeros(tokens, _KV_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    token_ids = torch.zeros(tokens, 3, dtype=torch.int32)
    return query, key, value, token_ids


def test_thd_kernel_modules_import_without_tilelang(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the optional kernel must not load TileLang until first use."""
    from nemo_automodel.shared.import_utils import UnavailableError

    module_names = (
        "nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd",
        "nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_thd_bwd",
        "nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_thd_fwd",
    )
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "tilelang", raising=False)
    monkeypatch.delitem(sys.modules, "tilelang.language", raising=False)

    original_import = builtins.__import__

    def block_tilelang(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "tilelang" or name.startswith("tilelang."):
            raise ImportError("blocked TileLang for lazy-import test")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_tilelang)
    wrapper = importlib.import_module(module_names[0])
    forward = importlib.import_module(module_names[2])

    assert callable(wrapper.tilelang_sparse_gqa_thd_attention)
    assert "tilelang" not in sys.modules
    with pytest.raises(UnavailableError, match="Qwen4-Exp TileLang kernels"):
        forward.sparse_gqa_thd_fwd(_QUERY_HEADS, _KV_HEADS, _HEAD_DIM, 64)


@pytest.mark.parametrize("route_rank", [2, 3])
@pytest.mark.parametrize("route_dtype", [torch.int32, torch.int64])
def test_thd_public_api_normalizes_route_layout_without_narrowing(
    monkeypatch: pytest.MonkeyPatch,
    route_rank: int,
    route_dtype: torch.dtype,
) -> None:
    """The public adapter accepts both route layouts and preserves integer width."""
    module = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd")
    query, key, value, _ = _cpu_inputs()
    topk = _MAX_TOPK if route_rank == 3 and route_dtype == torch.int64 else 7
    token_ids = torch.full((query.shape[0], topk), -1, dtype=route_dtype)
    token_ids[:, 0] = 0
    if route_dtype == torch.int64:
        token_ids[:, -1] = 2**32
    public_ids = token_ids.unsqueeze(1) if route_rank == 3 else token_ids
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def fake_apply(
        received_query: torch.Tensor,
        received_key: torch.Tensor,
        received_value: torch.Tensor,
        received_ids: torch.Tensor,
        received_scale: float,
    ) -> torch.Tensor:
        """Capture the normalized direct-THD call without launching CUDA.

        Args:
            received_query: Tensor of shape ``[tokens, 24, 256]``.
            received_key: Tensor of shape ``[tokens, 2, 256]``.
            received_value: Tensor of shape ``[tokens, 2, 256]``.
            received_ids: Global route tensor of shape ``[tokens, selected]``.
            received_scale: Scalar QK softmax scale.

        Returns:
            The input query tensor of shape ``[tokens, 24, 256]``.
        """
        captured.append((received_query, received_key, received_value, received_ids, received_scale))
        return received_query

    monkeypatch.setattr(module._Qwen4SparseGQAThdAttention, "apply", fake_apply)
    output = module.tilelang_sparse_gqa_thd_attention(query, key, value, public_ids)

    assert output is query
    assert len(captured) == 1
    received_query, received_key, received_value, received_ids, received_scale = captured[0]
    assert received_query is query
    assert received_key is key
    assert received_value is value
    assert received_ids.shape == (query.shape[0], topk)
    assert received_ids.dtype == route_dtype
    assert received_ids.is_contiguous()
    torch.testing.assert_close(received_ids, token_ids, rtol=0, atol=0)
    assert received_scale == _SCALE


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("query_rank", r"query must be \[T, Hq, D\]"),
        ("route_rank", r"rank-3 THD token_ids must be \[T, 1, K\]"),
        ("route_tokens", r"matching T=3"),
        ("empty_width", r"nonempty shape \[T, K\]"),
        ("too_wide", "at most 2051 selected slots"),
    ],
)
def test_thd_public_api_rejects_invalid_shapes(case: str, match: str) -> None:
    module = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd")
    query, key, value, token_ids = _cpu_inputs()
    if case == "query_rank":
        query = query.unsqueeze(0)
    elif case == "route_rank":
        token_ids = token_ids.unsqueeze(1).expand(-1, 2, -1)
    elif case == "route_tokens":
        token_ids = torch.zeros(query.shape[0] + 1, 3, dtype=torch.int32)
    elif case == "empty_width":
        token_ids = torch.empty(query.shape[0], 0, dtype=torch.int32)
    elif case == "too_wide":
        token_ids = torch.zeros(query.shape[0], _MAX_TOPK + 1, dtype=torch.int32)
    else:
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(ValueError, match=match):
        module.tilelang_sparse_gqa_thd_attention(query, key, value, token_ids)


@pytest.mark.parametrize("route_dtype", [torch.bool, torch.int16, torch.float32])
def test_thd_public_api_rejects_non_int32_int64_routes(route_dtype: torch.dtype) -> None:
    module = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd")
    query, key, value, _ = _cpu_inputs()
    token_ids = torch.zeros(query.shape[0], 3, dtype=route_dtype)

    with pytest.raises(TypeError, match="token_ids must be int32 or int64"):
        module.tilelang_sparse_gqa_thd_attention(query, key, value, token_ids)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("query_rank", ValueError, "ranks 3/3/3"),
        ("kv_mismatch", ValueError, "key and value shapes must match"),
        ("kv_heads", ValueError, r"key/value must be \[T, 2, D\]"),
        ("query_heads", ValueError, "requires Hq=24, Hkv=2, D=256"),
        ("head_dim", ValueError, "requires Hq=24, Hkv=2, D=256"),
        ("empty_tokens", ValueError, "requires at least one token"),
        ("compute_dtype", TypeError, "requires BF16 query, key, and value"),
        ("route_dtype", TypeError, "token_ids must be int32 or int64"),
    ],
)
def test_thd_forward_interface_rejects_invalid_tensor_contract(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    forward = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_thd_fwd")
    query, key, value, token_ids = _cpu_inputs()
    if case == "query_rank":
        query = query.unsqueeze(0)
    elif case == "kv_mismatch":
        value = value[:, :1]
    elif case == "kv_heads":
        key = key[:, :1]
        value = value[:, :1]
    elif case == "query_heads":
        query = query[:, :23]
    elif case == "head_dim":
        query = query[..., :255]
        key = key[..., :255]
        value = value[..., :255]
    elif case == "empty_tokens":
        query = query[:0]
        key = key[:0]
        value = value[:0]
        token_ids = token_ids[:0]
    elif case == "compute_dtype":
        query = query.float()
    elif case == "route_dtype":
        token_ids = token_ids.float()
    else:
        raise AssertionError(f"unknown case {case}")

    with pytest.raises(error, match=match):
        forward.sparse_gqa_thd_fwd_interface(query, key, value, token_ids)


def test_thd_forward_interface_rejects_cpu_before_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    forward = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_thd_fwd")
    query, key, value, token_ids = _cpu_inputs()

    def fail_if_compiled(*_args, **_kwargs):
        pytest.fail("CPU validation must fail before TileLang compilation")

    monkeypatch.setattr(forward, "sparse_gqa_thd_fwd", fail_if_compiled)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        forward.sparse_gqa_thd_fwd_interface(query, key, value, token_ids)


@pytest.mark.parametrize(
    ("scale", "error"),
    [
        (True, TypeError),
        ("0.125", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
    ],
)
def test_thd_public_api_rejects_invalid_softmax_scale(scale: object, error: type[Exception]) -> None:
    module = importlib.import_module("nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd")
    query, key, value, token_ids = _cpu_inputs()

    with pytest.raises(error, match="softmax_scale must be"):
        module.tilelang_sparse_gqa_thd_attention(query, key, value, token_ids, softmax_scale=scale)


def test_direct_thd_kernel_does_not_enable_packed_qwen_layer() -> None:
    from nemo_automodel.components.models.qwen4_exp.layers import Qwen4ExpQSAAttention

    attention = object.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(attention)
    with pytest.raises(NotImplementedError, match="packed THD attention is not yet supported"):
        attention(
            torch.zeros(1, 2, 8),
            freqs_cis=torch.zeros(1, 2, 4),
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        )


def test_direct_thd_kernel_does_not_enable_packed_qwen_cp() -> None:
    from nemo_automodel.components.models.qwen4_exp.cp import shard_batch_for_qwen4_cp

    with pytest.raises(NotImplementedError, match="does not support packed/THD batches"):
        shard_batch_for_qwen4_cp(object(), None, {"qkv_format": "thd"})


def _make_document_causal_routes(cu_seqlens: Sequence[int], topk: int, device: torch.device) -> torch.Tensor:
    """Build global, document-local causal routes.

    Args:
        cu_seqlens: Strictly increasing packed-document boundaries beginning at zero.
        topk: Fixed selected-token width.
        device: CUDA device on which to create the routes.

    Returns:
        Int32 global route tensor of shape ``[tokens, topk]``.
    """
    routes = torch.full((cu_seqlens[-1], topk), -1, dtype=torch.int32, device=device)
    for document_start, document_end in zip(cu_seqlens, cu_seqlens[1:]):
        for query_idx in range(document_start, document_end):
            valid_width = min(query_idx - document_start + 1, topk)
            first_token = query_idx - valid_width + 1
            routes[query_idx, :valid_width] = torch.arange(
                first_token,
                query_idx + 1,
                dtype=torch.int32,
                device=device,
            )
    return routes


def _make_cuda_inputs(cu_seqlens: Sequence[int], topk: int, seed: int) -> TensorTuple:
    """Build deterministic direct-THD inputs with document-safe global routes.

    Args:
        cu_seqlens: Strictly increasing packed-document boundaries beginning at zero.
        topk: Fixed selected-token width.
        seed: Random seed for Q/K/V values.

    Returns:
        Tuple containing BF16 query ``[tokens, 24, 256]``, key/value
        ``[tokens, 2, 256]``, and int32 routes ``[tokens, topk]`` on CUDA.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda", 0)
    tokens = cu_seqlens[-1]
    query = torch.randn(tokens, _QUERY_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16).mul_(0.125)
    key = torch.randn(tokens, _KV_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16).mul_(0.125)
    value = torch.randn_like(key).mul_(0.125)
    token_ids = _make_document_causal_routes(cu_seqlens, topk, device)
    return query, key, value, token_ids


def _noncontiguous_grad_output(query: torch.Tensor, seed: int) -> torch.Tensor:
    """Build a noncontiguous upstream gradient for direct-THD output.

    Args:
        query: Tensor of shape ``[tokens, 24, 256]`` whose shape/device define the gradient.
        seed: Random seed for gradient values.

    Returns:
        Noncontiguous BF16 tensor of shape ``[tokens, 24, 256]``.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    storage = torch.randn(
        query.shape[0],
        query.shape[1],
        query.shape[2] * 2,
        device=query.device,
        dtype=query.dtype,
    ).mul_(0.125)
    grad_output = storage[..., ::2]
    assert not grad_output.is_contiguous()
    return grad_output


def _direct_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the production direct-THD operator.

    Args:
        query: BF16 tensor of shape ``[tokens, 24, 256]``.
        key: BF16 tensor of shape ``[tokens, 2, 256]``.
        value: BF16 tensor of shape ``[tokens, 2, 256]``.
        token_ids: Global int32/int64 routes of shape ``[tokens, selected]`` or
            ``[tokens, 1, selected]``.

    Returns:
        BF16 attention output of shape ``[tokens, 24, 256]``.
    """
    from nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention_thd import (
        tilelang_sparse_gqa_thd_attention,
    )

    return tilelang_sparse_gqa_thd_attention(query, key, value, token_ids, softmax_scale=_SCALE)


def _bshd_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the existing fused BSHD operator for one document.

    Args:
        query: BF16 tensor of shape ``[tokens, 24, 256]``.
        key: BF16 tensor of shape ``[tokens, 2, 256]``.
        value: BF16 tensor of shape ``[tokens, 2, 256]``.
        token_ids: Document-local routes of shape ``[tokens, selected]``.

    Returns:
        BF16 attention output of shape ``[tokens, 24, 256]``.
    """
    from nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention import (
        tilelang_sparse_gqa_attention,
    )

    return tilelang_sparse_gqa_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        token_ids.unsqueeze(0),
        softmax_scale=_SCALE,
    ).squeeze(0)


def _eager_sparse_gqa_thd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Evaluate direct-THD sparse GQA with independent PyTorch math.

    Args:
        query: BF16 tensor of shape ``[tokens, 24, 256]``.
        key: BF16 tensor of shape ``[tokens, 2, 256]``.
        value: BF16 tensor of shape ``[tokens, 2, 256]``.
        token_ids: Global int32/int64 routes of shape ``[tokens, selected]`` or
            ``[tokens, 1, selected]``. Invalid slots are outside ``[0, tokens)``.

    Returns:
        BF16 attention output of shape ``[tokens, 24, 256]``.
    """
    if token_ids.ndim == 3:
        token_ids = token_ids[:, 0]
    tokens, query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    group_heads = query_heads // kv_heads
    valid = (token_ids >= 0) & (token_ids < tokens)
    safe_ids = token_ids.clamp(min=0, max=tokens - 1).long()
    gathered_key = key[safe_ids].permute(0, 2, 1, 3)
    gathered_value = value[safe_ids].permute(0, 2, 1, 3)
    grouped_query = query.unflatten(1, (kv_heads, group_heads))
    scores = torch.einsum("thgd,thkd->thgk", grouped_query.float(), gathered_key.float()) * (head_dim**-0.5)
    scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
    has_tokens = valid.any(dim=-1)
    scores = torch.where(has_tokens[:, None, None, None], scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1).masked_fill(~valid[:, None, None, :], 0.0)
    output = torch.einsum("thgk,thkd->thgd", probabilities, gathered_value.float())
    return output.flatten(1, 2).to(query.dtype)


def _per_document_bshd(cu_seqlens: Sequence[int]) -> Operator:
    """Create a BSHD oracle that runs each packed document independently."""

    def operator(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run independent BSHD attention over a packed THD allocation.

        Args:
            query: BF16 tensor of shape ``[tokens, 24, 256]``.
            key: BF16 tensor of shape ``[tokens, 2, 256]``.
            value: BF16 tensor of shape ``[tokens, 2, 256]``.
            token_ids: Physical-global routes of shape ``[tokens, selected]``.

        Returns:
            Concatenated BF16 output of shape ``[tokens, 24, 256]``.
        """
        outputs = []
        for document_start, document_end in zip(cu_seqlens, cu_seqlens[1:]):
            document_ids = token_ids[document_start:document_end]
            document_ids = torch.where(document_ids >= 0, document_ids - document_start, document_ids)
            outputs.append(
                _bshd_attention(
                    query[document_start:document_end],
                    key[document_start:document_end],
                    value[document_start:document_end],
                    document_ids,
                )
            )
        return torch.cat(outputs, dim=0)

    return operator


def _padded_storage_bshd(real_cu_seqlens: Sequence[int], padded_cu_seqlens: Sequence[int]) -> Operator:
    """Create a per-document BSHD oracle that leaves physical padding gaps unused."""

    def operator(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run BSHD over real tokens and emit zeros for physical gaps.

        Args:
            query: BF16 tensor of shape ``[physical_tokens, 24, 256]``.
            key: BF16 tensor of shape ``[physical_tokens, 2, 256]``.
            value: BF16 tensor of shape ``[physical_tokens, 2, 256]``.
            token_ids: Physical-global routes of shape ``[physical_tokens, selected]``.

        Returns:
            BF16 output of shape ``[physical_tokens, 24, 256]`` with exact-zero gaps.
        """
        segments = []
        for document_idx in range(len(real_cu_seqlens) - 1):
            real_length = real_cu_seqlens[document_idx + 1] - real_cu_seqlens[document_idx]
            physical_start = padded_cu_seqlens[document_idx]
            physical_end = padded_cu_seqlens[document_idx + 1]
            real_end = physical_start + real_length
            document_ids = token_ids[physical_start:real_end]
            document_ids = torch.where(document_ids >= 0, document_ids - physical_start, document_ids)
            segments.append(
                _bshd_attention(
                    query[physical_start:real_end],
                    key[physical_start:real_end],
                    value[physical_start:real_end],
                    document_ids,
                )
            )
            if real_end < physical_end:
                segments.append(torch.zeros_like(query[real_end:physical_end]))
        return torch.cat(segments, dim=0)

    return operator


def _forward_backward(
    operator: Operator,
    tensors: TensorTuple,
    grad_output: torch.Tensor,
) -> ForwardBackwardResult:
    """Run one attention operator and collect every differentiable result.

    Args:
        operator: Attention callable consuming direct THD Q/K/V and routes.
        tensors: Tuple containing query ``[tokens, 24, 256]``, key/value
            ``[tokens, 2, 256]``, and routes ``[tokens, selected]`` or
            ``[tokens, 1, selected]``.
        grad_output: Upstream gradient of shape ``[tokens, 24, 256]``.

    Returns:
        Tuple containing output/dQ ``[tokens, 24, 256]`` and dK/dV
        ``[tokens, 2, 256]``.
    """
    query, key, value, token_ids = tensors
    query_leaf = query.detach().requires_grad_(True)
    key_leaf = key.detach().requires_grad_(True)
    value_leaf = value.detach().requires_grad_(True)
    assert query_leaf.stride() == query.stride()
    assert key_leaf.stride() == key.stride()
    assert value_leaf.stride() == value.stride()
    output = operator(query_leaf, key_leaf, value_leaf, token_ids)
    grad_query, grad_key, grad_value = torch.autograd.grad(
        output,
        (query_leaf, key_leaf, value_leaf),
        grad_outputs=grad_output,
    )
    torch.cuda.synchronize()
    return tuple(tensor.detach() for tensor in (output, grad_query, grad_key, grad_value))


def _assert_parity(actual: ForwardBackwardResult, expected: ForwardBackwardResult) -> None:
    """Apply promotion-level numerical gates to output and Q/K/V gradients.

    Args:
        actual: Direct-kernel output/dQ ``[tokens, 24, 256]`` and dK/dV
            ``[tokens, 2, 256]``.
        expected: Oracle tensors with layouts corresponding exactly to ``actual``.
    """
    relative_l2_limits = {"output": 0.003, "dq": 0.005, "dk": 0.015, "dv": 0.015}
    max_abs_limits = {"output": 0.01, "dq": 0.001, "dk": 0.005, "dv": 0.05}
    for label, actual_tensor, expected_tensor in zip(
        ("output", "dq", "dk", "dv"),
        actual,
        expected,
        strict=True,
    ):
        actual_float = actual_tensor.float().flatten()
        expected_float = expected_tensor.float().flatten()
        assert torch.isfinite(actual_float).all(), f"{label} contains non-finite values"
        assert torch.isfinite(expected_float).all(), f"oracle {label} contains non-finite values"
        expected_norm = torch.linalg.vector_norm(expected_float)
        difference = actual_float - expected_float
        max_abs = difference.abs().max()
        if expected_norm == 0:
            assert max_abs <= max_abs_limits[label], f"{label} max_abs={max_abs.item()}"
            continue
        relative_l2 = torch.linalg.vector_norm(difference) / expected_norm
        cosine = torch.nn.functional.cosine_similarity(actual_float, expected_float, dim=0)
        assert relative_l2 <= relative_l2_limits[label], f"{label} relative_l2={relative_l2.item()}"
        assert cosine >= 0.999, f"{label} cosine={cosine.item()}"
        assert max_abs <= max_abs_limits[label], f"{label} max_abs={max_abs.item()}"


def test_parity_gate_uses_absolute_tolerance_for_zero_norm_oracle() -> None:
    """Allow cancellation noise against zero while retaining the max-absolute gate."""
    expected = (
        torch.zeros(1, _QUERY_HEADS, _HEAD_DIM),
        torch.zeros(1, _QUERY_HEADS, _HEAD_DIM),
        torch.zeros(1, _KV_HEADS, _HEAD_DIM),
        torch.zeros(1, _KV_HEADS, _HEAD_DIM),
    )
    within_tolerance = tuple(tensor.clone() for tensor in expected)
    within_tolerance[1][0, 0, 0] = 1.0e-8
    _assert_parity(within_tolerance, expected)

    outside_tolerance = tuple(tensor.clone() for tensor in expected)
    outside_tolerance[0][0, 0, 0] = 0.02
    with pytest.raises(AssertionError, match="output max_abs"):
        _assert_parity(outside_tolerance, expected)


@requires_h100_tilelang
@pytest.mark.parametrize(
    ("route_kind", "topk", "rank3_ids", "seed"),
    [
        ("duplicates", 8, False, 202),
        ("all_invalid", 7, False, 303),
        ("int64_overflow", 5, False, 353),
        ("tail_k2051", _MAX_TOPK, True, 404),
    ],
)
def test_thd_tilelang_matches_eager_forward_backward(
    route_kind: str,
    topk: int,
    rank3_ids: bool,
    seed: int,
) -> None:
    """Cover duplicates, invalid rows, int64 overflow, K=2051, and strided dO."""
    tokens = 8 if route_kind in ("duplicates", "tail_k2051") else 4
    tensors = _make_cuda_inputs((0, tokens), topk, seed)
    if route_kind == "duplicates":
        row = torch.tensor([0, 0, 1, 1, 2, 2, -1, -1], dtype=torch.int32, device="cuda")
        token_ids = row.unsqueeze(0).expand(tokens, -1).clone()
        query_positions = torch.arange(tokens, device="cuda")[:, None]
        token_ids = torch.where(token_ids <= query_positions, token_ids, -1)
        tensors = (*tensors[:3], token_ids)
    elif route_kind == "all_invalid":
        token_ids = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
        tensors = (*tensors[:3], token_ids)
    elif route_kind == "int64_overflow":
        token_ids = (
            torch.tensor(
                [[0, 2**32, -(2**40), -1, -1]],
                dtype=torch.int64,
                device="cuda",
            )
            .expand(tokens, -1)
            .clone()
        )
        token_ids[1:, 3] = 1
        tensors = (*tensors[:3], token_ids)
    elif route_kind != "tail_k2051":
        raise AssertionError(f"unknown route kind {route_kind}")
    if rank3_ids:
        tensors = (*tensors[:3], tensors[3].unsqueeze(1))

    grad_output = _noncontiguous_grad_output(tensors[0], seed + 1)
    actual = _forward_backward(_direct_attention, tensors, grad_output)
    expected = _forward_backward(_eager_sparse_gqa_thd, tensors, grad_output)
    _assert_parity(actual, expected)
    if route_kind == "all_invalid":
        for tensor in actual:
            assert torch.count_nonzero(tensor) == 0


@requires_h100_tilelang
def test_thd_tilelang_minimum_shape_accepts_strided_inputs() -> None:
    """Cover ``T=1, K=1`` and the advertised Q/K/V contiguity conversion."""
    torch.manual_seed(909)
    query = torch.randn(1, _QUERY_HEADS, _HEAD_DIM * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    key = torch.randn(1, _KV_HEADS, _HEAD_DIM * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    value = torch.randn(1, _KV_HEADS, _HEAD_DIM * 2, device="cuda", dtype=torch.bfloat16)[..., ::2]
    token_ids = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    tensors = (query, key, value, token_ids)
    assert not query.is_contiguous()
    assert not key.is_contiguous()
    assert not value.is_contiguous()

    grad_output = _noncontiguous_grad_output(query, seed=910)
    actual = _forward_backward(_direct_attention, tensors, grad_output)
    expected = _forward_backward(_eager_sparse_gqa_thd, tensors, grad_output)
    _assert_parity(actual, expected)


@requires_h100_tilelang
def test_thd_tilelang_matches_per_document_bshd_with_global_ids() -> None:
    """Prove one direct launch respects physical-global packed-document routes."""
    cu_seqlens = (0, 1, 6, 13, 32)
    tensors = _make_cuda_inputs(cu_seqlens, topk=17, seed=101)
    for document_start, document_end in zip(cu_seqlens, cu_seqlens[1:]):
        assert tensors[3][document_start, 0] == document_start
        assert tensors[3][document_end - 1].max() < document_end

    grad_output = _noncontiguous_grad_output(tensors[0], seed=102)
    actual = _forward_backward(_direct_attention, tensors, grad_output)
    expected = _forward_backward(_per_document_bshd(cu_seqlens), tensors, grad_output)
    _assert_parity(actual, expected)


@requires_h100_tilelang
def test_thd_tilelang_padded_physical_gaps_are_exact_zero() -> None:
    """Verify output and every gradient remain zero in unaddressed physical gaps."""
    real_cu_seqlens = (0, 3, 5, 10)
    padded_cu_seqlens = (0, 4, 8, 14)
    topk = 7
    tensors = _make_cuda_inputs((0, padded_cu_seqlens[-1]), topk=topk, seed=77)
    token_ids = torch.full((padded_cu_seqlens[-1], topk), -1, dtype=torch.int32, device="cuda")
    gap_indices: list[int] = []
    for document_idx in range(len(real_cu_seqlens) - 1):
        real_length = real_cu_seqlens[document_idx + 1] - real_cu_seqlens[document_idx]
        physical_start = padded_cu_seqlens[document_idx]
        physical_end = padded_cu_seqlens[document_idx + 1]
        for local_query in range(real_length):
            visible = min(local_query + 1, topk)
            query_idx = physical_start + local_query
            token_ids[query_idx, :visible] = torch.arange(
                query_idx - visible + 1,
                query_idx + 1,
                dtype=torch.int32,
                device="cuda",
            )
        gap_indices.extend(range(physical_start + real_length, physical_end))
    tensors = (*tensors[:3], token_ids)

    grad_output = _noncontiguous_grad_output(tensors[0], seed=78)
    actual = _forward_backward(_direct_attention, tensors, grad_output)
    expected = _forward_backward(
        _padded_storage_bshd(real_cu_seqlens, padded_cu_seqlens),
        tensors,
        grad_output,
    )
    _assert_parity(actual, expected)
    gaps = torch.tensor(gap_indices, dtype=torch.long, device="cuda")
    for tensor in actual:
        assert torch.count_nonzero(tensor.index_select(0, gaps)) == 0
