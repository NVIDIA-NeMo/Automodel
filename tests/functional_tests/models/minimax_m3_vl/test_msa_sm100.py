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

"""SM100 parity gates for MiniMax M3 flat MSA prefill training."""

import copy
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl import msa
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Attention, MiniMaxM3Indexer
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM
from nemo_automodel.components.models.minimax_m3_vl.msa import _MSAFlatAttention
from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSAPackedLayout
from nemo_automodel.shared.import_utils import UnavailableError

_BLOCK_SIZE = 128
_QUERY_HEADS = 64
_KV_HEADS = 4
_HEAD_DIM = 128
_TOPK = 16
_SOFTMAX_SCALE = _HEAD_DIM**-0.5
_PACKED_DOCUMENT_LENGTHS = (127, 129, 1, 2, 3, 5, 128)


def _msa_sm100_unavailable_reason() -> str | None:
    """Return why the optional SM100 test cannot run, or ``None``.

    Returns:
        A human-readable skip reason, or ``None`` when the device and optional
        MSA dependencies are available.
    """
    if not torch.cuda.is_available():
        return "MiniMax M3 MSA functional tests require CUDA"
    if torch.cuda.get_device_capability() != (10, 0):
        return "MiniMax M3 MSA functional tests require an SM100 GPU"
    try:
        msa._require_msa()
        msa._require_msa_backward()
    except UnavailableError:
        return "MiniMax M3 MSA functional tests require `uv sync --extra msa`"
    return None


_IS_PP_WORKER = __name__ == "__main__" and "--pp-worker" in sys.argv
_UNAVAILABLE_REASON = None if _IS_PP_WORKER else _msa_sm100_unavailable_reason()
pytestmark = pytest.mark.skipif(
    _UNAVAILABLE_REASON is not None,
    reason=_UNAVAILABLE_REASON or "MiniMax M3 MSA functional-test prerequisites are available",
)


def _packed_document_ids(device: torch.device) -> torch.Tensor:
    """Build B>1 metadata covering residues, short documents, and padding.

    Args:
        device: CUDA device for the returned metadata.

    Returns:
        Int64 document ids with shape ``[2, 260]``. Row zero contains 127- and
        129-token documents. Row one contains 1-, 2-, 3-, 5-, and 128-token
        documents. Positive ids are local to each batch row and zero is
        padding.
    """
    doc_ids = torch.zeros(2, 260, dtype=torch.int64, device=device)
    doc_ids[0, :127] = 1
    doc_ids[0, 127:256] = 2
    position = 0
    for document, length in enumerate(_PACKED_DOCUMENT_LENGTHS[2:], start=1):
        doc_ids[1, position : position + length] = document
        position += length
    return doc_ids


def _random_qkv(
    total_tokens: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic trainable compact BF16 Q/K/V tensors.

    Args:
        total_tokens: Number of compact token rows.
        device: CUDA device for the returned tensors.
        seed: Random seed used by the device-local generator.

    Returns:
        Q of shape ``[tokens, 64, 128]`` and K/V of shape
        ``[tokens, 4, 128]`` on ``device``.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(
            total_tokens,
            heads,
            _HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        for heads in (_QUERY_HEADS, _KV_HEADS, _KV_HEADS)
    )


def _full_causal_q2k(document_lengths: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Select every document-local causal block at fixed width 16.

    Args:
        document_lengths: Length of each document on the compact token axis.
        device: CUDA device for the returned support.

    Returns:
        Contiguous int32 support with shape ``[4, T, 16]``. Valid entries are
        local block ids and unused entries are ``-1``.
    """
    total_tokens = sum(document_lengths)
    lengths = torch.tensor(document_lengths, dtype=torch.int64, device=device)
    cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), lengths.cumsum(0)))
    compact_document_starts = torch.repeat_interleave(
        cu_seqlens[:-1],
        lengths,
        output_size=total_tokens,
    )
    local_query_positions = torch.arange(total_tokens, device=device) - compact_document_starts
    current_blocks = torch.div(local_query_positions, _BLOCK_SIZE, rounding_mode="floor")
    slots = torch.arange(_TOPK, dtype=torch.int64, device=current_blocks.device).view(1, 1, -1)
    support = torch.where(
        slots <= current_blocks.view(1, -1, 1),
        slots,
        torch.full_like(slots, -1),
    )
    return support.expand(_KV_HEADS, -1, -1).to(torch.int32).contiguous()


def _semantic_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    document_lengths: tuple[int, ...],
    query_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the exact canonical support with FP32 PyTorch operations.

    Args:
        q: FP32 compact queries of shape ``[T, 64, 128]``.
        k: FP32 compact keys of shape ``[T, 4, 128]``.
        v: FP32 compact values of shape ``[T, 4, 128]``.
        q2k: Document-local fixed-width support ``[4, T, 16]``.
        document_lengths: Length of each document on the compact token axis.
        query_rows: Optional int64 compact rows to evaluate. All rows are used
            when omitted.

    Returns:
        FP32 output of shape ``[selected_tokens, 64, 128]`` and natural-log
        LSE of shape ``[selected_tokens, 64]``. Both are differentiable with
        respect to ``q``, ``k``, and ``v``.
    """
    total_tokens = q.shape[0]
    if query_rows is None:
        query_rows = torch.arange(total_tokens, dtype=torch.int64, device=q.device)
    kv_for_query_head = torch.div(
        torch.arange(_QUERY_HEADS, device=q.device),
        _QUERY_HEADS // _KV_HEADS,
        rounding_mode="floor",
    )
    q_heads = q.index_select(0, query_rows).transpose(0, 1)
    k_heads = k.index_select(1, kv_for_query_head).transpose(0, 1)
    v_heads = v.index_select(1, kv_for_query_head).transpose(0, 1)
    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * _SOFTMAX_SCALE

    lengths = torch.tensor(document_lengths, dtype=torch.int64, device=q.device)
    cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int64, device=q.device), lengths.cumsum(0)))
    document_ordinals = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int64, device=q.device),
        lengths,
        output_size=total_tokens,
    )
    compact_document_starts = torch.repeat_interleave(
        cu_seqlens[:-1],
        lengths,
        output_size=total_tokens,
    )
    local_key_blocks = torch.div(
        torch.arange(total_tokens, dtype=torch.int64, device=q.device) - compact_document_starts,
        _BLOCK_SIZE,
        rounding_mode="floor",
    )
    selected_blocks = q2k.index_select(1, query_rows).to(torch.int64)
    selected_keys = (selected_blocks.unsqueeze(-1) == local_key_blocks.view(1, 1, 1, -1)).any(dim=2)
    same_document = document_ordinals.index_select(0, query_rows).view(-1, 1) == document_ordinals.view(1, -1)
    causal = torch.arange(total_tokens, device=q.device).view(1, -1) <= query_rows.view(-1, 1)
    keep_kv = selected_keys & same_document.unsqueeze(0) & causal.unsqueeze(0)
    keep = keep_kv.repeat_interleave(_QUERY_HEADS // _KV_HEADS, dim=0)

    masked_scores = scores.masked_fill(~keep, float("-inf"))
    out = torch.matmul(torch.softmax(masked_scores, dim=-1), v_heads).transpose(0, 1)
    return out, torch.logsumexp(masked_scores, dim=-1).transpose(0, 1).contiguous()


def _assert_semantic_error(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_abs: float,
    norm_rel: float,
) -> None:
    """Assert measured BF16 semantic-error bounds and report all metrics.

    Args:
        name: Quantity being compared.
        actual: Kernel-result tensor of arbitrary shape.
        reference: FP32 semantic-reference tensor with the same shape as
            ``actual``.
        max_abs: Maximum accepted absolute error.
        norm_rel: Maximum accepted L2-relative error.

    Returns:
        None.
    """
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = (actual_fp32 - reference_fp32).abs()
    stats = {
        "max_abs": difference.max().item(),
        "max_rel": (difference / reference_fp32.abs().clamp_min(1e-6)).max().item(),
        "norm_rel": (difference.norm() / reference_fp32.norm().clamp_min(1e-12)).item(),
    }
    assert stats["max_abs"] <= max_abs and stats["norm_rel"] <= norm_rel, f"{name}: {stats}"


def test_flat_packed_forward_backward_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate packed residues, short documents, and the delta tail."""
    device = torch.device("cuda", torch.cuda.current_device())
    doc_ids = _packed_document_ids(device)
    layout = _MSAPackedLayout.build(doc_ids)
    q2k = _full_causal_q2k(_PACKED_DOCUMENT_LENGTHS, device)
    total_tokens = sum(_PACKED_DOCUMENT_LENGTHS)
    assert total_tokens == 395

    q, k, v = _random_qkv(total_tokens, device=device, seed=20260902)
    generator = torch.Generator(device=device).manual_seed(20260903)
    grad_out = torch.randn(q.shape, device=device, dtype=torch.bfloat16, generator=generator)

    lse_results: list[torch.Tensor] = []
    dependency = msa._require_msa()
    real_forward = dependency.sparse_atten_func

    def capture_forward(
        q_compact: torch.Tensor,
        k_compact: torch.Tensor,
        v_compact: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        topk_blocks: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture the official flat call and its internal LSE result.

        Args:
            q_compact: BF16 query tensor of shape [tokens, 64, 128].
            k_compact: BF16 key tensor of shape [tokens, 4, 128].
            v_compact: BF16 value tensor of shape [tokens, 4, 128].
            k2q_row_ptr: Int32 CSR row-pointer tensor of shape [rows + 1].
            k2q_q_indices: Int32 CSR query-index tensor of shape [edges].
            topk_blocks: Fixed support width, equal to 16.
            **kwargs: Flat-varlen causal launch metadata.

        Returns:
            BF16 output of shape [tokens, 64, 128] and FP32 LSE of shape
            [tokens, 64].
        """
        result = real_forward(
            q_compact,
            k_compact,
            v_compact,
            k2q_row_ptr,
            k2q_q_indices,
            topk_blocks,
            **kwargs,
        )
        lse_results.append(result[1])
        return result

    monkeypatch.setattr(msa, "_resolve_msa_forward", lambda: replace(dependency, sparse_atten_func=capture_forward))

    out = _MSAFlatAttention(_SOFTMAX_SCALE)(q, k, v, q2k, layout=layout)
    out.backward(grad_out)

    q_reference = q.detach().float().requires_grad_()
    k_reference = k.detach().float().requires_grad_()
    v_reference = v.detach().float().requires_grad_()
    out_reference, lse_reference = _semantic_sparse_attention(
        q_reference,
        k_reference,
        v_reference,
        q2k,
        _PACKED_DOCUMENT_LENGTHS,
    )
    out_reference.backward(grad_out.float())

    _assert_semantic_error("out", out, out_reference, max_abs=0.025, norm_rel=0.006)
    _assert_semantic_error("lse", lse_results[-1], lse_reference, max_abs=0.002, norm_rel=0.001)
    _assert_semantic_error("dQ", q.grad, q_reference.grad, max_abs=0.035, norm_rel=0.007)
    _assert_semantic_error("dK", k.grad, k_reference.grad, max_abs=0.075, norm_rel=0.007)
    _assert_semantic_error("dV", v.grad, v_reference.grad, max_abs=0.1, norm_rel=0.007)


def test_flat_attention_rejects_contiguous_misaligned_query() -> None:
    """Reject a contiguous storage-offset view that violates the compiled ABI."""
    device = torch.device("cuda", torch.cuda.current_device())
    total_tokens = 1
    query_storage = torch.empty(
        total_tokens * _QUERY_HEADS * _HEAD_DIM + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    query = query_storage[1:].view(total_tokens, _QUERY_HEADS, _HEAD_DIM)
    key = torch.empty(total_tokens, _KV_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.empty_like(key)
    support = torch.full((_KV_HEADS, total_tokens, _TOPK), -1, dtype=torch.int32, device=device)
    support[:, :, 0] = 0
    layout = _MSAPackedLayout.build(torch.ones(1, total_tokens, dtype=torch.int64, device=device))

    assert query.is_contiguous()
    assert query.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte-aligned q/k/v"):
        _MSAFlatAttention(_SOFTMAX_SCALE)(query, key, value, support, layout=layout)


def test_flat_attention_binds_external_launches_to_tensor_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run external launch callbacks on q.device even when another GPU is current."""
    if torch.cuda.device_count() < 2:
        pytest.skip("current-device guard requires two CUDA devices")
    original_device = torch.cuda.current_device()
    tensor_device_index = 1 if original_device == 0 else 0
    if torch.cuda.get_device_capability(tensor_device_index) != (10, 0):
        pytest.skip("current-device guard requires a second SM100 GPU")
    tensor_device = torch.device("cuda", tensor_device_index)
    with torch.cuda.device(tensor_device):
        q, k, v = _random_qkv(1, device=tensor_device, seed=20260908)
        q2k = _full_causal_q2k((1,), tensor_device)
        layout = _MSAPackedLayout.build(torch.ones(1, 1, dtype=torch.int64, device=tensor_device))
        row_ptr = torch.zeros(4, 2, dtype=torch.int32, device=tensor_device)
        q_indices = torch.zeros(4, 16, dtype=torch.int32, device=tensor_device)
        scheduler_metadata = torch.zeros(1, 6, dtype=torch.int32, device=tensor_device)
        work_count = torch.zeros(1, dtype=torch.int32, device=tensor_device)

    class _Schedule:
        pass

    schedule = _Schedule()
    schedule.scheduler_metadata = scheduler_metadata
    schedule.work_count = work_count

    def fake_build(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
        del args, kwargs
        assert torch.cuda.current_device() == tensor_device_index
        return row_ptr, q_indices, schedule

    def fake_forward(q_compact: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        assert torch.cuda.current_device() == tensor_device_index
        return torch.zeros_like(q_compact), torch.zeros(q_compact.shape[:2], dtype=torch.float32, device=tensor_device)

    monkeypatch.setattr(
        msa,
        "_require_msa",
        lambda: msa._MSAForwardKernels(build_k2q_csr=fake_build, sparse_atten_func=fake_forward),
    )
    output = _MSAFlatAttention(_SOFTMAX_SCALE)(q, k, v, q2k, layout=layout)

    assert output.device == tensor_device
    assert torch.cuda.current_device() == original_device


def test_nontrivial_top16_support_drives_large_schedule_output_and_gradient() -> None:
    """Validate truncated top-16 support through the large backward schedule."""
    device = torch.device("cuda", torch.cuda.current_device())
    total_tokens = 17 * _BLOCK_SIZE + 1
    layout = _MSAPackedLayout.build(torch.ones(1, total_tokens, dtype=torch.int64, device=device))
    config = _fixed_attention_config()
    with device:
        indexer = MiniMaxM3Indexer(
            config,
            config.sparse_attention_config,
            _attention_backend("msa"),
        )

    index_q = torch.zeros(total_tokens, _KV_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    index_q[..., 0] = 1
    block_scores = torch.tensor(
        (-90, 90, -80, 80, -70, 70, -60, 60, -50, 50, -40, 40, -30, 30, -20, 20, -10, 0),
        dtype=torch.bfloat16,
        device=device,
    )
    index_k = torch.zeros(total_tokens, 1, _HEAD_DIM, dtype=torch.bfloat16, device=device)
    index_k[:, 0, 0] = block_scores.repeat_interleave(_BLOCK_SIZE)[:total_tokens]

    support = indexer._select_msa_blocks(index_q, index_k, layout=layout)
    final_support = support[0, -1]
    assert final_support.unique().numel() == _TOPK
    assert 17 in final_support
    assert 0 not in final_support and 2 not in final_support

    generator = torch.Generator(device=device).manual_seed(20260905)
    q, k, v = _random_qkv(total_tokens, device=device, seed=20260905)
    out = _MSAFlatAttention(_SOFTMAX_SCALE)(q, k, v, support, layout=layout)
    query_rows = torch.arange(total_tokens - 8, total_tokens, dtype=torch.int64, device=device)
    selected_grad = torch.randn(8, _QUERY_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device, generator=generator)
    grad_out = torch.zeros_like(out).index_copy(0, query_rows, selected_grad)
    out.backward(grad_out)

    q_reference = q.detach().float().requires_grad_()
    k_reference = k.detach().float().requires_grad_()
    v_reference = v.detach().float().requires_grad_()
    out_reference, _ = _semantic_sparse_attention(
        q_reference,
        k_reference,
        v_reference,
        support,
        (total_tokens,),
        query_rows=query_rows,
    )
    out_reference.backward(selected_grad.float())

    _assert_semantic_error(
        "truncated-top16 out", out.index_select(0, query_rows), out_reference, max_abs=0.025, norm_rel=0.006
    )
    _assert_semantic_error("truncated-top16 dQ", q.grad, q_reference.grad, max_abs=0.035, norm_rel=0.007)
    _assert_semantic_error("truncated-top16 dK", k.grad, k_reference.grad, max_abs=0.075, norm_rel=0.007)
    _assert_semantic_error("truncated-top16 dV", v.grad, v_reference.grad, max_abs=0.1, norm_rel=0.007)


def _fixed_attention_config() -> MiniMaxM3VLTextConfig:
    """Build a tiny dense-to-MSA model config with the fixed head topology.

    Returns:
        Configuration accepted by both the layer and PP parity tests.
    """
    return MiniMaxM3VLTextConfig(
        hidden_size=32,
        intermediate_size=32,
        dense_intermediate_size=48,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=_QUERY_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=64,
        max_position_embeddings=32,
        rotary_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        moe_layer_freq=[0, 0],
        num_mtp_modules=0,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_num_index_heads": _KV_HEADS,
            "sparse_index_dim": _HEAD_DIM,
            "sparse_block_size": _BLOCK_SIZE,
            "sparse_topk_blocks": _TOPK,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            "sparse_attention_freq": [0, 1],
            "sparse_disable_index_value": [0, 1],
        },
        attention_dropout=0.0,
    )


def _attention_backend(sparse_attn: str) -> BackendConfig:
    """Build a deterministic torch backend for one sparse attention layer.

    Args:
        sparse_attn: Either ``"generic"`` or ``"msa"``.

    Returns:
        Backend configuration using PyTorch projections and SDPA dense math.
    """
    return BackendConfig(
        attn="sdpa",
        sparse_attn=sparse_attn,
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_real_sparse_layer_checkpoint_parameter_gradient_parity() -> None:
    """Compare one checkpointed MSA layer backward with a generic peer."""
    device = torch.device("cuda", torch.cuda.current_device())
    config = _fixed_attention_config()
    torch.manual_seed(20260903)
    with device:
        msa_attention = MiniMaxM3Attention(
            config,
            _attention_backend("msa"),
            is_sparse_attention_layer=True,
        )
        generic_attention = MiniMaxM3Attention(
            config,
            _attention_backend("generic"),
            is_sparse_attention_layer=True,
        )
    generic_attention.load_state_dict(msa_attention.state_dict())
    msa_attention.train()
    generic_attention.train()

    sequence_length = 16
    doc_ids = torch.ones(2, sequence_length, dtype=torch.int64, device=device)
    doc_ids[0, -1] = 0
    layout = _MSAPackedLayout.build(doc_ids)
    rotary_dim = 64
    positions = torch.arange(sequence_length, dtype=torch.float32, device=device).view(1, -1, 1)
    inv_freq = 10_000 ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    angles = positions * inv_freq
    frequencies = torch.cat((angles.cos(), angles.sin()), dim=-1).expand(2, -1, -1)
    generator = torch.Generator(device=device).manual_seed(20260904)
    x = torch.randn(
        2,
        sequence_length,
        config.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    x_msa = x.detach().requires_grad_()
    x_generic = x.detach().clone().requires_grad_()
    token_keep = doc_ids > 0
    random_grad_out = torch.randn(x.shape, dtype=torch.bfloat16, device=device, generator=generator)
    grad_out = torch.where(token_keep.unsqueeze(-1), random_grad_out, torch.zeros_like(random_grad_out))

    def checkpoint_layer(hidden: torch.Tensor) -> torch.Tensor:
        """Run one recomputed MSA attention layer.

        Args:
            hidden: BF16 tensor of shape [batch, sequence, hidden].

        Returns:
            BF16 tensor of shape [batch, sequence, hidden].
        """
        return msa_attention(
            hidden,
            freqs_cis=frequencies,
            attention_mask=token_keep,
            _msa_layout=layout,
        )

    msa_out = checkpoint(checkpoint_layer, x_msa, use_reentrant=False)
    generic_out = generic_attention(x_generic, freqs_cis=frequencies, attention_mask=token_keep)
    assert torch.isfinite(msa_out).all() and torch.isfinite(generic_out).all()
    msa_out.backward(grad_out)
    generic_out.backward(grad_out)

    assert torch.count_nonzero(msa_out[~token_keep]) == 0
    assert torch.count_nonzero(x_msa.grad[~token_keep]) == 0
    assert torch.count_nonzero(x_generic.grad[~token_keep]) == 0

    # Each batch row contains one at-most-one-block document, so both selectors
    # expose the same support on real rows.
    _assert_semantic_error(
        "layer out",
        layout.pack(msa_out),
        layout.pack(generic_out),
        max_abs=0.04,
        norm_rel=0.008,
    )
    _assert_semantic_error(
        "layer input grad",
        layout.pack(x_msa.grad),
        layout.pack(x_generic.grad),
        max_abs=0.04,
        norm_rel=0.012,
    )

    projection_names = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
    projection_grad_max_abs = {
        "q_proj.weight": 0.125,
        "k_proj.weight": 0.5,
        "v_proj.weight": 0.5,
        "o_proj.weight": 0.125,
    }
    msa_parameters = dict(msa_attention.named_parameters())
    generic_parameters = dict(generic_attention.named_parameters())
    for name in projection_names:
        assert msa_parameters[name].grad is not None
        assert generic_parameters[name].grad is not None
        assert torch.isfinite(msa_parameters[name].grad).all()
        assert torch.count_nonzero(msa_parameters[name].grad) > 0
        _assert_semantic_error(
            f"{name} grad",
            msa_parameters[name].grad,
            generic_parameters[name].grad,
            max_abs=projection_grad_max_abs[name],
            norm_rel=0.015,
        )


def _pp_sm100_unavailable_reason() -> str | None:
    """Return why the two-rank MSA gate cannot run, or ``None``."""
    if _UNAVAILABLE_REASON is not None:
        return _UNAVAILABLE_REASON
    if torch.cuda.device_count() < 2:
        return "MiniMax M3 MSA PP parity requires two CUDA GPUs"
    if any(torch.cuda.get_device_capability(device_index) != (10, 0) for device_index in range(2)):
        return "MiniMax M3 MSA PP parity requires two SM100 GPUs"
    return None


_PP_UNAVAILABLE_REASON = None if _IS_PP_WORKER else _pp_sm100_unavailable_reason()
_PP_RESULT_PREFIX = "MINIMAX_M3_MSA_PP_RESULT "


def _build_pp_model(device: torch.device) -> torch.nn.Module:
    """Build the two-layer dense-to-MSA model used by the PP parity gate."""
    torch.manual_seed(20260906)
    with device:
        model = MiniMaxM3SparseForCausalLM(_fixed_attention_config(), backend=_attention_backend("msa"))
        model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    return model.train()


def _run_pp_worker() -> None:
    """Compare a real two-rank pipeline with its single-process model oracle."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.pipelining import AutoPipeline

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError(f"MiniMax M3 MSA PP parity requires SM100, got {torch.cuda.get_device_name(device)}")
    msa._require_msa()
    msa._require_msa_backward()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    try:
        model = _build_pp_model(device)
        reference = copy.deepcopy(model)

        def loss_fn(output: torch.Tensor, upstream: torch.Tensor) -> torch.Tensor:
            """Apply a deterministic upstream gradient to pipeline logits.

            Args:
                output: BF16 logits of shape [microbatch, sequence, vocab].
                upstream: BF16 gradient tensor with the same shape as ``output``.

            Returns:
                FP32 scalar inner-product loss.
            """
            return (output.float() * upstream.float()).sum()

        mesh = init_device_mesh("cuda", (2, 1), mesh_dim_names=("pp", "dp"))
        pipeline = AutoPipeline(
            world_mesh=mesh,
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=device,
            dtype=torch.bfloat16,
            pp_seq_len=8,
        ).build(model, loss_fn=loss_fn)

        generator = torch.Generator().manual_seed(20260907)
        input_ids = torch.randint(1, 64, (2, 8), generator=generator).to(device)
        document_ids = torch.tensor(
            [[1, 1, 1, 1, 2, 2, 2, 2], [7, 7, 7, 9, 9, 9, 9, 9]],
            device=device,
        )
        same_document = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
        attention_mask = (same_document & torch.ones(8, 8, dtype=torch.bool, device=device).tril()).unsqueeze(1)
        upstream = torch.randn(2, 8, 64, generator=generator).to(device=device, dtype=torch.bfloat16)

        losses: list[torch.Tensor] | None = [] if pipeline.info.has_last_stage else None
        output = pipeline.step(
            input_ids,
            target=upstream,
            losses=losses,
            attention_mask=attention_mask,
        )

        reference_output = reference(input_ids, attention_mask=attention_mask)
        reference_loss = loss_fn(reference_output, upstream)
        reference_loss.backward()

        if pipeline.info.has_last_stage:
            assert output is not None and losses is not None
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, reference_output, rtol=0.02, atol=0.02)
            torch.testing.assert_close(torch.stack(losses).sum(), reference_loss, rtol=0.01, atol=0.01)
        else:
            assert output is None

        reference_parameters = dict(reference.named_parameters())
        local_parameters = dict(pipeline.parts[0].named_parameters())
        assert local_parameters
        compared_gradients = 0
        for name, parameter in local_parameters.items():
            expected = reference_parameters[name]
            if expected.grad is None:
                assert parameter.grad is None, f"[rank {rank}] unexpected gradient for {name}"
                continue
            assert parameter.grad is not None, f"[rank {rank}] missing gradient for {name}"
            assert torch.isfinite(parameter.grad).all(), f"[rank {rank}] non-finite gradient for {name}"
            torch.testing.assert_close(
                parameter.grad.float(),
                expected.grad.float(),
                rtol=0.04,
                atol=0.02,
                msg=lambda message: f"[rank {rank}] {name}: {message}",
            )
            compared_gradients += 1
        assert compared_gradients > 0

        if rank == 0:
            print(_PP_RESULT_PREFIX + "PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    _PP_UNAVAILABLE_REASON is not None,
    reason=_PP_UNAVAILABLE_REASON or "two-rank MiniMax M3 MSA PP prerequisites are available",
)
def test_two_rank_dense_to_msa_pipeline_forward_backward_parity() -> None:
    """Real PP=2 output, loss, and local parameter gradients match one-process execution."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_devices = visible_devices.split(",")[:2] if visible_devices else ["0", "1"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
    repo_root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--pp-worker",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert _PP_RESULT_PREFIX + "PASS" in completed.stdout, completed.stdout


if __name__ == "__main__" and "--pp-worker" in sys.argv:
    _run_pp_worker()
