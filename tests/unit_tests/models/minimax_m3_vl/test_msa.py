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

"""CPU contracts for MiniMax M3 MSA planning, policy, and Adapter state."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import TEFp8Config
from nemo_automodel.components.models.minimax_m3_vl import msa
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    _build_backward_tasks,
    _chunk_map,
    _cta_row_interval,
    _MSABackwardSchedule,
)
from nemo_automodel.components.models.minimax_m3_vl.msa import (
    _MSAFlatAttention,
    _MSAForwardKernels,
    _MSASparseAttentionFunction,
    _reject_unsupported_msa_configuration,
    _reject_unsupported_msa_runtime,
    _validate_msa_topology,
)
from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSAPackedLayout, _resolve_canonical_document_map
from nemo_automodel.shared.import_utils import UnavailableError

_FIXED_TOPOLOGY = {
    "num_heads": 64,
    "num_kv_heads": 4,
    "head_dim": 128,
    "num_index_heads": 4,
    "block_size": 128,
    "topk_blocks": 16,
    "attention_dropout": 0.0,
}

_UNSUPPORTED_RUNTIME_CASES = [
    pytest.param({"qkv_format": "thd"}, False, "BSHD", id="thd"),
    pytest.param({"use_cache": True}, False, "cache-free prefill", id="use-cache"),
    pytest.param({"past_key_values": ()}, False, "past_key_values", id="past-key-values"),
    pytest.param({"cache_position": torch.arange(2)}, False, "cache_position", id="cache-position"),
    pytest.param({"page_table": torch.zeros(1, 1, dtype=torch.int32)}, False, "page_table", id="page-table"),
    pytest.param({"seqused_k": torch.ones(1, dtype=torch.int32)}, False, "seqused_k", id="seqused-k"),
    pytest.param({"prefix_cache": object()}, False, "prefix_cache", id="prefix-cache"),
    pytest.param({"is_causal": False}, False, "causal self-attention", id="noncausal"),
    pytest.param({"window_size": (128, 0)}, False, "sliding window", id="windowed"),
    pytest.param({"encoder_hidden_states": torch.zeros(1, 2, 4)}, False, "self-attention", id="encoder"),
    pytest.param({"key_value_states": torch.zeros(1, 2, 4)}, False, "cross-attention", id="key-value"),
    pytest.param({}, True, "cp_size=1", id="context-parallel"),
]


def _msa_backend(*, attn: str = "sdpa") -> BackendConfig:
    """Build the CPU backend used at the model-owned MSA seam."""
    return BackendConfig(
        attn=attn,
        sparse_attn="msa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _block_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build an independent same-document causal mask.

    Args:
        doc_ids: Integer tensor of shape [batch, sequence], with 0 for padding.

    Returns:
        Bool tensor of shape [batch, 1, sequence, sequence].
    """
    real = doc_ids > 0
    same_document = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.ones(doc_ids.shape[-1], doc_ids.shape[-1], dtype=torch.bool, device=doc_ids.device).tril()
    return (real.unsqueeze(-1) & real.unsqueeze(-2) & same_document & causal).unsqueeze(1)


def test_packed_layout_maps_adversarial_documents_and_gradients() -> None:
    """Exercise residues, padding locations, batch isolation, and autograd once."""
    doc_ids = torch.zeros(3, 262, dtype=torch.int64)
    doc_ids[0, 1:128] = 42
    doc_ids[0, 130:259] = 7
    doc_ids[1, :128] = 7
    doc_ids[1, 128] = 9
    layout = _MSAPackedLayout.build(doc_ids)
    external = torch.randn(3, 262, 3, requires_grad=True)
    upstream = torch.randn_like(external)

    packed = layout.pack(external)
    restored = layout.unpack(packed)
    metadata = layout.launch_metadata()
    assert packed.shape == (385, 3)
    assert torch.equal(packed, external[doc_ids > 0])
    assert torch.equal(restored[doc_ids > 0], external[doc_ids > 0])
    assert torch.count_nonzero(restored[doc_ids == 0]) == 0
    assert layout.has_padding is True
    assert layout.has_multiple_documents_per_row is True
    assert (metadata.total_tokens, metadata.workspace_size, metadata.max_seqlen) == (385, 640, 129)
    assert metadata.cu_seqlens.tolist() == [0, 127, 256, 384, 385]
    assert metadata.document_workspace_starts.tolist() == [0, 128, 384, 512]
    assert metadata.workspace_positions[[0, 126, 127, 255, 256, 383, 384]].tolist() == [
        0,
        126,
        128,
        256,
        384,
        511,
        512,
    ]

    restored.backward(upstream)
    assert torch.equal(external.grad[doc_ids > 0], upstream[doc_ids > 0])
    assert torch.count_nonzero(external.grad[doc_ids == 0]) == 0


@pytest.mark.parametrize(
    ("packed_seq_ids", "attention_mask", "padding_mask", "expected"),
    [
        pytest.param(
            torch.tensor([[9, 9, 4, 4, 0]], dtype=torch.int32),
            torch.ones(1, 5, dtype=torch.bool),
            torch.ones(1, 5, dtype=torch.bool),
            torch.tensor([[9, 9, 4, 4, 0]]),
            id="packed-ids-win",
        ),
        pytest.param(
            None,
            torch.tensor([[3, 3, 8, 8, 0]], dtype=torch.int32),
            None,
            torch.tensor([[3, 3, 8, 8, 0]]),
            id="indexed-mask",
        ),
        pytest.param(
            None,
            torch.tensor([[True, True, False, True, False]]),
            None,
            torch.tensor([[1, 1, 0, 1, 0]]),
            id="keep-mask",
        ),
        pytest.param(
            None,
            _block_causal_mask(torch.tensor([[1, 1, 0, 2, 2]])),
            None,
            torch.tensor([[1, 1, 0, 2, 2]]),
            id="block-causal-mask",
        ),
        pytest.param(
            None,
            None,
            torch.tensor([[False, False, True, False, True]]),
            torch.tensor([[1, 1, 0, 1, 0]]),
            id="padding-mask",
        ),
        pytest.param(None, None, None, torch.ones(1, 5, dtype=torch.int64), id="single-document"),
    ],
)
def test_document_map_source_precedence(
    packed_seq_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    expected: torch.Tensor,
) -> None:
    """Recover one canonical document map from the supported metadata sources.

    Args:
        packed_seq_ids: Optional integer tensor of shape [batch, sequence].
        attention_mask: Optional tensor of shape [batch, sequence] or [batch, 1, sequence, sequence].
        padding_mask: Optional tensor of shape [batch, sequence], true for padding.
        expected: Expected int64 tensor of shape [batch, sequence].
    """
    recovered = _resolve_canonical_document_map(
        torch.empty(1, 5, 8),
        packed_seq_ids=packed_seq_ids,
        attention_mask=attention_mask,
        padding_mask=padding_mask,
    )

    assert recovered.dtype == torch.int64
    assert recovered.is_contiguous()
    assert torch.equal(recovered, expected)


@pytest.mark.parametrize(
    ("doc_ids", "match"),
    [
        pytest.param(torch.ones(4, dtype=torch.int64), r"\[batch, sequence\]", id="rank"),
        pytest.param(torch.ones(1, 4), "integer tensor", id="dtype"),
        pytest.param(torch.tensor([[1, -1, 1]]), "non-negative", id="negative"),
        pytest.param(torch.zeros(1, 4, dtype=torch.int64), "at least one real token", id="all-padding"),
        pytest.param(torch.tensor([[1, 0, 1]]), "contiguous run", id="resumed-document"),
    ],
)
def test_packed_layout_rejects_invalid_document_maps(doc_ids: torch.Tensor, match: str) -> None:
    """Reject one representative for every canonical-map invariant.

    Args:
        doc_ids: Candidate tensor whose required shape is [batch, sequence].
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _MSAPackedLayout.build(doc_ids)


@pytest.mark.parametrize(
    ("attention_mask", "match"),
    [
        pytest.param(torch.ones(1, 4), "integer or bool 2-D", id="float-2d"),
        pytest.param(torch.ones(1, 2, 4, 4, dtype=torch.bool), "only a bool 4-D", id="heads-4d"),
        pytest.param(torch.ones(1, 4, 4, dtype=torch.bool), "must have shape", id="rank-3"),
        pytest.param(
            torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "standard bool block-causal",
            id="noncausal-4d",
        ),
    ],
)
def test_document_map_rejects_ambiguous_attention_masks(attention_mask: torch.Tensor, match: str) -> None:
    """Reject ambiguous mask tensors before document recovery.

    Args:
        attention_mask: Candidate mask whose supported layouts are [batch, sequence] and [batch, 1, sequence, sequence].
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        _resolve_canonical_document_map(
            torch.empty(1, 4, 8),
            packed_seq_ids=None,
            attention_mask=attention_mask,
            padding_mask=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (key, value)
        for key, value in {
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
            "num_index_heads": 2,
            "block_size": 64,
            "topk_blocks": 8,
            "attention_dropout": 0.1,
        }.items()
    ],
)
def test_msa_topology_rejects_every_fixed_dimension(field: str, value: int | float) -> None:
    invalid = dict(_FIXED_TOPOLOGY)
    invalid[field] = value

    with pytest.raises(ValueError, match="requires|supports"):
        _validate_msa_topology(**invalid)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("rope_fusion", True, "rope_fusion=False", id="fused-rope"),
        pytest.param("te_fp8", TEFp8Config(), "te_fp8=None", id="fp8"),
    ],
)
def test_msa_configuration_rejects_unsupported_backends(field: str, value: object, match: str) -> None:
    backend = _msa_backend()
    if not hasattr(backend, field):
        raise ValueError(f"BackendConfig has no field {field!r}")
    setattr(backend, field, value)

    with pytest.raises(NotImplementedError, match=match):
        _reject_unsupported_msa_configuration(backend)


@pytest.mark.parametrize(("runtime_kwargs", "cp_enabled", "match"), _UNSUPPORTED_RUNTIME_CASES)
def test_msa_runtime_policy_rejects_unsupported_modes(
    runtime_kwargs: dict[str, object],
    cp_enabled: bool,
    match: str,
) -> None:
    with pytest.raises(NotImplementedError, match=match):
        _reject_unsupported_msa_runtime(runtime_kwargs, cp_enabled=cp_enabled)


def test_msa_runtime_policy_rejects_cuda_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(NotImplementedError, match="CUDA graph capture"):
        _reject_unsupported_msa_runtime({})


def test_flat_attention_rejects_deterministic_algorithms() -> None:
    """Reject the atomic backward when deterministic algorithms are required."""
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    layout = _MSAPackedLayout.build(torch.ones(1, 1, dtype=torch.int64))
    placeholder = torch.empty(0)
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(NotImplementedError, match="not bitwise deterministic"):
            _MSAFlatAttention(0.125)(placeholder, placeholder, placeholder, placeholder, layout=layout)
    finally:
        torch.use_deterministic_algorithms(deterministic_enabled, warn_only=deterministic_warn_only)


def test_optional_dependency_failures_are_deferred_and_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe() -> None:
        raise AssertionError("MSA dependencies must not be resolved during construction")

    monkeypatch.setattr(msa, "_resolve_msa_forward", unexpected_probe)
    monkeypatch.setattr(msa, "_resolve_msa_backward", unexpected_probe)
    _MSAFlatAttention(0.125)

    monkeypatch.setattr(msa, "_resolve_msa_forward", lambda: None)
    monkeypatch.setattr(msa, "_resolve_msa_backward", lambda: None)
    with pytest.raises(UnavailableError, match=r"uv sync --extra msa"):
        msa._require_msa()
    with pytest.raises(UnavailableError, match=r"uv sync --extra msa"):
        msa._require_msa_backward()


def test_custom_autograd_reuses_forward_schedule_and_returns_compact_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_queries = 3
    row_ptr = torch.tensor([[0, 3], [0, 3], [0, 3], [0, 3]], dtype=torch.int32)
    q_indices = torch.zeros((4, 16), dtype=torch.int32)
    q_indices[:, :total_queries] = torch.arange(total_queries, dtype=torch.int32)
    scheduler_metadata = torch.tensor(
        [[head, 0, 0, total_queries, 0, 0] for head in range(4)],
        dtype=torch.int32,
    )
    work_count = torch.tensor([4], dtype=torch.int32)
    forward_schedule = SimpleNamespace(scheduler_metadata=scheduler_metadata, work_count=work_count)
    captured: dict[str, _MSABackwardSchedule] = {}

    def fake_build_k2q_csr(
        q2k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        block_size: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Return forward metadata for compact support.

        Args:
            q2k: Int32 support tensor of shape [4, tokens, 16].
            cu_seqlens_q: Int32 query offsets of shape [documents + 1].
            cu_seqlens_k: Int32 key offsets of shape [documents + 1].
            block_size: Key-block width in tokens.
            **kwargs: Remaining schedule options.

        Returns:
            CSR tensors and forward execution metadata.
        """
        del q2k, cu_seqlens_q, cu_seqlens_k, block_size, kwargs
        return row_ptr, q_indices, forward_schedule

    def fake_sparse_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        topk_blocks: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shape-correct flat output and LSE.

        Args:
            q: BF16 tensor of shape [tokens, 64, 128].
            k: BF16 tensor of shape [tokens, 4, 128].
            v: BF16 tensor of shape [tokens, 4, 128].
            k2q_row_ptr: Int32 CSR offsets of shape [4, rows + 1].
            k2q_q_indices: Int32 CSR query rows of shape [4, edge_capacity].
            topk_blocks: Fixed support width in key blocks.
            **kwargs: Remaining launch metadata.

        Returns:
            BF16 output of shape [tokens, 64, 128] and FP32 LSE of shape [tokens, 64].
        """
        del k, v, k2q_row_ptr, k2q_q_indices, topk_blocks, kwargs
        return q.clone(), torch.zeros((q.shape[0], q.shape[1]), dtype=torch.float32)

    def fake_backward(
        q: torch.Tensor,
        k_aligned: torch.Tensor,
        v_aligned: torch.Tensor,
        grad_out: torch.Tensor,
        lse: torch.Tensor,
        out: torch.Tensor,
        schedule: _MSABackwardSchedule,
        *,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Capture reused schedule state and return workspace gradients.

        Args:
            q: BF16 tensor of shape [tokens, 64, 128].
            k_aligned: BF16 tensor of shape [workspace, 4, 128].
            v_aligned: BF16 tensor of shape [workspace, 4, 128].
            grad_out: BF16 tensor of shape [tokens, 64, 128].
            lse: FP32 tensor of shape [tokens, 64].
            out: BF16 tensor of shape [tokens, 64, 128].
            schedule: Forward-derived backward execution metadata.
            softmax_scale: QK scale.

        Returns:
            Compact dQ and aligned-workspace dK/dV tensors.
        """
        del lse, out, softmax_scale
        captured["schedule"] = schedule
        assert q.shape == grad_out.shape == (total_queries, 64, 128)
        assert k_aligned.shape == v_aligned.shape == (128, 4, 128)
        assert torch.count_nonzero(k_aligned[total_queries:]) == 0
        assert torch.count_nonzero(v_aligned[total_queries:]) == 0
        return grad_out.clone(), torch.ones_like(k_aligned), torch.full_like(v_aligned, 2)

    monkeypatch.setattr(
        msa,
        "_require_msa",
        lambda: _MSAForwardKernels(
            build_k2q_csr=fake_build_k2q_csr,
            sparse_atten_func=fake_sparse_attention,
        ),
    )
    monkeypatch.setattr(msa, "_require_msa_backward", lambda: fake_backward)

    q = torch.randn(total_queries, 64, 128, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total_queries, 4, 128, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total_queries, 4, 128, dtype=torch.bfloat16, requires_grad=True)
    metadata = _MSAPackedLayout.build(torch.ones(1, total_queries, dtype=torch.int64)).launch_metadata()
    q2k = torch.full((4, total_queries, 16), -1, dtype=torch.int32)
    q2k[:, :, 0] = 0
    out = _MSASparseAttentionFunction.apply(q, k, v, q2k, metadata, 0.125)
    upstream = torch.randn_like(out)

    out.backward(upstream)

    saved_schedule = captured["schedule"]
    assert torch.equal(saved_schedule.row_ptr, row_ptr)
    assert torch.equal(saved_schedule.q_indices, q_indices)
    assert torch.equal(saved_schedule.scheduler_metadata, scheduler_metadata)
    assert torch.equal(saved_schedule.work_count, work_count)
    assert torch.equal(q.grad, upstream)
    assert torch.equal(k.grad, torch.ones_like(k))
    assert torch.equal(v.grad, torch.full_like(v, 2))


def _forward_schedule_fixture() -> _MSABackwardSchedule:
    """Build forward execution metadata with document and capacity tails."""
    row_ptr = torch.zeros((4, 3), dtype=torch.int32)
    row_ptr[0] = torch.tensor([0, 10, 10], dtype=torch.int32)
    row_ptr[3] = torch.tensor([0, 2, 2], dtype=torch.int32)
    q_indices = torch.full((4, 16), -1, dtype=torch.int32)
    q_indices[0, :10] = torch.arange(10, dtype=torch.int32)
    q_indices[3, :2] = torch.tensor([5, 6], dtype=torch.int32)
    scheduler_metadata = torch.full((4, 6), torch.iinfo(torch.int32).max, dtype=torch.int32)
    scheduler_metadata[:2] = torch.tensor(
        [[0, 0, 0, 10, 1, 0], [3, 0, 0, 2, 0, 1]],
        dtype=torch.int32,
    )
    return _MSABackwardSchedule(
        row_ptr=row_ptr,
        q_indices=q_indices,
        scheduler_metadata=scheduler_metadata,
        work_count=torch.tensor([2], dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 130, 142], dtype=torch.int32),
        document_workspace_starts=torch.tensor([0, 256], dtype=torch.int32),
    )


def _task_edges(
    task_meta: torch.Tensor,
    task_qrows: torch.Tensor,
    task_qpos: torch.Tensor,
) -> Counter[tuple[int, int, int, int]]:
    """Decode the semantic edges carried by backward tasks.

    Args:
        task_meta: Int32 tensor of shape [tasks, 4].
        task_qrows: Int32 compact query rows of shape [tasks, 8].
        task_qpos: Int32 aligned query positions of shape [tasks, 8].

    Returns:
        Multiplicity of (index_head, workspace_key_block, compact_query, aligned_query) edges.
    """
    edges: Counter[tuple[int, int, int, int]] = Counter()
    for meta, rows, positions in zip(task_meta.tolist(), task_qrows.tolist(), task_qpos.tolist(), strict=True):
        _, head, key_block, valid = meta
        edges.update((head, key_block, row, position) for row, position in zip(rows[:valid], positions[:valid]))
    return edges


def test_forward_schedule_tasks_cover_each_edge_once_and_ignore_capacity() -> None:
    task_meta, task_qrows, task_qpos = _build_backward_tasks(_forward_schedule_fixture())
    expected = Counter(
        [(0, 2, 130 + offset, 256 + offset) for offset in range(10)]
        + [(3, 1, 5 + offset, 5 + offset) for offset in range(2)]
    )

    assert task_meta.dtype == task_qrows.dtype == task_qpos.dtype == torch.int32
    assert task_meta.shape == (3, 4)
    assert task_qrows.shape == task_qpos.shape == (3, 8)
    assert _task_edges(task_meta, task_qrows, task_qpos) == expected
    for meta, rows, positions in zip(task_meta, task_qrows, task_qpos, strict=True):
        valid = int(meta[-1])
        assert torch.all(rows[valid:] == -1)
        assert torch.all(positions[valid:] == -1)


def _assert_exact_chunk_cover(num_rows: int, rows_per_cta: int, num_sms: int) -> None:
    """Assert that CTA intervals partition ``range(num_rows)`` exactly once."""
    num_full_ctas, tail_rows, grid_ctas = _chunk_map(num_rows, rows_per_cta, num_sms)
    intervals = sorted(
        _cta_row_interval(block, num_rows, rows_per_cta, num_full_ctas, tail_rows) for block in range(grid_ctas)
    )

    assert intervals[0][0] == 0
    assert intervals[-1][1] == num_rows
    assert all(0 <= start < end <= num_rows for start, end in intervals)
    assert all(previous_end == next_start for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]))


@pytest.mark.parametrize("num_sms", [132, 148])
@pytest.mark.parametrize("rows_per_cta", [4, 8])
def test_chunk_map_covers_every_row_exactly_once(num_sms: int, rows_per_cta: int) -> None:
    row_counts = [*range(1, 1025), *range(1025, 20001, 97), 1177, 3545]
    for num_rows in row_counts:
        _assert_exact_chunk_cover(num_rows, rows_per_cta, num_sms)
