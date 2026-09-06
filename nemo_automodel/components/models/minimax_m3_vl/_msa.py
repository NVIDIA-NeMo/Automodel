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

"""Packed MSA training: compact forward, saved schedule, backward-only aligned K/V."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_patch import _patch_msa_fmax
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import _MSABackwardSchedule
from nemo_automodel.shared.import_utils import UnavailableError, safe_import, safe_import_from

_MSA_BLOCK_SIZE = 128
_MSA_TOPK_BLOCKS = 16
_MSA_QUERY_HEADS = 64
_MSA_KV_HEADS = 4
_MSA_INDEX_HEADS = 4
_MSA_HEAD_DIM = 128
_MSA_ATTENTION_DROPOUT = 0.0

_MSA_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' requires the fixed fmha-sm100 optional dependency. "
    "Install the project with uv sync --extra msa on a CUDA SM100 system; the MSA revision "
    "must be compatible with nvidia-cutlass-dsl==4.6.2."
)
_MSA_BACKWARD_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' backward requires nvidia-cutlass-dsl==4.6.2 and cuda-python from the "
    "msa optional dependency. Install the project with uv sync --extra msa on a CUDA SM100 system."
)

_MSA_BACKWARD_MODULE = "nemo_automodel.components.models.minimax_m3_vl.kernels.msa_backward_sm100"


def _check_document_runs(
    doc_ids: torch.Tensor,
    batch_rows: torch.Tensor,
    is_real: torch.Tensor,
    external_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check flat int64 doc_ids/batch_rows/external_rows[N] and bool is_real[N]; return valid/bad-row scalars."""
    num_external_tokens = external_rows.numel()
    if num_external_tokens < 2:
        valid = torch.ones((), dtype=torch.bool, device=external_rows.device)
        first_bad = torch.full((), -1, dtype=torch.int64, device=external_rows.device)
        return valid, first_bad

    # Distinct negative padding keys cannot form documents or collide with positive ids.
    document_sort_key = torch.where(is_real, doc_ids, -(external_rows + 1))
    order = torch.argsort(document_sort_key, stable=True)
    order = order[torch.argsort(batch_rows[order], stable=True)]
    sorted_batch_rows = batch_rows[order]
    sorted_document_keys = document_sort_key[order]
    sorted_external_rows = external_rows[order]

    same_document = (sorted_batch_rows[1:] == sorted_batch_rows[:-1]) & (
        sorted_document_keys[1:] == sorted_document_keys[:-1]
    )
    interrupted = same_document & (sorted_external_rows[1:] != sorted_external_rows[:-1] + 1)
    sentinel = torch.full_like(sorted_external_rows[1:], num_external_tokens)
    first_bad = torch.where(interrupted, sorted_external_rows[1:], sentinel).min()
    first_bad = torch.where(
        first_bad == num_external_tokens,
        torch.full_like(first_bad, -1),
        first_bad,
    )
    return ~interrupted.any(), first_bad


def _resolve_canonical_document_map(
    reference: torch.Tensor,
    *,
    packed_seq_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Recover int64 ids[B,S] from reference[B,S,D], packed ids/masks[B,S] or bool causal mask[B,1,S,S]."""
    if reference.dim() != 3:
        raise NotImplementedError(f"MSA requires BSHD hidden states [batch, sequence, hidden], got {reference.shape}")
    batch_size, sequence_length = reference.shape[:2]
    expected_shape = (batch_size, sequence_length)

    # Dense layers still consume attention_mask when packed ids are authoritative.
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask must be a tensor, got {type(attention_mask).__name__}")
        if attention_mask.dim() == 2:
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(f"2-D attention_mask must have shape {expected_shape}, got {attention_mask.shape}")
            if attention_mask.dtype.is_floating_point or attention_mask.dtype.is_complex:
                raise ValueError(f"MSA requires an integer or bool 2-D attention_mask, got {attention_mask.dtype}.")
        elif attention_mask.dim() == 4:
            expected_4d_shape = (batch_size, 1, sequence_length, sequence_length)
            if tuple(attention_mask.shape) != expected_4d_shape or attention_mask.dtype != torch.bool:
                raise ValueError(
                    f"MSA requires a bool 4-D block-causal attention_mask {expected_4d_shape}; "
                    f"got {attention_mask.shape}/{attention_mask.dtype}."
                )
            if attention_mask.device != reference.device:
                raise ValueError(
                    f"4-D attention_mask must be on the hidden-state device {reference.device}, "
                    f"got {attention_mask.device}"
                )
        else:
            raise ValueError(
                "MiniMax M3 MSA attention_mask must have shape [batch, sequence] or "
                f"[batch, 1, sequence, sequence], got {tuple(attention_mask.shape)}"
            )

    if packed_seq_ids is not None:
        if not isinstance(packed_seq_ids, torch.Tensor):
            raise TypeError(f"_packed_seq_ids must be a tensor, got {type(packed_seq_ids).__name__}")
        if tuple(packed_seq_ids.shape) != expected_shape:
            raise ValueError(
                f"_packed_seq_ids must have shape [batch, sequence]={expected_shape}, got {tuple(packed_seq_ids.shape)}"
            )
        if (
            packed_seq_ids.dtype == torch.bool
            or packed_seq_ids.dtype.is_floating_point
            or packed_seq_ids.dtype.is_complex
        ):
            raise ValueError(f"_packed_seq_ids must be an integer tensor, got dtype {packed_seq_ids.dtype}")
        return packed_seq_ids.to(device=reference.device, dtype=torch.int64).contiguous()

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            return attention_mask.to(device=reference.device, dtype=torch.int64).contiguous()
        if attention_mask.dim() == 4:
            block_causal = attention_mask[:, 0]
            real_tokens = torch.diagonal(block_causal, dim1=-2, dim2=-1)
            previous_token_visible = torch.diagonal(block_causal, offset=-1, dim1=-2, dim2=-1)
            document_starts = torch.cat(
                (real_tokens[:, :1], real_tokens[:, 1:] & ~previous_token_visible),
                dim=-1,
            )
            document_ids = document_starts.cumsum(dim=-1, dtype=torch.int64) * real_tokens
            key_positions = torch.arange(sequence_length, device=reference.device)
            is_standard_block_causal = torch.ones((), dtype=torch.bool, device=reference.device)
            for query_start in range(0, sequence_length, 256):
                query_end = min(query_start + 256, sequence_length)
                query_ids = document_ids[:, query_start:query_end]
                expected = (
                    (query_ids.unsqueeze(-1) > 0)
                    & (query_ids.unsqueeze(-1) == document_ids.unsqueeze(1))
                    & (key_positions.view(1, 1, -1) <= key_positions[query_start:query_end].view(1, -1, 1))
                )
                is_standard_block_causal &= (block_causal[:, query_start:query_end] == expected).all()
            if not bool(is_standard_block_causal.item()):
                raise ValueError(
                    "MiniMax M3 MSA requires a standard bool block-causal attention_mask: each real query "
                    "must keep exactly the causal keys from its own contiguous document, with padding rows false."
                )
            return document_ids.contiguous()

    if padding_mask is not None:
        if not isinstance(padding_mask, torch.Tensor):
            raise TypeError(f"padding_mask must be a tensor, got {type(padding_mask).__name__}")
        if tuple(padding_mask.shape) != expected_shape:
            raise ValueError(f"padding_mask must have shape {expected_shape}, got {padding_mask.shape}")
        if padding_mask.dtype.is_complex:
            raise ValueError(f"padding_mask cannot have complex dtype {padding_mask.dtype}")
        return (~padding_mask.to(device=reference.device).bool()).to(torch.int64).contiguous()

    return torch.ones(expected_shape, dtype=torch.int64, device=reference.device)


@dataclass(frozen=True, slots=True)
class _MSAPackedLayout:
    """Own compact token rows and aligned workspace coordinates for one BSHD forward/stage."""

    _token_rows: torch.Tensor  # Int64 [T] external row indices; T is the real-token count.
    _workspace_positions: torch.Tensor  # Int64 [T] compact-to-aligned row indices, on the input device.
    _query_doc_starts: torch.Tensor  # Int64 [T] aligned start of each query's document.
    _document_workspace_starts: torch.Tensor  # Int32 [documents] aligned document starts.
    _cu_seqlens: torch.Tensor  # Contiguous int32 [documents + 1] compact document offsets.
    _external_shape: tuple[int, int]  # Batch and sequence dimensions.
    _workspace_size: int  # W: positive multiple of 128, including each document's alignment tail.
    _max_seqlen: int  # Longest real document.
    has_multiple_documents_per_row: bool

    @classmethod
    def validate(cls, doc_ids: torch.Tensor) -> tuple[bool, bool]:
        """Validate integer doc_ids[B,S] (0 padding); return has-padding and has-multiple-documents flags."""
        _, has_padding, has_multiple_documents = cls._prepare(doc_ids, materialize=False)
        return has_padding, has_multiple_documents

    @classmethod
    def build(cls, doc_ids: torch.Tensor) -> "_MSAPackedLayout":
        """Build a layout from integer doc_ids[B,S] with contiguous positive documents and 0 padding."""
        layout, _, _ = cls._prepare(doc_ids, materialize=True)
        assert layout is not None
        return layout

    @classmethod
    def _prepare(cls, doc_ids: torch.Tensor, *, materialize: bool) -> tuple["_MSAPackedLayout | None", bool, bool]:
        """Probe integer doc_ids[B,S] once; return optional layout, has-padding and has-multiple-documents."""
        if doc_ids.dim() != 2:
            raise ValueError(f"doc_ids must have shape [batch, sequence], got {tuple(doc_ids.shape)}")
        if doc_ids.dtype == torch.bool or doc_ids.dtype.is_floating_point or doc_ids.dtype.is_complex:
            raise ValueError(f"doc_ids must be an integer tensor, got dtype {doc_ids.dtype}")
        batch_size, sequence_length = doc_ids.shape
        num_external_tokens = doc_ids.numel()
        if num_external_tokens == 0:
            raise ValueError(f"doc_ids must be non-empty, got shape {tuple(doc_ids.shape)}")

        device = doc_ids.device
        ids = doc_ids.reshape(-1).to(torch.int64)
        external_rows = torch.arange(num_external_tokens, device=device, dtype=torch.int64)
        batch_rows = torch.div(external_rows, sequence_length, rounding_mode="floor")
        is_real = ids > 0

        previous_ids = torch.cat((ids.new_full((1,), -1), ids[:-1]))
        previous_batch_rows = torch.cat((batch_rows.new_full((1,), -1), batch_rows[:-1]))
        next_ids = torch.cat((ids[1:], ids.new_full((1,), -1)))
        next_batch_rows = torch.cat((batch_rows[1:], batch_rows.new_full((1,), -1)))
        is_run_start = is_real & ((ids != previous_ids) | (batch_rows != previous_batch_rows))
        is_run_end = is_real & ((ids != next_ids) | (batch_rows != next_batch_rows))

        run_start = torch.where(is_run_start, external_rows, torch.full_like(external_rows, -1)).cummax(0).values
        run_end = (
            torch.where(is_run_end, external_rows, torch.full_like(external_rows, num_external_tokens))
            .flip(0)
            .cummin(0)
            .values.flip(0)
        )
        run_length = run_end - run_start + 1
        aligned_run_length = torch.where(
            is_real,
            ((run_length + _MSA_BLOCK_SIZE - 1) // _MSA_BLOCK_SIZE) * _MSA_BLOCK_SIZE,
            torch.zeros_like(run_length),
        )
        aligned_lengths_at_starts = torch.where(is_run_start, aligned_run_length, torch.zeros_like(run_length))
        aligned_prefix = aligned_lengths_at_starts.cumsum(0)

        runs_are_valid, first_bad = _check_document_runs(ids, batch_rows, is_real, external_rows)
        document_lengths_at_starts = torch.where(is_run_start, run_length, torch.zeros_like(run_length))
        documents_per_batch_row = torch.zeros(batch_size, dtype=torch.int64, device=device)
        documents_per_batch_row.scatter_add_(0, batch_rows, is_run_start.to(torch.int64))

        probe = torch.stack(
            (
                is_real.sum(dtype=torch.int64),
                is_run_start.sum(dtype=torch.int64),
                aligned_prefix[-1],
                document_lengths_at_starts.max(),
                (ids >= 0).all().to(torch.int64),
                runs_are_valid.to(torch.int64),
                first_bad,
                documents_per_batch_row.max(),
            )
        )
        (
            num_real_tokens,
            num_documents,
            workspace_size,
            max_seqlen,
            ids_are_valid,
            structure_is_valid,
            bad_external_row,
            max_documents_per_row,
        ) = probe.tolist()  # The single device-to-host synchronization.

        if not ids_are_valid:
            raise ValueError("doc_ids must be non-negative (0 = padding, positive = document id)")
        if num_real_tokens == 0:
            raise ValueError("doc_ids must contain at least one real token (a positive document id)")
        if not structure_is_valid:
            raise ValueError(
                "doc_ids must give each document one contiguous run of tokens; the document at flat token "
                f"index {bad_external_row} resumes after an interruption"
            )
        int32_max = torch.iinfo(torch.int32).max
        if num_real_tokens > int32_max or workspace_size > int32_max or max_seqlen > int32_max:
            raise ValueError(
                "MSA document coordinates must fit int32, got "
                f"tokens={num_real_tokens}, workspace_size={workspace_size}, max_seqlen={max_seqlen}"
            )

        has_padding = num_real_tokens != num_external_tokens
        has_multiple_documents = max_documents_per_row > 1
        if not materialize:
            return None, has_padding, has_multiple_documents

        aligned_start = aligned_prefix - aligned_run_length
        workspace_row = aligned_start + (external_rows - run_start)

        # Prefix ranks permit a fixed-shape scatter; clone token rows before reusing the partition buffer.
        partitioned_rows = torch.empty_like(external_rows)
        real_rank = is_real.cumsum(0, dtype=torch.int64) - 1
        padding_rank = (~is_real).cumsum(0, dtype=torch.int64) + num_real_tokens - 1
        partition_destinations = torch.where(is_real, real_rank, padding_rank)
        partitioned_rows.scatter_(0, partition_destinations, external_rows)
        token_rows = partitioned_rows[:num_real_tokens].clone()

        run_rank = is_run_start.cumsum(0, dtype=torch.int64) - 1
        non_run_rank = (~is_run_start).cumsum(0, dtype=torch.int64) + num_documents - 1
        partition_destinations = torch.where(is_run_start, run_rank, non_run_rank)
        partitioned_rows.scatter_(0, partition_destinations, external_rows)
        run_rows = partitioned_rows[:num_documents]

        workspace_positions = workspace_row[token_rows].contiguous()
        query_doc_starts = aligned_start[token_rows].contiguous()
        document_workspace_starts = aligned_start[run_rows].to(torch.int32).contiguous()
        document_lengths = run_length[run_rows].to(torch.int32)
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                document_lengths.cumsum(0, dtype=torch.int32),
            )
        ).contiguous()

        return (
            cls(
                _token_rows=token_rows,
                _workspace_positions=workspace_positions,
                _query_doc_starts=query_doc_starts,
                _document_workspace_starts=document_workspace_starts,
                _cu_seqlens=cu_seqlens,
                _external_shape=(batch_size, sequence_length),
                _workspace_size=workspace_size,
                _max_seqlen=max_seqlen,
                has_multiple_documents_per_row=has_multiple_documents,
            ),
            has_padding,
            has_multiple_documents,
        )

    @property
    def has_padding(self) -> bool:
        """Whether the external token grid contains padding rows."""
        return self._token_rows.numel() != self._external_shape[0] * self._external_shape[1]

    def pack(self, external: torch.Tensor) -> torch.Tensor:
        """Pack external[B,S,...] to [T,...] in document order; the no-padding result may alias external."""
        if external.dim() < 2 or tuple(external.shape[:2]) != self._external_shape:
            raise ValueError(
                f"external must start with [batch, sequence]={self._external_shape}, got shape {tuple(external.shape)}"
            )
        if external.device != self._token_rows.device:
            raise ValueError(f"layout is on {self._token_rows.device} but external is on {external.device}")

        flattened = external.reshape(self._external_shape[0] * self._external_shape[1], *external.shape[2:])
        if not self.has_padding:
            return flattened
        return flattened.index_select(0, self._token_rows)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Restore packed[T,...] to [B,S,...] with zero padding; the no-padding result may alias packed."""
        if packed.dim() < 1 or packed.shape[0] != self._token_rows.numel():
            raise ValueError(
                f"packed must have leading token size {self._token_rows.numel()}, got shape {tuple(packed.shape)}"
            )
        if packed.device != self._token_rows.device:
            raise ValueError(f"layout is on {self._token_rows.device} but packed is on {packed.device}")

        batch_size, sequence_length = self._external_shape
        if not self.has_padding:
            return packed.reshape(batch_size, sequence_length, *packed.shape[1:])
        restored = packed.new_zeros((batch_size * sequence_length, *packed.shape[1:]))
        restored = restored.index_copy(0, self._token_rows, packed)
        return restored.reshape(batch_size, sequence_length, *packed.shape[1:])

    def _selection_inputs(
        self,
        index_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map index_k[T,1,D] to aligned keys[1,W,1,D], query workspace rows[T] and document starts[T]."""
        expected_tokens = self._token_rows.numel()
        if index_k.dim() != 3 or index_k.shape[0] != expected_tokens or index_k.shape[1] != 1:
            raise ValueError(
                "index_k must have shape [tokens, 1, index_dim], got "
                f"{tuple(index_k.shape)} for {expected_tokens} packed tokens"
            )
        if index_k.device != self._workspace_positions.device:
            raise ValueError(f"layout is on {self._workspace_positions.device} but index_k is on {index_k.device}")

        if self._workspace_size == expected_tokens:
            aligned_index_k = index_k
        else:
            aligned_index_k = index_k.new_zeros((self._workspace_size, *index_k.shape[1:]))
            aligned_index_k = aligned_index_k.index_copy(0, self._workspace_positions, index_k)
        return aligned_index_k.unsqueeze(0), self._workspace_positions, self._query_doc_starts


@dataclass(frozen=True, slots=True)
class _MSAForwardKernels:
    """Optional CSR/schedule builder and flat SM100 forward launchers."""

    build_k2q_csr: Callable[..., Any]
    sparse_atten_func: Callable[..., Any]


# Cache success and absence; load CuTe only after runtime and tensor validation.
@lru_cache(maxsize=1)
def _resolve_msa_forward() -> _MSAForwardKernels | None:
    """Resolve the optional forward entry points once, or ``None`` if absent."""
    available, module = safe_import("fmha_sm100.sparse", msg=_MSA_IMPORT_ERROR)
    if not available:
        return None
    build_k2q_csr = getattr(module, "build_k2q_csr", None)
    sparse_atten_func = getattr(module, "sparse_atten_func", None)
    if not callable(build_k2q_csr) or not callable(sparse_atten_func):
        return None
    _patch_msa_fmax(module)
    return _MSAForwardKernels(build_k2q_csr=build_k2q_csr, sparse_atten_func=sparse_atten_func)


@lru_cache(maxsize=1)
def _resolve_msa_backward() -> Callable[..., Any] | None:
    """Resolve the model-private SM100 backward launcher once, or ``None``."""
    available, launcher = safe_import_from(
        _MSA_BACKWARD_MODULE,
        "_run_msa_backward",
        msg=_MSA_BACKWARD_IMPORT_ERROR,
    )
    return launcher if available and callable(launcher) else None


def _require_msa() -> _MSAForwardKernels:
    """Return cached forward launchers, raising UnavailableError when absent."""
    kernels = _resolve_msa_forward()
    if kernels is None:
        raise UnavailableError(_MSA_IMPORT_ERROR)
    return kernels


def _require_msa_backward() -> Callable[..., Any]:
    """Return the cached backward launcher, raising UnavailableError when absent."""
    launcher = _resolve_msa_backward()
    if launcher is None:
        raise UnavailableError(_MSA_BACKWARD_IMPORT_ERROR)
    return launcher


def _validate_msa_topology(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_index_heads: int,
    block_size: int,
    topk_blocks: int,
    attention_dropout: float,
) -> None:
    """Require the fixed 64-query/4-KV-head, 128-channel, top-16 MSA topology and zero dropout."""
    actual = (num_heads, num_kv_heads, head_dim, num_index_heads, block_size, topk_blocks)
    expected = (_MSA_QUERY_HEADS, _MSA_KV_HEADS, _MSA_HEAD_DIM, _MSA_INDEX_HEADS, _MSA_BLOCK_SIZE, _MSA_TOPK_BLOCKS)
    if actual != expected:
        raise ValueError(
            "MSA requires (num_heads, num_kv_heads, head_dim, num_index_heads, block_size, topk_blocks) "
            f"= {expected}; got {actual}."
        )
    if attention_dropout != _MSA_ATTENTION_DROPOUT:
        raise ValueError(f"MSA requires attention_dropout={_MSA_ATTENTION_DROPOUT:g}; got {attention_dropout}.")


def _validate_flat_msa_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    layout: _MSAPackedLayout,
    *,
    softmax_scale: float,
) -> None:
    """Check aligned BF16 q[T,64,128], k/v[T,4,128], int32 q2k[4,T,16] and layout on one SM100."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"MSA requires flat rank-3 q/k/v; got q={q.shape}, k={k.shape}, v={v.shape}.")

    total_tokens, query_heads, head_dim = q.shape
    expected_kv = (total_tokens, _MSA_KV_HEADS, _MSA_HEAD_DIM)
    if (query_heads, head_dim) != (_MSA_QUERY_HEADS, _MSA_HEAD_DIM):
        raise ValueError(f"MiniMax M3 MSA requires q[T,64,128], got q={tuple(q.shape)}.")
    if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
        raise ValueError(f"MSA requires matching k/v[T,4,128]={expected_kv}; got k={k.shape}, v={v.shape}.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"MiniMax M3 MSA first supports BF16 q/k/v only; got q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError(f"MSA requires CUDA q/k/v on one SM100; got q={q.device}, k={k.device}, v={v.device}.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(f"q/k/v must share one CUDA device, got q={q.device}, k={k.device}, v={v.device}.")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("MiniMax M3 MSA requires contiguous flat q/k/v tensors.")
    misaligned = [name for name, tensor in (("q", q), ("k", k), ("v", v)) if tensor.data_ptr() % 16 != 0]
    if misaligned:
        raise ValueError(f"MSA requires 16-byte-aligned q/k/v storage; misaligned tensors={misaligned}.")
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (10, 0):
        raise NotImplementedError(
            "MiniMax M3 MSA first supports SM100 (compute capability 10.0) only; got compute capability "
            f"{capability[0]}.{capability[1]} on {q.device}. Use sparse_attn='generic' on this GPU."
        )

    metadata_tensors = (
        layout._workspace_positions,
        layout._document_workspace_starts,
        layout._cu_seqlens,
    )
    if any(tensor.device != q.device for tensor in metadata_tensors):
        devices = tuple(tensor.device for tensor in metadata_tensors)
        raise ValueError(f"MSA packed-layout tensors must be on {q.device}, got devices={devices}.")
    if layout._token_rows.numel() != total_tokens:
        raise ValueError(f"MSA layout contains {layout._token_rows.numel()} tokens, but q/k/v contain {total_tokens}.")
    expected_q2k = (_MSA_KV_HEADS, total_tokens, _MSA_TOPK_BLOCKS)
    if tuple(q2k.shape) != expected_q2k:
        raise ValueError(f"q2k must have fixed layout [4,T,16], got {tuple(q2k.shape)}, expected={expected_q2k}.")
    if q2k.dtype != torch.int32 or not q2k.is_contiguous():
        raise ValueError(f"q2k must be contiguous int32, got dtype={q2k.dtype}, contiguous={q2k.is_contiguous()}.")
    if q2k.device != q.device:
        raise ValueError(f"q2k must be on {q.device}, got {q2k.device}.")
    metadata_and_support = (
        ("q2k", q2k),
        ("workspace_positions", layout._workspace_positions),
        ("document_workspace_starts", layout._document_workspace_starts),
        ("cu_seqlens", layout._cu_seqlens),
    )
    misaligned = [name for name, tensor in metadata_and_support if tensor.data_ptr() % 16 != 0]
    if misaligned:
        raise ValueError(f"MSA requires 16-byte-aligned support and layout; misaligned tensors={misaligned}.")
    if layout._workspace_positions.dtype != torch.int64:
        raise ValueError(f"MSA workspace positions must be int64, got {layout._workspace_positions.dtype}.")
    if layout._document_workspace_starts.dtype != torch.int32:
        raise ValueError(f"MSA document workspace starts must be int32, got {layout._document_workspace_starts.dtype}.")
    if layout._cu_seqlens.dtype != torch.int32 or not layout._cu_seqlens.is_contiguous():
        raise ValueError(
            "MSA packed-layout cu_seqlens must be contiguous int32, got "
            f"dtype={layout._cu_seqlens.dtype}, contiguous={layout._cu_seqlens.is_contiguous()}."
        )
    if layout._workspace_size <= 0 or layout._workspace_size % _MSA_BLOCK_SIZE != 0:
        raise ValueError(
            f"MSA packed-layout workspace size must be a positive multiple of 128, got {layout._workspace_size}."
        )
    if layout._max_seqlen <= 0:
        raise ValueError(f"MSA packed-layout max sequence length must be positive, got {layout._max_seqlen}.")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}.")


def _align_backward_tensor(
    compact: torch.Tensor,
    workspace_positions: torch.Tensor,
    workspace_size: int,
) -> torch.Tensor:
    """Scatter compact[T,H,D] via int64 workspace_positions[T] into zero-filled output[W,H,D]."""
    if workspace_positions.ndim != 1 or workspace_positions.shape[0] != compact.shape[0]:
        raise ValueError(
            f"workspace_positions{workspace_positions.shape} must have one entry per row of {compact.shape}."
        )
    workspace = compact.new_zeros((workspace_size, compact.shape[1], compact.shape[2]))
    return workspace.index_copy(0, workspace_positions, compact).contiguous()


class _MSASparseAttentionFunction(torch.autograd.Function):
    """Own flat forward state and the backward-only aligned workspace lifetime."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        layout: _MSAPackedLayout,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run aligned BF16 q[T,64,128], k/v[T,4,128], int32 q2k[4,T,16] and layout; return BF16 O[T,64,128]."""
        ctx.set_materialize_grads(False)
        cu_seqlens = layout._cu_seqlens
        max_seqlen = int(layout._max_seqlen)
        kernels = _require_msa()
        # The CSR extension lacks a CUDAGuard; bind both external launches to the tensor's device.
        with torch.cuda.device(q.device):
            row_ptr, q_indices, schedule = kernels.build_k2q_csr(
                q2k,
                cu_seqlens,
                cu_seqlens,
                _MSA_BLOCK_SIZE,
                total_k=q.shape[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                total_rows=int(layout._workspace_size) // _MSA_BLOCK_SIZE,
                qhead_per_kv=_MSA_QUERY_HEADS // _MSA_KV_HEADS,
                return_schedule=True,
            )
            out, lse = kernels.sparse_atten_func(
                q,
                k,
                v,
                row_ptr,
                q_indices,
                _MSA_TOPK_BLOCKS,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                blk_kv=_MSA_BLOCK_SIZE,
                causal=True,
                softmax_scale=float(softmax_scale),
                partial_dtype=torch.bfloat16,
                return_softmax_lse=True,
                schedule=schedule,
            )
        expected_out = (q.shape[0], _MSA_QUERY_HEADS, _MSA_HEAD_DIM)
        expected_lse = (q.shape[0], _MSA_QUERY_HEADS)
        if tuple(out.shape) != expected_out or out.dtype != torch.bfloat16:
            raise RuntimeError(
                f"MSA forward returned out {tuple(out.shape)}/{out.dtype}, expected {expected_out}/bf16."
            )
        if tuple(lse.shape) != expected_lse or lse.dtype != torch.float32:
            raise RuntimeError(
                f"MSA forward returned LSE {tuple(lse.shape)}/{lse.dtype}, expected {expected_lse}/fp32."
            )

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            row_ptr,
            q_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            layout._workspace_positions,
            layout._document_workspace_starts,
            cu_seqlens,
        )
        ctx.workspace_size = int(layout._workspace_size)
        ctx.softmax_scale = float(softmax_scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor | None) -> tuple[Any, ...]:
        """Map BF16 grad_out[T,64,128] to compact dQ[T,64,128], dK/dV[T,4,128] and three None slots."""
        (
            q,
            k,
            v,
            out,
            lse,
            row_ptr,
            q_indices,
            scheduler_metadata,
            work_count,
            workspace_positions,
            document_workspace_starts,
            cu_seqlens,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros_like(out)

        k_aligned = _align_backward_tensor(k, workspace_positions, ctx.workspace_size)
        v_aligned = _align_backward_tensor(v, workspace_positions, ctx.workspace_size)
        schedule = _MSABackwardSchedule(
            row_ptr=row_ptr,
            q_indices=q_indices,
            scheduler_metadata=scheduler_metadata,
            work_count=work_count,
            cu_seqlens=cu_seqlens,
            document_workspace_starts=document_workspace_starts,
        )
        run_msa_backward = _require_msa_backward()
        with torch.cuda.device(q.device):
            dq, dk_workspace, dv_workspace = run_msa_backward(
                q,
                k_aligned,
                v_aligned,
                grad_out.contiguous(),
                lse,
                out,
                schedule,
                softmax_scale=ctx.softmax_scale,
            )
        dk = dk_workspace.index_select(0, workspace_positions)
        dv = dv_workspace.index_select(0, workspace_positions)
        return dq, dk, dv, None, None, None


class _MSAFlatAttention(nn.Module):
    """MiniMax M3's model-private flat MSA training Adapter."""

    def __init__(self, softmax_scale: float) -> None:
        super().__init__()
        if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
            raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}.")
        self.softmax_scale = float(softmax_scale)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        *,
        layout: _MSAPackedLayout,
    ) -> torch.Tensor:
        """Run aligned BF16 q[T,64,128], k/v[T,4,128], int32 q2k[4,T,16] and layout; return BF16 O[T,64,128]."""
        if not isinstance(layout, _MSAPackedLayout):
            raise TypeError(f"layout must be an _MSAPackedLayout, got {type(layout).__name__}.")
        if torch.are_deterministic_algorithms_enabled():
            raise NotImplementedError(
                "MiniMax M3 MSA backward accumulates dK/dV with FP32 atomics and dQ with packed 16-bit atomics, "
                "so it is not bitwise deterministic; "
                "disable torch deterministic algorithms or use sparse_attn='generic'."
            )
        _validate_flat_msa_inputs(q, k, v, q2k, layout, softmax_scale=self.softmax_scale)
        return _MSASparseAttentionFunction.apply(q, k, v, q2k, layout, self.softmax_scale)


_MSA_CACHE_ARGUMENTS = ("past_key_values", "cache_position", "page_table", "seqused_k", "prefix_cache")
_MSA_CROSS_ATTENTION_ARGUMENTS = ("encoder_hidden_states", "key_value_states")


def _reject_unsupported_msa_configuration(backend: BackendConfig) -> None:
    """Reject backend selections that cannot change after MSA construction."""
    if backend.te_fp8 is not None:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports BF16 projection only; set backend.te_fp8=None or use sparse_attn='generic'."
        )
    if backend.rope_fusion:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports rope_fusion=False only: the fused BSHD rotary path uses batch row 0's "
            "positions for every row (position_ids_to_freqs_cis), which corrupts packed per-document positions. "
            "Set backend.rope_fusion=False."
        )


def _msa_cp_enabled(owner: Any) -> bool:
    """Check the apply_cp flag or attention CP mesh; runtime kwargs do not encode BSHD CP topology."""
    if getattr(owner, "_cp_enabled", False):
        return True
    cp_mesh = getattr(owner, "_cp_mesh", None)
    return cp_mesh is not None and cp_mesh.size() > 1


def _reject_unsupported_msa_runtime(attn_kwargs: Mapping[str, Any], *, cp_enabled: bool = False) -> None:
    """Reject CP/cache/THD/window/cross-attention/capture; tensor kwargs are checked only for presence."""
    if cp_enabled:
        raise NotImplementedError(
            "MiniMax M3 MSA requires cp_size=1; disable context parallelism or set backend.sparse_attn='generic'."
        )
    qkv_format = attn_kwargs.get("qkv_format", "bshd")
    if qkv_format != "bshd":
        raise NotImplementedError(
            "MiniMax M3 MSA sparse attention supports BSHD (qkv_format='bshd') only; "
            f"got {qkv_format!r}. Set backend.sparse_attn='generic' for THD."
        )
    if attn_kwargs.get("use_cache", False):
        raise NotImplementedError("MiniMax M3 MSA supports cache-free prefill training only; set use_cache=False.")
    for cache_argument in _MSA_CACHE_ARGUMENTS:
        if attn_kwargs.get(cache_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA supports cache-free flat prefill only; "
                f"got non-None {cache_argument}. Remove cache metadata or use sparse_attn='generic'."
            )
    if attn_kwargs.get("is_causal", True) is not True:
        raise NotImplementedError("MiniMax M3 MSA first supports causal self-attention only; set is_causal=True.")
    window_size = attn_kwargs.get("window_size", (-1, 0))
    full_causal_window = window_size is None or (isinstance(window_size, int) and window_size == -1)
    if isinstance(window_size, (tuple, list)):
        full_causal_window = tuple(window_size) == (-1, 0)
    if not full_causal_window:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports full causal attention only; "
            f"got window_size={window_size!r}. Disable the sliding window."
        )
    for cross_attention_argument in _MSA_CROSS_ATTENTION_ARGUMENTS:
        if attn_kwargs.get(cross_attention_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA first supports causal self-attention only; "
                f"got non-None {cross_attention_argument}. Use sparse_attn='generic' for cross-attention."
            )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "MiniMax M3 MSA does not support CUDA graph capture in the first delivery boundary; "
            "run outside capture or use sparse_attn='generic'."
        )
