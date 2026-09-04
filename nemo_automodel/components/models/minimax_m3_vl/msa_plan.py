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

"""Opaque packed-document layout for MiniMax M3 MSA flat prefill."""

from dataclasses import dataclass

import torch

_MSA_KEY_ALIGNMENT = 128


def _check_document_runs(
    doc_ids: torch.Tensor,
    batch_rows: torch.Tensor,
    is_real: torch.Tensor,
    external_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check that each positive document id occupies one run in its batch row.

    Args:
        doc_ids: Int64 tensor of shape ``[external_tokens]`` containing flattened
            document ids, with 0 denoting padding.
        batch_rows: Int64 tensor of shape ``[external_tokens]`` containing the
            batch row of each flattened token.
        is_real: Bool tensor of shape ``[external_tokens]``, true for positive
            document ids.
        external_rows: Int64 tensor of shape ``[external_tokens]`` containing
            ``arange(external_tokens)``.

    Returns:
        A pair of scalar tensors on ``doc_ids.device``. The first is true when
        every ``(batch row, document id)`` occupies one contiguous run. The
        second is the first offending flattened row, or -1 when valid.
    """
    num_external_tokens = external_rows.numel()
    if num_external_tokens < 2:
        valid = torch.ones((), dtype=torch.bool, device=external_rows.device)
        first_bad = torch.full((), -1, dtype=torch.int64, device=external_rows.device)
        return valid, first_bad

    # A distinct negative key per padding token: padding never forms a document
    # group and never collides with a positive document id.
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
    """Recover the canonical MSA document map from model inputs.

    Args:
        reference: Tensor of shape [batch, sequence, hidden] whose leading shape
            and device define the active BSHD model input.
        packed_seq_ids: Optional integer tensor of shape [batch, sequence], with
            positive document ids and 0 for padding. This is authoritative when
            present.
        attention_mask: Optional integer or bool tensor of shape [batch,
            sequence] containing indexed document ids or a keep mask, or a bool
            tensor of shape [batch, 1, sequence, sequence] containing the exact
            standard causal, same-document keep relation for every real token.
        padding_mask: Optional tensor of shape [batch, sequence], with nonzero
            values denoting padding. Used only when neither higher-priority
            source is present.

    Returns:
        Contiguous int64 tensor of shape [batch, sequence] on
        ``reference.device``, with positive document ids and 0 for padding.

    Raises:
        TypeError: If a supplied metadata value is not a tensor.
        ValueError: If a supplied tensor has an unsupported shape or dtype.
        NotImplementedError: If ``reference`` is not a BSHD model input.
    """
    if reference.dim() != 3:
        raise NotImplementedError(
            "MiniMax M3 MSA sparse attention supports BSHD input only; expected hidden states of shape "
            f"[batch, sequence, hidden], got {tuple(reference.shape)}"
        )
    batch_size, sequence_length = reference.shape[:2]
    expected_shape = (batch_size, sequence_length)

    # The attention mask stays a dense-layer input even when packed ids are
    # authoritative, so it must satisfy the MSA boundary.
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(f"attention_mask must be a tensor, got {type(attention_mask).__name__}")
        if attention_mask.dim() == 2:
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"2-D attention_mask must have shape [batch, sequence]={expected_shape}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            if attention_mask.dtype.is_floating_point or attention_mask.dtype.is_complex:
                raise ValueError(
                    "MiniMax M3 MSA accepts only an integer or bool 2-D attention_mask; "
                    f"got dtype {attention_mask.dtype}. Additive/float masks cannot uniquely encode document ids."
                )
        elif attention_mask.dim() == 4:
            expected_4d_shape = (batch_size, 1, sequence_length, sequence_length)
            if tuple(attention_mask.shape) != expected_4d_shape or attention_mask.dtype != torch.bool:
                raise ValueError(
                    "MiniMax M3 MSA accepts only a bool 4-D block-causal attention_mask of shape "
                    f"[batch, 1, sequence, sequence]={expected_4d_shape}; got shape "
                    f"{tuple(attention_mask.shape)} and dtype {attention_mask.dtype}"
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
            mask = attention_mask.to(device=reference.device)
            if mask.dtype == torch.bool:
                # Non-bool masks keep values > 1 as explicit document labels.
                return mask.bool().to(torch.int64).contiguous()
            return mask.to(torch.int64).contiguous()
        if attention_mask.dim() == 4:
            block_causal = attention_mask[:, 0]
            real_tokens = torch.diagonal(block_causal, dim1=-2, dim2=-1).to(device=reference.device)
            previous_token_visible = torch.diagonal(block_causal, offset=-1, dim1=-2, dim2=-1).to(
                device=reference.device
            )
            document_starts = torch.cat(
                (
                    real_tokens[:, :1],
                    real_tokens[:, 1:] & ~previous_token_visible,
                ),
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
            raise ValueError(
                f"padding_mask must have shape [batch, sequence]={expected_shape}, got {tuple(padding_mask.shape)}"
            )
        if padding_mask.dtype.is_complex:
            raise ValueError(f"padding_mask cannot have complex dtype {padding_mask.dtype}")
        return (~padding_mask.to(device=reference.device).bool()).to(torch.int64).contiguous()

    return torch.ones(expected_shape, dtype=torch.int64, device=reference.device)


@dataclass(frozen=True, slots=True)
class _MSALaunchMetadata:
    """Packed coordinates one MSA launch needs, derived from a packed layout.

    This is the only layout-derived value the MSA adapter consumes. It carries
    no attention semantics: canonical support stays in ``q2k`` and these fields
    only locate compact rows inside their documents and inside the temporary
    128-aligned backward workspace. Build it with
    :meth:`_MSAPackedLayout.launch_metadata`; it is valid for exactly the
    forward that produced it.

    Attributes:
        workspace_positions: Int64 tensor of shape ``[tokens]`` mapping each
            compact row to its 128-aligned workspace row.
        document_workspace_starts: Int32 tensor of shape ``[documents]`` with
            each document's first aligned workspace row.
        cu_seqlens: Contiguous int32 tensor of shape ``[documents + 1]`` with
            cumulative compact document lengths.
        total_tokens: Number of compact rows ``T``; the leading extent every
            flat Q/K/V/output tensor must match.
        workspace_size: Aligned workspace height ``W``, a positive multiple of
            128 and the leading extent of the backward-only aligned K/V.
        max_seqlen: Longest real document length in compact rows.
    """

    workspace_positions: torch.Tensor
    document_workspace_starts: torch.Tensor
    cu_seqlens: torch.Tensor
    total_tokens: int
    workspace_size: int
    max_seqlen: int


@dataclass(frozen=True, slots=True)
class _MSAPackedLayout:
    """Model-owned mapping between external, compact, and aligned MSA rows.

    The class is private so callers cross the layout seam through semantic
    operations instead of depending on workspace coordinates. The MSA adapter
    in the same model-owned package crosses it through
    :meth:`launch_metadata`, so no caller reads the private coordinate fields.
    """

    _token_rows: torch.Tensor
    _workspace_positions: torch.Tensor
    _query_doc_starts: torch.Tensor
    _document_workspace_starts: torch.Tensor
    _cu_seqlens: torch.Tensor
    _external_shape: tuple[int, int]
    _workspace_size: int
    _max_seqlen: int
    _has_multiple_documents_per_row: bool

    @classmethod
    def validate(cls, doc_ids: torch.Tensor) -> tuple[bool, bool]:
        """Validate document ids without materializing an MSA layout.

        Args:
            doc_ids: Integer tensor of shape ``[batch, sequence]`` using 0 for
                padding and positive values for document ids.

        Returns:
            ``(has_padding, has_multiple_documents_per_row)``.

        Raises:
            ValueError: If the document ids violate the MSA layout contract.
        """
        _, has_padding, has_multiple_documents = cls._prepare(doc_ids, materialize=False)
        return has_padding, has_multiple_documents

    @classmethod
    def build(cls, doc_ids: torch.Tensor) -> "_MSAPackedLayout":
        """Build one packed layout from canonical document ids.

        All data-dependent sizes and validation results travel in one probe, so
        the builder performs exactly one device-to-host transfer.

        Args:
            doc_ids: Integer tensor of shape ``[batch, sequence]`` on the
                compute device. Positive values identify documents and 0
                denotes padding. A document id must occupy exactly one
                contiguous run within a batch row. Equal ids in different
                batch rows identify different documents.

        Returns:
            An opaque layout whose tensor metadata remains on
            ``doc_ids.device``.

        Raises:
            ValueError: If ``doc_ids`` is not a non-empty rank-2 integer tensor,
                contains negative ids, contains no real token, resumes a
                document after an interruption, or exceeds the int32 coordinate
                range required by the MSA kernels.
        """
        layout, _, _ = cls._prepare(doc_ids, materialize=True)
        assert layout is not None
        return layout

    @classmethod
    def _prepare(cls, doc_ids: torch.Tensor, *, materialize: bool) -> tuple["_MSAPackedLayout | None", bool, bool]:
        """Run the shared validation probe and optionally materialize layout tensors.

        Args:
            doc_ids: Integer tensor of shape ``[batch, sequence]`` using 0 for
                padding and positive values for document ids.
            materialize: Whether to create the device tensors used by MSA.

        Returns:
            ``(layout, has_padding, has_multiple_documents_per_row)``. ``layout``
            is ``None`` when ``materialize`` is false.
        """
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
            ((run_length + _MSA_KEY_ALIGNMENT - 1) // _MSA_KEY_ALIGNMENT) * _MSA_KEY_ALIGNMENT,
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

        # Stable prefix ranks give each row its destination in a true-first
        # partition, so one scatter replaces a trailing stable sort.
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
                _has_multiple_documents_per_row=has_multiple_documents,
            ),
            has_padding,
            has_multiple_documents,
        )

    @property
    def has_padding(self) -> bool:
        """Whether the external token grid contains padding rows."""
        return self._token_rows.numel() != self._external_shape[0] * self._external_shape[1]

    @property
    def has_multiple_documents_per_row(self) -> bool:
        """Whether any external batch row contains multiple documents."""
        return self._has_multiple_documents_per_row

    def launch_metadata(self) -> _MSALaunchMetadata:
        """Expose the packed coordinates an MSA kernel launch consumes.

        Returns:
            An :class:`_MSALaunchMetadata` view of this layout. The tensor
            fields alias the layout's own tensors and must not be mutated;
            their layouts are documented on that class.
        """
        return _MSALaunchMetadata(
            workspace_positions=self._workspace_positions,
            document_workspace_starts=self._document_workspace_starts,
            cu_seqlens=self._cu_seqlens,
            total_tokens=self._token_rows.numel(),
            workspace_size=self._workspace_size,
            max_seqlen=self._max_seqlen,
        )

    def pack(self, external: torch.Tensor) -> torch.Tensor:
        """Remove padding and flatten a BSHD-aligned tensor.

        Args:
            external: Tensor of shape ``[batch, sequence, ...]`` matching this
                layout's external token grid and device.

        Returns:
            Tensor of shape ``[tokens, ...]`` in document order. If there is no
            padding, the result may alias ``external``.

        Raises:
            ValueError: If the leading shape or device does not match the layout.
        """
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
        """Restore compact rows to the external BSHD-aligned token grid.

        Args:
            packed: Tensor of shape ``[tokens, ...]`` on the layout's device.

        Returns:
            Tensor of shape ``[batch, sequence, ...]``. Real rows contain their
            packed values and padding rows are exactly zero. If there is no
            padding, the result may alias ``packed``.

        Raises:
            ValueError: If the leading size or device does not match the layout.
        """
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
        """Prepare private workspace coordinates for MSA block selection.

        Args:
            index_k: Compact post-RoPE index keys with layout
                ``[tokens, 1, index_dim]``.

        Returns:
            The aligned index keys ``[1, workspace_rows, 1, index_dim]``,
            compact-query workspace positions ``[tokens]``, and aligned
            document starts ``[tokens]``.

        Raises:
            ValueError: If the shape or device does not match the layout.
        """
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
